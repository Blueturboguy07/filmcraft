//! AI through publik API or the person's own key, desktop only. The setup flow, the credential
//! file and the HTTP calls live in the shared `craft-publik` crate (one per Craft app). This module
//! is FilmCraft's thin host:
//!
//! - the setup window on the first launch (once, after a few frames, when no other dialog shows;
//!   "Later" ends it for good; never while an agent drives the app over `--control`),
//! - Preferences ▸ AI (status, sign-in and key controls, the setup flow in place), also reached
//!   from Edit ▸ Preferences ▸ AI… and Sequence ▸ AI ▸ AI Settings…,
//! - Sequence ▸ AI ▸ Transcribe to Captions… and Voice-over from Text…, two windows.
//!
//! Nothing is sent, and nothing is charged, until the person clicks Transcribe or Generate.
//!
//! **Transcribe to Captions** renders the selected clip's audio as it plays on its track (clip
//! gain, effects and speed; the linked audio of a selected video clip), or the sequence mix from
//! In to Out, through the app's own audio renderer; resamples it to 16 kHz mono; packs it as AAC in
//! an `.m4a` with the app's own encoder and MP4 writer (about 4 KB a second, so 15 minutes fit one
//! request); and sends it through `client::transcribe`. Each segment that comes back becomes one
//! caption, at the segment's time from the start of the audio, on a new caption track, through the
//! app's own `captions.import` command (one undo step).
//!
//! **Voice-over from Text** sends the text and the chosen voice through `client::speech`. The audio
//! file goes where voice-over recordings and narrations go (Scratch Disks ▸ Captured, else next to
//! the project) and onto the timeline through the app's own `edit.pasteMedia` command: a new clip
//! at the playhead on the first free audio track (a track is added when every one is busy), one
//! undo step.
//!
//! A paid result goes in when it arrives. When it cannot (the sequence it was made for is not open,
//! playback runs, a command fails), it waits behind Place Captions / Place Voice-over.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};

use craft_publik::client::{self, Fail, Reply};
use craft_publik::ui::{Action, AiState, spawn};
use craft_publik::{Line, Provider, WHY_IT_COSTS};
use egui::RichText;
use filmcraft_engine::Session;
use filmcraft_project::{ClipId, ItemId, Project, Track};
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde_json::{Value, json};

pub use craft_publik::AppInfo;

use crate::FilmcraftApp;
use crate::menus::MenuItem;
use crate::widgets::CenteredDialog as _;

/// Sequence ▸ AI ▸ Transcribe to Captions…
pub const TRANSCRIBE: &str = "ai.transcribeCaptions";
/// Sequence ▸ AI ▸ Voice-over from Text…
pub const VOICE_OVER: &str = "ai.voiceOver";
/// Sequence ▸ AI ▸ AI Settings… (Preferences on its AI page).
pub const SETTINGS: &str = "ai.settings";
/// Edit ▸ Preferences ▸ AI… (FilmCraft ▸ Settings ▸ AI in the macOS menu bar).
pub const PREFERENCES: &str = "app.settings.ai";
/// The submenu the tools sit in, at the end of the Sequence menu (next to Transcript and Captions).
const MENU: [&str; 2] = ["Sequence", "AI"];

/// The audio a transcription sends: 16 kHz mono AAC-LC at 32 kbit/s.
pub const SEND_RATE: u32 = 16_000;
const SEND_BITRATE: u32 = 32_000;
/// The longest audio one transcription sends: 15 minutes is about 3.6 MB at [`SEND_BITRATE`],
/// under the 4.5 MB one request to publik API can carry.
pub const MAX_TRANSCRIBE_SECONDS: i64 = 15 * 60;
/// The most captions one transcription makes.
const MAX_SEGMENTS: usize = 10_000;
/// The longest text one voice-over takes (the speech line's limit), in characters.
pub const MAX_SPEECH_CHARS: usize = 4_096;
/// publik's published prices on 2026-10-19 (the `whisper-1` and `tts-1` lines of the price sheet),
/// for the estimate shown before a button. The charge line after the request is what was charged.
const TRANSCRIBE_MICROS_PER_MINUTE: i64 = 6_000;
const TRANSCRIBE_RATE: &str = "$0.006 a minute";
const SPEECH_MICROS_PER_1K_CHARS: i64 = 15_000;
const SPEECH_RATE: &str = "$0.015 for 1,000 characters";
/// The speech line's voices; the first is the crate's default.
pub const VOICES: [(&str, &str); 6] = [("alloy", "Alloy"), ("echo", "Echo"), ("fable", "Fable"), ("onyx", "Onyx"), ("nova", "Nova"), ("shimmer", "Shimmer")];
/// Name of the caption track a transcription makes (" 2", " 3"… when it is taken).
const TRACK_NAME: &str = "AI Captions";
/// File names of saved voice-overs: `AI Voice-over <n>.mp3`, in `AI Voice-overs` when the project
/// has no folder yet.
const VOICE_BASE: &str = "AI Voice-over";
const VOICE_FOLDER: &str = "AI Voice-overs";

const OFF: &str = "AI is off in this session.";
const TRANSCRIBING: &str = "Transcribing… This can take a minute.";
const GENERATING: &str = "Generating speech… This can take a minute.";
const STOPPED: &str = "The request stopped unexpectedly. Try again.";

/// AI state of the desktop app (`FilmcraftApp::ai`).
pub struct Ai {
    pub state: AiState,
    frames: u32,
    first_run_checked: bool,
    /// The setup window is wanted; it opens once no other dialog shows.
    open_setup: bool,
    setup_window: bool,
    /// Preferences ▸ AI shows the setup flow in place of its status.
    inline_setup: bool,
    transcribe_window: bool,
    voice_window: bool,
    /// Transcribe the selected clip (else the sequence from In to Out) when both are there.
    use_clip: bool,
    transcribe: Task<Receiver<Result<(Captions, Reply), Fail>>, Captions>,
    voice: Task<Receiver<Result<(VoiceOut, Reply), Fail>>, VoiceOut>,
    /// The voice-over text and voice (kept between openings of the window).
    pub text: String,
    pub voice_id: String,
}

/// One feature's request in flight, its result waiting to be placed, and what the window says.
struct Task<J, O> {
    job: Option<J>,
    waiting: Option<O>,
    note: Option<String>,
    error: Option<String>,
}

impl<J, O> Default for Task<J, O> {
    fn default() -> Self {
        Task { job: None, waiting: None, note: None, error: None }
    }
}

impl<J, O> Task<J, O> {
    fn busy(&self) -> bool {
        self.job.is_some()
    }
    fn start(&mut self, job: J) {
        self.job = Some(job);
        self.note = None;
        self.error = None;
    }
}

/// The audio a transcription is made of: one clip alone, or the sequence mix, over `start..end`
/// (sequence time).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Source {
    seq: ItemId,
    /// The audio clip to render alone; `None` = the sequence mix.
    clip: Option<ClipId>,
    start: Tick,
    end: Tick,
    /// What the window says about it.
    label: String,
}

impl Source {
    fn length(&self) -> Tick {
        (self.end - self.start).max(Tick::ZERO)
    }
}

/// A part of the transcription: seconds from the start of the audio, and its words.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Segment {
    start: f64,
    end: f64,
    text: String,
}

/// A finished transcription: where it goes and what came back.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Captions {
    seq: ItemId,
    /// Sequence time of the start of the audio.
    start: Tick,
    segments: Vec<Segment>,
    /// The whole text, for Copy Text when the service sent no times.
    text: String,
}

/// A finished voice-over: the audio file's bytes, and where it was saved once it was.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VoiceOut {
    bytes: Vec<u8>,
    ext: &'static str,
    path: Option<String>,
}

impl Ai {
    pub fn new(app: AppInfo) -> Self {
        Ai {
            state: AiState::new(app),
            frames: 0,
            first_run_checked: false,
            open_setup: false,
            setup_window: false,
            inline_setup: false,
            transcribe_window: false,
            voice_window: false,
            use_clip: true,
            transcribe: Task::default(),
            voice: Task::default(),
            text: String::new(),
            voice_id: VOICES[0].0.to_string(),
        }
    }

    /// Do not open the setup window on this launch (an agent drives the app over `--control`).
    /// Preferences ▸ AI and the AI windows still offer it.
    pub fn skip_first_run(&mut self) {
        self.first_run_checked = true;
    }

    /// A transcription or a voice-over is on its way.
    pub fn working(&self) -> bool {
        self.transcribe.busy() || self.voice.busy()
    }
}

/// `PUBLIK_SELF_TEST=1`: provision for real and print one line (publikhq.com/skills/setup.md).
pub fn self_test(app: &AppInfo) -> String {
    craft_publik::self_test(app)
}

fn local(message: impl Into<String>) -> Fail {
    Fail { kind: "local".into(), message: message.into(), ..Fail::default() }
}

/// Run a background request; a panic in it becomes a failure the window shows (AGENTS.md §0.6).
fn guarded<T>(work: impl FnOnce() -> Result<T, Fail>) -> Result<T, Fail> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_else(|_| Err(local(STOPPED)))
}

// ---- once a frame -------------------------------------------------------------------------------

/// Once a frame, before drawing: take finished requests (and place what they made), and open the
/// setup window on the first launch once no other dialog asks for the person's attention.
pub fn frame(app: &mut FilmcraftApp, _ctx: &egui::Context) {
    let quiet = app.dialog.is_none() && app.ui.settings.is_none() && app.ui.extras.dialog.is_none() && app.ui_error.is_none();
    let Some(ai) = app.ai.as_mut() else { return };
    ai.state.poll();
    let captions = take(&mut ai.transcribe, &mut ai.state);
    let voice = take(&mut ai.voice, &mut ai.state);
    ai.frames = ai.frames.saturating_add(1);
    if !ai.first_run_checked && ai.frames > 3 {
        ai.first_run_checked = true;
        ai.open_setup = ai.state.wants_first_run() || ai.state.owes_first_card();
    }
    if ai.open_setup && quiet {
        ai.open_setup = false;
        open_setup(ai);
    }
    if let Some(c) = captions {
        arrived_captions(app, c);
    }
    if let Some(v) = voice {
        arrived_voice(app, v);
    }
}

/// A finished request, if one came in: its charge noted, its failure shown.
fn take<O>(task: &mut Task<Receiver<Result<(O, Reply), Fail>>, O>, state: &mut AiState) -> Option<O> {
    let rx = task.job.as_ref()?;
    let r = match rx.try_recv() {
        Ok(r) => r,
        Err(TryRecvError::Empty) => return None,
        Err(TryRecvError::Disconnected) => Err(local(STOPPED)),
    };
    task.job = None;
    match r {
        Ok((out, reply)) => {
            state.note_reply(&reply);
            Some(out)
        }
        Err(f) => {
            // a 402 or a retired key is the result bar's (one message, one link); the rest is ours
            state.note_fail(&f);
            task.error = state.notice.is_none().then(|| f.message.clone());
            None
        }
    }
}

fn open_setup(ai: &mut Ai) {
    ai.state.begin();
    ai.setup_window = true;
}

/// Captions came in: they go in at once when their sequence is in front; else they wait behind
/// Place Captions.
fn arrived_captions(app: &mut FilmcraftApp, c: Captions) {
    if c.segments.is_empty() {
        let Some(ai) = app.ai.as_mut() else { return };
        let note = if c.text.is_empty() {
            "The service found no speech in this audio. No captions were made."
        } else {
            "The service sent the text without times, so FilmCraft cannot make captions from it. Copy the text instead."
        };
        ai.transcribe.note = Some(note.into());
        if !c.text.is_empty() {
            ai.transcribe.waiting = Some(c);
        }
        if !ai.transcribe_window {
            app.ui.status = format!("{note} (Sequence ▸ AI ▸ Transcribe to Captions)");
        }
        return;
    }
    let placed = place_captions(app, &c);
    let window = app.ai.as_ref().is_some_and(|ai| ai.transcribe_window);
    match placed {
        Ok(note) => {
            if !window {
                app.ui.status = note.clone();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.transcribe.note = Some(note);
            }
        }
        Err(why) => {
            if !window {
                app.ui.status = "The captions are ready. Open Sequence ▸ AI ▸ Transcribe to Captions and click Place Captions.".into();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.transcribe.note = Some(waiting_note("The captions are ready, but they are not in the sequence yet.", &why, "Click Place Captions."));
                ai.transcribe.waiting = Some(c);
            }
        }
    }
}

fn arrived_voice(app: &mut FilmcraftApp, mut v: VoiceOut) {
    let placed = place_voice(app, &mut v);
    let window = app.ai.as_ref().is_some_and(|ai| ai.voice_window);
    match placed {
        Ok(note) => {
            if !window {
                app.ui.status = note.clone();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.voice.note = Some(note);
            }
        }
        Err(why) => {
            if !window {
                app.ui.status = "The voice-over is ready. Open Sequence ▸ AI ▸ Voice-over from Text and click Place Voice-over.".into();
            }
            if let Some(ai) = app.ai.as_mut() {
                ai.voice.note = Some(waiting_note("The voice-over is ready, but it is not on the timeline yet.", &why, "Click Place Voice-over."));
                ai.voice.waiting = Some(v);
            }
        }
    }
}

/// "<ready>. <why it is not in yet>. <what to do>"
fn waiting_note(ready: &str, why: &str, todo: &str) -> String {
    let mut s = ready.to_string();
    let why = why.trim().trim_end_matches('.');
    if !why.is_empty() {
        s.push(' ');
        s.push_str(why);
        s.push('.');
    }
    s.push(' ');
    s.push_str(todo);
    s
}

// ---- the menu -----------------------------------------------------------------------------------

/// Sequence ▸ AI (Transcribe to Captions…, Voice-over from Text…, AI Settings…) at the end of the
/// Sequence menu, and Edit ▸ Preferences ▸ AI…, where AI is on.
pub fn menu_items(app: &FilmcraftApp, items: &mut Vec<MenuItem>) {
    if app.ai.is_none() {
        return;
    }
    let item = |id: &str, label: &str, path: &[&str]| MenuItem {
        id: id.into(),
        label: app.ui.language.tr(label).to_string(),
        path: path.iter().map(|s| s.to_string()).collect(),
        shortcut: app.session.shortcuts.primary(id),
        enabled: is_enabled(app, id).unwrap_or(false),
        checked: None,
    };
    let at = items.iter().rposition(|i| i.path.first().map(String::as_str) == Some(MENU[0])).map_or(items.len(), |i| i + 1);
    let rows = [(TRANSCRIBE, "Transcribe to Captions…"), (VOICE_OVER, "Voice-over from Text…"), (SETTINGS, "AI Settings…")];
    for (k, (id, label)) in rows.into_iter().enumerate() {
        items.insert(at + k, item(id, label, &MENU));
    }
    let at = items.iter().rposition(|i| i.id.starts_with("app.settings.")).map_or(items.len(), |i| i + 1);
    items.insert(at, item(PREFERENCES, "AI…", &["Edit", "Preferences"]));
}

/// Enabled state of an AI menu item (`None` for other ids).
pub fn is_enabled(app: &FilmcraftApp, id: &str) -> Option<bool> {
    let on = app.ai.is_some();
    match id {
        TRANSCRIBE | VOICE_OVER => Some(on && app.session.active_sequence().is_some()),
        SETTINGS | PREFERENCES => Some(on),
        _ => None,
    }
}

/// Run an AI menu item: open its window (`None` for other ids). Nothing is sent here.
pub fn route(app: &mut FilmcraftApp, id: &str) -> Option<Result<Value, String>> {
    if !matches!(id, TRANSCRIBE | VOICE_OVER | SETTINGS) {
        return None;
    }
    if app.ai.is_none() {
        return Some(Err(OFF.into()));
    }
    if id == SETTINGS {
        return Some(crate::panels::settings::open(app, crate::panels::settings::AI_PAGE));
    }
    if app.session.active_sequence().is_none() {
        return Some(Err("Open a sequence first.".into()));
    }
    let ai = app.ai.as_mut()?;
    if id == TRANSCRIBE {
        ai.transcribe_window = true;
        // the clip when one is selected, else In to Out
        ai.use_clip = true;
        Some(Ok(json!({"dialog": "aiTranscribe"})))
    } else {
        ai.voice_window = true;
        Some(Ok(json!({"dialog": "aiVoiceOver"})))
    }
}

// ---- what a transcription sends -----------------------------------------------------------------

/// The selected clip's audio: the first selected clip on an audio track, else the audio clip linked
/// to the first selected video clip.
pub(crate) fn clip_source(s: &Session) -> Option<Source> {
    let seq_id = s.state.active_sequence?;
    let seq = s.project.sequence(seq_id)?;
    let audio_of = |c: ClipId| -> Option<(usize, ClipId)> {
        let (track, item) = seq.find_item(c)?;
        if let Some(k) = seq.audio_tracks.iter().position(|t| t.id == track) {
            return Some((k, item.id));
        }
        let link = item.link?;
        seq.audio_tracks.iter().enumerate().find_map(|(k, t)| t.items.iter().find(|i| i.link == Some(link)).map(|i| (k, i.id)))
    };
    let (k, clip) = s.state.selection.iter().find_map(|c| audio_of(*c))?;
    let (_, item) = seq.find_item(clip)?;
    let name = if item.name.trim().is_empty() { "Clip".to_string() } else { item.name.clone() };
    Some(Source { seq: seq_id, clip: Some(clip), start: item.start, end: item.end(), label: format!("The selected clip: {name} on A{}.", k + 1) })
}

/// The sequence mix from In to Out (the Out frame included), when both are marked.
pub(crate) fn range_source(s: &Session) -> Option<Source> {
    let seq_id = s.state.active_sequence?;
    let seq = s.project.sequence(seq_id)?;
    let (start, out) = (seq.mark_in?, seq.mark_out?);
    let end = out + seq.settings.frame_rate.frame_duration();
    (end > start && start >= Tick::ZERO).then(|| Source {
        seq: seq_id,
        clip: None,
        start,
        end,
        label: format!("The sequence audio from In ({}) to Out ({}).", clock(start.seconds()), clock(end.seconds())),
    })
}

/// The language the transcription asks for: Preferences ▸ Media Analysis & Transcription's default
/// language, or none (the service detects it) when auto-detect is on.
fn language(s: &Session) -> String {
    let ma = &s.prefs.media_analysis;
    let code = ma.default_language.trim();
    if ma.language_auto_detect || code.is_empty() || code.len() > 3 || !code.chars().all(|c| c.is_ascii_alphabetic()) {
        String::new()
    } else {
        code.to_ascii_lowercase()
    }
}

/// `m:ss` (or `h:mm:ss`) for a length in seconds.
pub(crate) fn clock(seconds: f64) -> String {
    let s = if seconds.is_finite() { seconds.max(0.0).round() as i64 } else { 0 };
    let (h, m, sec) = (s / 3600, (s / 60) % 60, s % 60);
    if h > 0 { format!("{h}:{m:02}:{sec:02}") } else { format!("{m}:{sec:02}") }
}

/// The estimate before Transcribe.
pub(crate) fn transcribe_estimate(seconds: f64) -> String {
    let micros = (seconds.max(0.0) * TRANSCRIBE_MICROS_PER_MINUTE as f64 / 60.0).ceil() as i64;
    format!(
        "publik API charges for each minute of audio, about {TRANSCRIBE_RATE}. This audio costs about {}. The exact charge shows after the request.",
        craft_publik::dollars(micros)
    )
}

/// The estimate before Generate.
pub(crate) fn speech_estimate(chars: usize) -> String {
    let chars = i64::try_from(chars).unwrap_or(i64::MAX);
    let micros = chars.saturating_mul(SPEECH_MICROS_PER_1K_CHARS).saturating_add(999) / 1000;
    format!(
        "publik API charges for each character, about {SPEECH_RATE}. This text costs about {}. The exact charge shows after the request.",
        craft_publik::dollars(micros)
    )
}

/// The cost lines before a button that sends: publik's estimate and why it costs, or who bills an
/// own key.
fn cost_lines(state: &AiState, provider: &Provider, estimate: String) -> Vec<String> {
    match provider {
        Provider::Publik { .. } => {
            let why = match &state.credential {
                Some(c) if !c.cost_sentence.trim().is_empty() => c.cost_sentence.clone(),
                _ => WHY_IT_COSTS.to_string(),
            };
            vec![estimate, why]
        }
        Provider::Own { service, .. } => vec![format!("Your own key's service ({}) bills this request, not publik.", service.name())],
        Provider::NotSetUp => Vec::new(),
    }
}

/// A track holding only `clip` of its track (no transitions): the clip's audio alone.
fn solo_track(project: &Project, seq: ItemId, clip: ClipId) -> Option<Track> {
    let q = project.sequence(seq)?;
    let (tid, _) = q.find_item(clip)?;
    let mut t = q.audio_tracks.iter().find(|t| t.id == tid)?.clone();
    t.items.retain(|i| i.id == clip);
    t.transitions.clear();
    Some(t)
}

/// Mono average of planar channels, `n` samples (missing samples read as silence).
fn downmix(channels: &[Vec<f32>], n: usize) -> Vec<f32> {
    let k = 1.0 / channels.len().max(1) as f32;
    (0..n).map(|i| channels.iter().map(|c| c.get(i).copied().unwrap_or(0.0)).sum::<f32>() * k).collect()
}

/// The audio of `src` as an `.m4a`: rendered by the app's mixer at the sequence rate, one second
/// at a time, resampled to [`SEND_RATE`] mono and encoded as AAC.
pub(crate) fn encode_source(project: &Project, src: &Source, sources: &dyn filmcraft_render::SourceProvider) -> Result<Vec<u8>, String> {
    let seq = project.sequence(src.seq).ok_or("the sequence was deleted")?;
    let sr = seq.settings.sample_rate.max(1);
    let first = src.start.to_units_floor(i64::from(sr));
    let total = src.length().to_units_floor(i64::from(SEND_RATE)).max(0);
    let solo = match src.clip {
        Some(c) => Some(solo_track(project, src.seq, c).ok_or("the clip was deleted")?),
        None => None,
    };
    let render = |start: i64, n: usize| -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        let b = match &solo {
            Some(t) => filmcraft_render::audio::track_input(project, t, start, n, sr, sources),
            None => filmcraft_render::audio::mix_sequence(project, seq, start, n, sources),
        };
        downmix(&b.channels, n)
    };
    // `pull` reads source frames from just before the start; time before zero is silence
    let mut pull = |s0: i64, n: usize| -> Vec<Vec<f32>> {
        let at = first.saturating_add(s0);
        let quiet = usize::try_from(at.saturating_neg()).unwrap_or(0).min(n);
        let mut mono = vec![0.0f32; quiet];
        mono.extend(render(at.max(0), n - quiet));
        vec![mono]
    };
    let mut resampler = filmcraft_audio_dsp::resample::StreamResampler::new(sr, SEND_RATE);
    let mut m4a = M4a::new()?;
    let mut out = 0i64;
    while out < total {
        let n = usize::try_from((total - out).min(i64::from(SEND_RATE))).unwrap_or(0);
        if n == 0 {
            break;
        }
        let block = resampler.process(out, n, &mut pull);
        m4a.push(block.first().map(Vec::as_slice).unwrap_or_default())?;
        out = out.saturating_add(n as i64);
    }
    m4a.finish()
}

/// AAC-LC in an MPEG-4 audio file, written in memory.
struct M4a {
    enc: filmcraft_aac::Encoder,
    mux: filmcraft_isobmff::Mp4Writer<std::io::Cursor<Vec<u8>>>,
    track: usize,
}

impl M4a {
    fn new() -> Result<Self, String> {
        let enc = filmcraft_aac::Encoder::new(filmcraft_aac::EncoderConfig::cbr(SEND_RATE, 1, SEND_BITRATE)).map_err(|e| e.to_string())?;
        let entry = filmcraft_isobmff::SampleEntry::aac(enc.audio_specific_config(), 1, SEND_RATE);
        let opts = filmcraft_isobmff::WriterOptions::new(filmcraft_isobmff::Brand::Mp4);
        let mut mux = filmcraft_isobmff::Mp4Writer::new(std::io::Cursor::new(Vec::new()), opts).map_err(|e| e.to_string())?;
        let mut cfg = filmcraft_isobmff::TrackConfig::new(entry, SEND_RATE);
        // the encoder delay, signalled by an edit list so the times line up with the sequence
        cfg.media_start = Some(i64::from(enc.priming_samples()));
        let track = mux.add_track(cfg).map_err(|e| e.to_string())?;
        Ok(M4a { enc, mux, track })
    }

    fn write(&mut self, units: Vec<Vec<u8>>) -> Result<(), String> {
        for au in units {
            let sample = filmcraft_isobmff::WriteSample { data: &au, duration: 1024, composition_offset: 0, is_sync: true };
            self.mux.write_sample(self.track, sample).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn push(&mut self, mono: &[f32]) -> Result<(), String> {
        let units = self.enc.encode(&[mono]);
        self.write(units)
    }

    fn finish(mut self) -> Result<Vec<u8>, String> {
        let units = self.enc.flush();
        self.write(units)?;
        Ok(self.mux.finish().map_err(|e| e.to_string())?.into_inner())
    }
}

/// The segments of a transcription (`client::transcribe` hands them over as JSON in
/// `Reply::bytes`), clipped to the audio's `length`; segments without time or words are dropped.
pub(crate) fn parse_segments(bytes: &[u8], length: Tick) -> Vec<Segment> {
    let max = length.seconds().max(0.0);
    let v: Value = serde_json::from_slice(bytes).unwrap_or(Value::Null);
    let Some(list) = v.as_array() else { return Vec::new() };
    list.iter()
        .filter_map(|s| {
            let (start, end) = (s.get("start")?.as_f64()?, s.get("end")?.as_f64()?);
            let text = clean_text(s.get("text")?.as_str()?);
            if !start.is_finite() || !end.is_finite() || text.is_empty() {
                return None;
            }
            let start = start.clamp(0.0, max);
            let end = end.clamp(start, max);
            (end > start).then_some(Segment { start, end, text })
        })
        .take(MAX_SEGMENTS)
        .collect()
}

/// One line of caption text: runs of white space (line breaks too) become one space.
fn clean_text(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `HH:MM:SS,mmm` for milliseconds.
fn srt_time(ms: i64) -> String {
    let ms = ms.max(0);
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000)
}

/// The segments as an SRT caption file, at sequence time (`offset` is where the audio starts).
pub(crate) fn srt(segments: &[Segment], offset: Tick) -> String {
    let base = i128::from(offset.0.max(0)) * 1000 / i128::from(TICKS_PER_SECOND);
    let base = i64::try_from(base).unwrap_or(i64::MAX / 2);
    let at = |s: f64| base.saturating_add((s * 1000.0).round() as i64);
    let mut out = String::new();
    for (i, seg) in segments.iter().enumerate() {
        out.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, srt_time(at(seg.start)), srt_time(at(seg.end)), seg.text));
    }
    out
}

/// `AI Captions`, or `AI Captions <n>` when a caption track of the sequence has the name.
fn track_name(s: &Session, seq: ItemId) -> String {
    let taken = |n: &str| s.project.sequence(seq).is_some_and(|q| q.caption_tracks.iter().any(|t| t.name == n));
    if !taken(TRACK_NAME) {
        return TRACK_NAME.into();
    }
    (2..1000u32).map(|k| format!("{TRACK_NAME} {k}")).find(|n| !taken(n)).unwrap_or_else(|| TRACK_NAME.into())
}

/// Temporary files of one request (the audio sent, the caption file imported), removed with it.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("filmcraft-ai-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Scratch(dir))
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Render, pack and send the audio; read the segments that come back. Runs on a thread.
fn transcribe_work(
    provider: &Provider,
    project: &Project,
    sources: &dyn filmcraft_render::SourceProvider,
    src: &Source,
    language: &str,
) -> Result<(Captions, Reply), Fail> {
    let m4a = encode_source(project, src, sources).map_err(|e| local(format!("FilmCraft could not prepare the audio: {e}")))?;
    let scratch = Scratch::new().map_err(local)?;
    let path = scratch.write("audio.m4a", &m4a).map_err(local)?;
    let reply = client::transcribe(provider, &path, language)?;
    let segments = parse_segments(&reply.bytes, src.length());
    let text = clean_text(&reply.text);
    Ok((Captions { seq: src.seq, start: src.start, segments, text }, reply))
}

/// Send the audio of `src` (the Transcribe button).
fn start_transcribe(app: &mut FilmcraftApp, ctx: &egui::Context, src: Source) -> Result<(), String> {
    let seconds = src.length().seconds();
    if seconds <= 0.0 {
        return Err("The audio is empty.".into());
    }
    if seconds > MAX_TRANSCRIBE_SECONDS as f64 {
        return Err(too_long(seconds));
    }
    let project = app.session.project.clone();
    let sources = app.session.media.provider(project.clone(), app.session.services.clone());
    let language = language(&app.session);
    let ai = app.ai.as_mut().ok_or(OFF)?;
    let provider = ai.state.provider();
    if !provider.supports(Line::Transcribe) {
        return Err(Fail::unsupported(Line::Transcribe).message);
    }
    let rx = spawn(ctx, move || guarded(|| transcribe_work(&provider, &project, &sources, &src, &language)));
    ai.transcribe.start(rx);
    Ok(())
}

fn too_long(seconds: f64) -> String {
    format!(
        "This audio is {} long. One request can send {} at most. Mark a shorter range with In and Out.",
        clock(seconds),
        clock(MAX_TRANSCRIBE_SECONDS as f64)
    )
}

/// The captions on a new caption track of their sequence, through `captions.import` (one undo
/// step). Returns what the window says.
pub(crate) fn place_captions(app: &mut FilmcraftApp, c: &Captions) -> Result<String, String> {
    if app.session.project.sequence(c.seq).is_none() {
        return Err("The sequence that the audio came from was deleted".into());
    }
    if app.session.state.active_sequence != Some(c.seq) {
        return Err("Open the sequence that the audio came from".into());
    }
    if app.playback.playing {
        return Err("Stop playback first".into());
    }
    if c.segments.is_empty() {
        return Err("There are no captions to place".into());
    }
    let name = track_name(&app.session, c.seq);
    let scratch = Scratch::new()?;
    let path = scratch.write(&format!("{name}.srt"), srt(&c.segments, c.start).as_bytes())?;
    let r = app.session.execute("captions.import", json!({"path": path.to_string_lossy(), "name": name})).map_err(|e| e.to_string())?;
    let n = r["captions"].as_u64().unwrap_or(c.segments.len() as u64);
    Ok(format!("{n} captions are on the new caption track \"{name}\". Edit ▸ Undo removes them."))
}

fn place_waiting_captions(app: &mut FilmcraftApp) {
    let Some(c) = app.ai.as_mut().and_then(|ai| ai.transcribe.waiting.take()) else { return };
    let placed = place_captions(app, &c);
    let Some(ai) = app.ai.as_mut() else { return };
    match placed {
        Ok(note) => ai.transcribe.note = Some(note),
        Err(why) => {
            ai.transcribe.note = Some(waiting_note("The captions are ready, but they are not in the sequence yet.", &why, "Click Place Captions."));
            ai.transcribe.waiting = Some(c);
        }
    }
}

// ---- voice-over ---------------------------------------------------------------------------------

/// The file type of audio bytes, by their first bytes (the speech line answers MP3).
fn audio_ext(bytes: &[u8]) -> &'static str {
    if bytes.starts_with(b"RIFF") {
        "wav"
    } else if bytes.starts_with(b"OggS") {
        "ogg"
    } else if bytes.starts_with(b"fLaC") {
        "flac"
    } else {
        "mp3"
    }
}

fn start_voice(app: &mut FilmcraftApp, ctx: &egui::Context) -> Result<(), String> {
    let ai = app.ai.as_mut().ok_or(OFF)?;
    let text = ai.text.trim().to_string();
    if text.is_empty() {
        return Err("Type the text first.".into());
    }
    if text.chars().count() > MAX_SPEECH_CHARS {
        return Err(format!("The text is too long. One request takes {MAX_SPEECH_CHARS} characters at most."));
    }
    let provider = ai.state.provider();
    if !provider.supports(Line::Speech) {
        return Err(Fail::unsupported(Line::Speech).message);
    }
    let voice = VOICES.iter().find(|(id, _)| *id == ai.voice_id).map_or(VOICES[0].0, |(id, _)| *id);
    let rx = spawn(ctx, move || {
        guarded(|| {
            let mut reply = client::speech(&provider, &text, voice)?;
            let bytes = std::mem::take(&mut reply.bytes);
            if bytes.is_empty() {
                return Err(local("The service sent no audio. Try again."));
            }
            Ok((VoiceOut { ext: audio_ext(&bytes), bytes, path: None }, reply))
        })
    });
    ai.voice.start(rx);
    Ok(())
}

/// Where generated media goes, as for voice-over recordings and narrations: Project Settings ▸
/// Scratch Disks ▸ Captured, else next to the project, else `fallback` in the data folder.
fn media_dir(s: &Session, fallback: &str) -> PathBuf {
    if let Some(d) =
        filmcraft_engine::project_tools::trusted_scratch(s.project.settings.scratch.captured.as_deref(), s.path.as_deref(), &s.prefs.auto_save.scratch_folders)
    {
        return d;
    }
    if let Some(d) = s.path.as_deref().and_then(|p| Path::new(p).parent()).filter(|d| !d.as_os_str().is_empty()) {
        return d.to_path_buf();
    }
    let base = s.prefs_path.as_ref().and_then(|p| p.parent()).map(Path::to_path_buf).unwrap_or_else(|| std::env::temp_dir().join("FilmCraft"));
    base.join(fallback)
}

/// `<dir>/<base> <n>.<ext>` with the first `n` used by neither a project item nor a file.
fn unique_path(s: &Session, dir: &Path, base: &str, ext: &str) -> Result<String, String> {
    let names: std::collections::HashSet<&str> = s.project.items.values().map(|i| i.name.as_str()).collect();
    (1..=99_999u32)
        .map(|k| format!("{base} {k}.{ext}"))
        .map(|file| (dir.join(&file).to_string_lossy().into_owned(), file))
        .find(|(path, file)| !names.contains(file.as_str()) && s.services.file_size(path).is_err())
        .map(|(path, _)| path)
        .ok_or_else(|| format!("{}: no free file name for \"{base} <n>.{ext}\"", dir.display()))
}

/// The voice-over saved and on the timeline at the playhead through `edit.pasteMedia` (one undo
/// step). The saved path is kept, so a retry places the same file.
pub(crate) fn place_voice(app: &mut FilmcraftApp, v: &mut VoiceOut) -> Result<String, String> {
    if app.session.active_sequence().is_none() {
        return Err("Open a sequence first".into());
    }
    if app.session.voiceover.recording() {
        return Err("Stop the voice-over recording first".into());
    }
    if app.playback.playing {
        return Err("Stop playback first".into());
    }
    let path = match v.path.clone().filter(|p| app.session.services.file_size(p).is_ok()) {
        Some(p) => p,
        None => {
            let dir = media_dir(&app.session, VOICE_FOLDER);
            let path = unique_path(&app.session, &dir, VOICE_BASE, v.ext)?;
            std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            app.session.services.write_file(&path, &v.bytes).map_err(|e| format!("{path}: {e}"))?;
            v.path = Some(path.clone());
            path
        }
    };
    // into the bin the Project panel shows, as Paste does
    let bin = app.import_bin().0;
    app.session.execute("edit.pasteMedia", json!({"paths": [path], "bin": bin})).map_err(|e| e.to_string())?;
    let file = Path::new(&path).file_name().map_or_else(|| path.clone(), |f| f.to_string_lossy().into_owned());
    Ok(format!("The voice-over is on the timeline at the playhead ({file}). Edit ▸ Undo removes it."))
}

fn place_waiting_voice(app: &mut FilmcraftApp) {
    let Some(mut v) = app.ai.as_mut().and_then(|ai| ai.voice.waiting.take()) else { return };
    let placed = place_voice(app, &mut v);
    let Some(ai) = app.ai.as_mut() else { return };
    match placed {
        Ok(note) => ai.voice.note = Some(note),
        Err(why) => {
            ai.voice.note = Some(waiting_note("The voice-over is ready, but it is not on the timeline yet.", &why, "Click Place Voice-over."));
            ai.voice.waiting = Some(v);
        }
    }
}

// ---- the windows --------------------------------------------------------------------------------

/// What a click in a window asks for, done after the window is drawn.
#[derive(Default)]
struct Clicks {
    send: bool,
    place: bool,
    discard: bool,
    setup: bool,
    url: Option<String>,
}

type Elems = Vec<(String, egui::Rect, String)>;

/// The AI windows, drawn with the app's other dialogs.
pub fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    setup_window(app, ctx);
    transcribe_window(app, ctx);
    voice_window(app, ctx);
}

fn setup_window(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let Some(ai) = app.ai.as_mut() else { return };
    if !ai.setup_window {
        return;
    }
    let mut open = true;
    let mut action = Action::None;
    let resp = egui::Window::new("Set Up AI")
        .id(egui::Id::new("ai-setup"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(460.0)
        .centered(ctx)
        .show(ctx, |ui| action = ai.state.setup_ui(ui));
    match action {
        Action::Close => ai.setup_window = false,
        Action::OpenUrl(url) => crate::links::open(app, ctx, &url),
        Action::None => {}
    }
    if !open && let Some(ai) = app.ai.as_mut() {
        ai.setup_window = false;
    }
    if let Some(r) = resp {
        app.auto.add("ai.setup.window", r.response.rect, "Set Up AI");
    }
}

fn dim(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).small().weak()).wrap());
}

fn warn(ui: &mut egui::Ui, text: &str) {
    let c = ui.visuals().warn_fg_color;
    ui.add(egui::Label::new(RichText::new(text).color(c)).wrap());
}

/// A button that records its automation id.
fn button(ui: &mut egui::Ui, elems: &mut Elems, id: &str, label: &str, enabled: bool) -> bool {
    let r = ui.add_enabled(enabled, egui::Button::new(label));
    elems.push((id.into(), r.rect, label.into()));
    r.clicked()
}

/// What a window shows when AI is not set up: why, and Set Up AI….
fn needs_setup(ui: &mut egui::Ui, elems: &mut Elems, prefix: &str, what: &str, clicks: &mut Clicks) {
    ui.add(egui::Label::new(format!("{what} needs AI. Set it up first: publik API (no key to paste) or your own key.")).wrap());
    ui.add_space(8.0);
    if button(ui, elems, &format!("{prefix}.setup"), "Set Up AI…", true) {
        clicks.setup = true;
    }
}

/// Under the buttons: the request in flight, its note or error, the result bar (a 402's one link,
/// or the charge) and what is in use.
fn status_rows<J, O>(ui: &mut egui::Ui, state: &mut AiState, task: &Task<J, O>, busy_text: &str, clicks: &mut Clicks) {
    if task.busy() {
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(RichText::new(busy_text).weak());
        });
    }
    if let Some(note) = &task.note {
        ui.add_space(6.0);
        ui.add(egui::Label::new(note.as_str()).wrap());
    }
    if let Some(e) = &task.error {
        ui.add_space(6.0);
        let c = ui.visuals().error_fg_color;
        ui.add(egui::Label::new(RichText::new(e.as_str()).color(c)).wrap());
    }
    ui.add_space(6.0);
    if let Action::OpenUrl(url) = state.result_bar(ui) {
        clicks.url = Some(url);
    }
    dim(ui, &state.status_line());
}

fn transcribe_window(app: &mut FilmcraftApp, ctx: &egui::Context) {
    if !app.ai.as_ref().is_some_and(|ai| ai.transcribe_window) {
        return;
    }
    let clip = clip_source(&app.session);
    let range = range_source(&app.session);
    let has_seq = app.session.active_sequence().is_some();
    let lang = language(&app.session);
    let Some(ai) = app.ai.as_mut() else { return };
    let provider = ai.state.provider();
    let mut clicks = Clicks::default();
    let mut elems: Elems = Vec::new();
    let mut open = true;
    let mut chosen: Option<Source> = None;
    let resp = egui::Window::new("Transcribe to Captions")
        .id(egui::Id::new("ai-transcribe"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(460.0)
        .centered(ctx)
        .show(ctx, |ui| {
            ui.set_max_width(460.0);
            if !provider.is_set_up() {
                needs_setup(ui, &mut elems, "ai.transcribe", "Transcribe to Captions", &mut clicks);
                return;
            }
            let (busy, waiting) = (ai.transcribe.busy(), ai.transcribe.waiting.is_some());
            ui.add(
                egui::Label::new(
                    "Transcribe to Captions sends the audio to the AI service. The text comes back as captions on a new caption track, one caption for each spoken part.",
                )
                .wrap(),
            );
            dim(ui, "FilmCraft sends the audio as 16 kHz mono sound. Nothing is sent until you click Transcribe.");
            ui.add_space(8.0);
            if clip.is_some() && range.is_some() {
                ui.add_enabled_ui(!busy && !waiting, |ui| {
                    ui.horizontal(|ui| {
                        let r = ui.radio(ai.use_clip, "The selected clip");
                        elems.push(("ai.transcribe.source.clip".into(), r.rect, "The selected clip".into()));
                        if r.clicked() {
                            ai.use_clip = true;
                        }
                        let r = ui.radio(!ai.use_clip, "The sequence from In to Out");
                        elems.push(("ai.transcribe.source.range".into(), r.rect, "The sequence from In to Out".into()));
                        if r.clicked() {
                            ai.use_clip = false;
                        }
                    });
                });
            }
            chosen = if ai.use_clip { clip.clone().or_else(|| range.clone()) } else { range.clone().or_else(|| clip.clone()) };
            let mut can_send = !busy && !waiting && provider.supports(Line::Transcribe);
            match &chosen {
                Some(src) => {
                    ui.add(egui::Label::new(src.label.as_str()).wrap());
                    let seconds = src.length().seconds();
                    ui.label(format!("Length: {}.", clock(seconds)));
                    if seconds > MAX_TRANSCRIBE_SECONDS as f64 {
                        warn(ui, &too_long(seconds));
                        can_send = false;
                    }
                    if seconds <= 0.0 {
                        can_send = false;
                    }
                    if !provider.supports(Line::Transcribe) {
                        warn(ui, &Fail::unsupported(Line::Transcribe).message);
                    }
                    ui.add_space(4.0);
                    if lang.is_empty() {
                        dim(ui, "Language: the service finds it from the audio (Preferences ▸ Media Analysis & Transcription).");
                    } else {
                        dim(ui, &format!("Language: {lang}. You can change it in Preferences ▸ Media Analysis & Transcription."));
                    }
                    for line in cost_lines(&ai.state, &provider, transcribe_estimate(seconds)) {
                        dim(ui, &line);
                    }
                }
                None => {
                    warn(ui, if has_seq { "Select a clip that has audio, or mark In and Out in the sequence." } else { "Open a sequence first." });
                    can_send = false;
                }
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if let Some(w) = &ai.transcribe.waiting {
                    let can_place = !w.segments.is_empty();
                    if button(ui, &mut elems, "ai.transcribe.place", "Place Captions", can_place) {
                        clicks.place = true;
                    }
                    if !w.text.is_empty() && button(ui, &mut elems, "ai.transcribe.copy", "Copy Text", true) {
                        ui.ctx().copy_text(w.text.clone());
                    }
                    if button(ui, &mut elems, "ai.transcribe.discard", "Discard", true) {
                        clicks.discard = true;
                    }
                } else if button(ui, &mut elems, "ai.transcribe.run", "Transcribe", can_send) {
                    clicks.send = true;
                }
            });
            status_rows(ui, &mut ai.state, &ai.transcribe, TRANSCRIBING, &mut clicks);
        });
    if let Some(r) = &resp {
        app.auto.add("ai.transcribe.window", r.response.rect, "Transcribe to Captions");
    }
    for (id, rect, label) in &elems {
        app.auto.add(id, *rect, label);
    }
    if !open && let Some(ai) = app.ai.as_mut() {
        ai.transcribe_window = false;
    }
    if clicks.setup
        && let Some(ai) = app.ai.as_mut()
    {
        open_setup(ai);
    }
    if let Some(url) = clicks.url {
        crate::links::open(app, ctx, &url);
    }
    if clicks.discard
        && let Some(ai) = app.ai.as_mut()
    {
        ai.transcribe.waiting = None;
        ai.transcribe.note = None;
    }
    if clicks.place {
        place_waiting_captions(app);
    }
    if clicks.send
        && let Some(src) = chosen
        && let Err(e) = start_transcribe(app, ctx, src)
        && let Some(ai) = app.ai.as_mut()
    {
        ai.transcribe.error = Some(e);
    }
}

fn voice_window(app: &mut FilmcraftApp, ctx: &egui::Context) {
    if !app.ai.as_ref().is_some_and(|ai| ai.voice_window) {
        return;
    }
    let has_seq = app.session.active_sequence().is_some();
    let Some(ai) = app.ai.as_mut() else { return };
    let provider = ai.state.provider();
    let mut clicks = Clicks::default();
    let mut elems: Elems = Vec::new();
    let mut open = true;
    let resp = egui::Window::new("Voice-over from Text")
        .id(egui::Id::new("ai-voiceover"))
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(460.0)
        .centered(ctx)
        .show(ctx, |ui| {
            ui.set_max_width(460.0);
            if !provider.is_set_up() {
                needs_setup(ui, &mut elems, "ai.voiceover", "Voice-over from Text", &mut clicks);
                return;
            }
            let (busy, waiting) = (ai.voice.busy(), ai.voice.waiting.is_some());
            ui.add(
                egui::Label::new(
                    "Voice-over from Text sends your text to the AI service. The speech comes back as a new audio clip at the playhead, on a free audio track.",
                )
                .wrap(),
            );
            dim(ui, "Nothing is sent until you click Generate.");
            ui.add_space(8.0);
            ui.label("Text");
            let edit = egui::TextEdit::multiline(&mut ai.text)
                .id(egui::Id::new("ai-voiceover-text"))
                .hint_text("Welcome back. In this part, we look at the first results.")
                .desired_rows(5)
                .desired_width(f32::INFINITY)
                .char_limit(MAX_SPEECH_CHARS);
            let r = ui.add_enabled(!busy && !waiting, edit);
            elems.push(("ai.voiceover.text".into(), r.rect, "Text".into()));
            let chars = ai.text.trim().chars().count();
            dim(ui, &format!("{chars} of {MAX_SPEECH_CHARS} characters."));
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Voice");
                let shown = VOICES.iter().find(|(id, _)| *id == ai.voice_id).map_or(VOICES[0].1, |(_, name)| *name);
                let cb = ui.add_enabled_ui(!busy && !waiting, |ui| {
                    egui::ComboBox::from_id_salt("ai-voiceover-voice").selected_text(shown).width(160.0).show_ui(ui, |ui| {
                        for (id, name) in VOICES {
                            let r = ui.selectable_label(ai.voice_id == id, name);
                            elems.push((format!("ai.voiceover.voice.{id}"), r.rect, name.into()));
                            if r.clicked() {
                                ai.voice_id = id.to_string();
                            }
                        }
                    })
                });
                elems.push(("ai.voiceover.voice".into(), cb.inner.response.rect, shown.into()));
            });
            ui.add_space(4.0);
            if !has_seq {
                warn(ui, "Open a sequence first. The voice-over goes on its timeline.");
            }
            if !provider.supports(Line::Speech) {
                warn(ui, &Fail::unsupported(Line::Speech).message);
            }
            for line in cost_lines(&ai.state, &provider, speech_estimate(chars)) {
                dim(ui, &line);
            }
            ui.add_space(8.0);
            let can_send = !busy && !waiting && chars > 0 && has_seq && provider.supports(Line::Speech);
            ui.horizontal(|ui| {
                if waiting {
                    if button(ui, &mut elems, "ai.voiceover.place", "Place Voice-over", has_seq) {
                        clicks.place = true;
                    }
                    if button(ui, &mut elems, "ai.voiceover.discard", "Discard", true) {
                        clicks.discard = true;
                    }
                } else if button(ui, &mut elems, "ai.voiceover.run", "Generate", can_send) {
                    clicks.send = true;
                }
            });
            status_rows(ui, &mut ai.state, &ai.voice, GENERATING, &mut clicks);
        });
    if let Some(r) = &resp {
        app.auto.add("ai.voiceover.window", r.response.rect, "Voice-over from Text");
    }
    for (id, rect, label) in &elems {
        app.auto.add(id, *rect, label);
    }
    if !open && let Some(ai) = app.ai.as_mut() {
        ai.voice_window = false;
    }
    if clicks.setup
        && let Some(ai) = app.ai.as_mut()
    {
        open_setup(ai);
    }
    if let Some(url) = clicks.url {
        crate::links::open(app, ctx, &url);
    }
    if clicks.discard
        && let Some(ai) = app.ai.as_mut()
    {
        ai.voice.waiting = None;
        ai.voice.note = None;
    }
    if clicks.place {
        place_waiting_voice(app);
    }
    if clicks.send
        && let Err(e) = start_voice(app, ctx)
        && let Some(ai) = app.ai.as_mut()
    {
        ai.voice.error = Some(e);
    }
}

// ---- Preferences ▸ AI ---------------------------------------------------------------------------

/// Preferences ▸ AI: what is in use, sign-in and key controls, and the setup flow in place. AI
/// settings take effect at once (OK and Cancel do not apply to them).
pub fn prefs_section(app: &mut FilmcraftApp, ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    let Some(ai) = app.ai.as_mut() else {
        ui.label(RichText::new(OFF).weak());
        return;
    };
    ui.add(
        egui::Label::new(
            RichText::new("Sequence ▸ AI ▸ Transcribe to Captions and Voice-over from Text use publik API or your own key. Changes here take effect at once.")
                .weak(),
        )
        .wrap(),
    );
    ui.add_space(6.0);
    let action = if ai.inline_setup {
        let action = ai.state.setup_ui(ui);
        if action == Action::Close {
            ai.inline_setup = false;
        }
        action
    } else {
        let mut open_setup = false;
        let action = ai.state.options_ui(ui, &mut open_setup);
        if open_setup {
            ai.inline_setup = true;
        }
        action
    };
    if let Action::OpenUrl(url) = action {
        crate::links::open(app, &ctx, &url);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_app(name: &str) -> (FilmcraftApp, PathBuf) {
        let dir = std::env::temp_dir().join(format!("filmcraft-ai-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut session = Session::default();
        session.execute("file.openDemoProject", json!({})).unwrap();
        session.path = Some(dir.join("film.fcproj").to_string_lossy().into_owned());
        (FilmcraftApp::new(session), dir)
    }

    fn with_ai(app: &mut FilmcraftApp, dir: &Path) {
        let info = AppInfo {
            slug: "filmcraft-test",
            name: "FilmCraft",
            version: "0.0.0",
            app_token: "",
            dialects: &["audio_transcriptions", "audio_speech"],
            config_dir: Some(dir.join("config")),
        };
        app.ai = Some(Ai::new(info));
    }

    fn first_a1_clip(app: &FilmcraftApp) -> ClipId {
        app.session.active_sequence().unwrap().audio_tracks[0].items[0].id
    }

    #[test]
    fn segments_are_read_clipped_and_cleaned() {
        let bytes = br#"[
            {"id":0,"start":0.0,"end":2.5,"text":" Hello there."},
            {"id":1,"start":2.5,"end":4.0,"text":"  Two\n lines  "},
            {"id":2,"start":3.0,"end":99.0,"text":"runs past the end"},
            {"id":3,"start":1.0,"end":1.0,"text":"no length"},
            {"id":4,"start":1.0,"end":2.0,"text":"   "},
            {"id":5,"start":"x","end":2.0,"text":"bad time"},
            {"id":6,"start":-3.0,"end":0.5,"text":"before zero"}
        ]"#;
        let s = parse_segments(bytes, Tick::from_seconds_f64(5.0));
        assert_eq!(s.len(), 4, "{s:?}");
        assert_eq!(s[0], Segment { start: 0.0, end: 2.5, text: "Hello there.".into() });
        assert_eq!(s[1].text, "Two lines");
        assert_eq!((s[2].start, s[2].end), (3.0, 5.0));
        assert_eq!((s[3].start, s[3].end), (0.0, 0.5));
        assert!(parse_segments(b"null", Tick::from_seconds_f64(5.0)).is_empty());
        assert!(parse_segments(b"not json", Tick::from_seconds_f64(5.0)).is_empty());
        assert!(parse_segments(br#"[{"start":0,"end":1,"text":"x"}]"#, Tick(-5)).is_empty(), "a negative length keeps nothing");
    }

    #[test]
    fn srt_puts_segments_at_sequence_time() {
        let segs = vec![Segment { start: 0.0, end: 1.25, text: "One".into() }, Segment { start: 61.5, end: 3_725.0, text: "Two".into() }];
        let s = srt(&segs, Tick(2 * TICKS_PER_SECOND));
        assert_eq!(s, "1\n00:00:02,000 --> 00:00:03,250\nOne\n\n2\n00:01:03,500 --> 01:02:07,000\nTwo\n\n");
    }

    #[test]
    fn lengths_and_estimates_read_in_minutes_and_dollars() {
        assert_eq!(clock(0.0), "0:00");
        assert_eq!(clock(151.4), "2:31");
        assert_eq!(clock(3_725.0), "1:02:05");
        assert_eq!(clock(f64::NAN), "0:00");
        assert_eq!(clock(-4.0), "0:00");
        assert!(transcribe_estimate(151.0).contains("about $0.006 a minute"));
        assert!(transcribe_estimate(151.0).contains("costs about $0.02."), "{}", transcribe_estimate(151.0));
        assert!(transcribe_estimate(20.0).contains("costs about $0.0020."), "{}", transcribe_estimate(20.0));
        assert!(speech_estimate(412).contains("about $0.015 for 1,000 characters"));
        assert!(speech_estimate(412).contains("costs about $0.0062."), "{}", speech_estimate(412));
        assert!(speech_estimate(0).contains("costs about $0.00."), "{}", speech_estimate(0));
    }

    #[test]
    fn audio_files_are_named_by_their_bytes() {
        assert_eq!(audio_ext(b"RIFF....WAVE"), "wav");
        assert_eq!(audio_ext(b"ID3\x04"), "mp3");
        assert_eq!(audio_ext(&[0xFF, 0xFB, 0x90]), "mp3");
        assert_eq!(audio_ext(b"OggS"), "ogg");
        assert_eq!(audio_ext(b""), "mp3");
    }

    #[test]
    fn the_audio_sent_is_16_khz_mono_aac_in_mp4() {
        let mut m4a = M4a::new().unwrap();
        let tone: Vec<f32> = (0..SEND_RATE * 2).map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / SEND_RATE as f32).sin() * 0.5).collect();
        for chunk in tone.chunks(SEND_RATE as usize) {
            m4a.push(chunk).unwrap();
        }
        let bytes = m4a.finish().unwrap();
        assert!(bytes.len() < 12_000, "about 4 KB a second: {}", bytes.len());
        let mp4 = filmcraft_isobmff::open(&bytes[..]).unwrap();
        let t = &mp4.tracks[0];
        assert_eq!(t.timescale, SEND_RATE);
        let filmcraft_isobmff::CodecConfig::Aac(cfg) = &t.entries[0].codec else { panic!("not AAC: {:?}", t.entries[0].codec) };
        let mut dec = filmcraft_aac::Decoder::new(&cfg.asc).unwrap();
        let mut decoded = 0;
        for i in 0..t.samples.len() {
            let au = mp4.read_sample(&bytes[..], 0, i).unwrap();
            let pcm = dec.decode(&au).unwrap();
            assert_eq!(pcm.len(), 1, "mono");
            decoded += pcm[0].len();
        }
        assert!(decoded >= tone.len(), "{decoded} samples decoded for {}", tone.len());
    }

    #[test]
    fn sources_follow_the_selection_and_the_marks() {
        let (mut app, _dir) = demo_app("sources");
        assert_eq!(clip_source(&app.session), None, "nothing selected");
        assert_eq!(range_source(&app.session), None, "nothing marked");
        let clip = first_a1_clip(&app);
        app.session.state.selection = vec![clip];
        let src = clip_source(&app.session).unwrap();
        let (_, item) = app.session.active_sequence().unwrap().find_item(clip).unwrap();
        assert_eq!((src.clip, src.start, src.end), (Some(clip), item.start, item.end()));
        assert!(src.label.contains("on A1"), "{}", src.label);
        // a linked video clip stands for its audio
        let seq = app.session.active_sequence().unwrap();
        let video = seq
            .video_tracks
            .iter()
            .flat_map(|t| t.items.iter())
            .find(|v| v.link.is_some() && seq.audio_tracks.iter().any(|a| a.items.iter().any(|i| i.link == v.link)));
        if let Some(v) = video {
            let (vid, link) = (v.id, v.link);
            app.session.state.selection = vec![vid];
            let src = clip_source(&app.session).unwrap();
            let (_, audio) = app.session.active_sequence().unwrap().find_item(src.clip.unwrap()).unwrap();
            assert_eq!(audio.link, link);
        }
        app.session.execute("markers.markIn", json!({"seconds": 1.0})).unwrap();
        app.session.execute("markers.markOut", json!({"seconds": 2.0})).unwrap();
        let r = range_source(&app.session).unwrap();
        assert_eq!(r.clip, None);
        let len = r.length().seconds();
        assert!((1.0..1.1).contains(&len), "In to Out with the Out frame: {len}");
    }

    #[test]
    fn a_clip_renders_to_audio_of_its_length() {
        let (mut app, _dir) = demo_app("render");
        let clip = first_a1_clip(&app);
        app.session.state.selection = vec![clip];
        let mut src = clip_source(&app.session).unwrap();
        src.end = src.start + Tick::from_seconds_f64(1.5);
        let project = app.session.project.clone();
        let sources = app.session.media.provider(project.clone(), app.session.services.clone());
        let bytes = encode_source(&project, &src, &sources).unwrap();
        let mp4 = filmcraft_isobmff::open(&bytes[..]).unwrap();
        let t = &mp4.tracks[0];
        let seconds = t.duration as f64 / f64::from(t.timescale);
        assert!((1.45..1.8).contains(&seconds), "{seconds} s");
        // the mix of In to Out renders too
        app.session.execute("markers.markIn", json!({"seconds": 0.5})).unwrap();
        app.session.execute("markers.markOut", json!({"seconds": 1.0})).unwrap();
        let r = range_source(&app.session).unwrap();
        assert!(encode_source(&project, &r, &sources).is_ok());
    }

    #[test]
    fn captions_go_on_a_new_track_in_one_undo_step() {
        let (mut app, _dir) = demo_app("captions");
        let seq = app.session.state.active_sequence.unwrap();
        let tracks = app.session.active_sequence().unwrap().caption_tracks.len();
        let undo = app.session.history.undo.len();
        let c = Captions {
            seq,
            start: Tick::from_seconds_f64(2.0),
            segments: vec![Segment { start: 0.0, end: 1.5, text: "First line.".into() }, Segment { start: 1.5, end: 3.0, text: "Second line.".into() }],
            text: "First line. Second line.".into(),
        };
        let note = place_captions(&mut app, &c).unwrap();
        assert!(note.starts_with("2 captions"), "{note}");
        let q = app.session.active_sequence().unwrap();
        assert_eq!(q.caption_tracks.len(), tracks + 1);
        let t = &q.caption_tracks[0];
        assert_eq!(t.name, TRACK_NAME);
        assert_eq!(t.captions.len(), 2);
        assert_eq!(t.captions[0].text, "First line.");
        let at = t.captions[0].start.seconds();
        assert!((at - 2.0).abs() < 0.05, "{at}");
        assert_eq!(app.session.history.undo.len(), undo + 1, "one undo step");
        // a second run names its track apart
        place_captions(&mut app, &c).unwrap();
        assert_eq!(app.session.active_sequence().unwrap().caption_tracks[0].name, format!("{TRACK_NAME} 2"));
        let _ = app.session.undo();
        let _ = app.session.undo();
        assert_eq!(app.session.active_sequence().unwrap().caption_tracks.len(), tracks);
    }

    #[test]
    fn captions_wait_while_their_sequence_is_not_in_front() {
        let (mut app, dir) = demo_app("captions-wait");
        with_ai(&mut app, &dir);
        let c = Captions { seq: ItemId(u64::MAX), start: Tick::ZERO, segments: vec![Segment { start: 0.0, end: 1.0, text: "x".into() }], text: "x".into() };
        arrived_captions(&mut app, c.clone());
        let ai = app.ai.as_ref().unwrap();
        assert_eq!(ai.transcribe.waiting, Some(c));
        assert!(ai.transcribe.note.as_deref().unwrap_or("").contains("Click Place Captions"));
        assert!(app.ui.status.contains("Place Captions"), "the window is closed: the status bar says where it waits");
    }

    #[test]
    fn a_voice_over_goes_at_the_playhead_in_one_undo_step() {
        let (mut app, dir) = demo_app("voice");
        let samples: Vec<f32> = (0..24_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mut v = VoiceOut { bytes: filmcraft_engine::voiceover::write_wav_f32(&samples, 24_000), ext: "wav", path: None };
        let at = Tick::from_seconds_f64(3.0);
        app.session.set_playhead(at);
        let undo = app.session.history.undo.len();
        let audio_clips = |app: &FilmcraftApp| app.session.active_sequence().unwrap().audio_tracks.iter().map(|t| t.items.len()).sum::<usize>();
        let before = audio_clips(&app);
        let note = place_voice(&mut app, &mut v).unwrap();
        assert!(note.contains("AI Voice-over 1.wav"), "{note}");
        let path = v.path.clone().unwrap();
        assert!(Path::new(&path).starts_with(&dir), "saved next to the project: {path}");
        assert_eq!(audio_clips(&app), before + 1);
        assert_eq!(app.session.history.undo.len(), undo + 1, "one undo step");
        let clip = app.session.state.selection[0];
        let (_, item) = app.session.active_sequence().unwrap().find_item(clip).unwrap();
        assert!((item.start - at).0.abs() <= app.session.sequence_rate().frame_duration().0, "at the playhead: {:?}", item.start);
        let _ = app.session.undo();
        assert_eq!(audio_clips(&app), before);
    }

    #[test]
    fn a_voice_over_waits_while_playback_runs() {
        let (mut app, dir) = demo_app("voice-wait");
        with_ai(&mut app, &dir);
        app.playback.playing = true;
        let v = VoiceOut { bytes: vec![1, 2, 3], ext: "mp3", path: None };
        arrived_voice(&mut app, v.clone());
        let ai = app.ai.as_ref().unwrap();
        assert_eq!(ai.voice.waiting, Some(v));
        assert!(ai.voice.note.as_deref().unwrap_or("").contains("Stop playback first."));
    }

    #[test]
    fn the_menu_has_ai_items_only_where_ai_is_on() {
        let (mut app, dir) = demo_app("menu");
        let ids = |app: &FilmcraftApp| crate::menus::menu_items(app).into_iter().map(|i| (i.id, i.path)).collect::<Vec<_>>();
        assert!(!ids(&app).iter().any(|(id, _)| id.starts_with("ai.") || id == PREFERENCES));
        assert_eq!(route(&mut app, TRANSCRIBE), Some(Err(OFF.into())));
        with_ai(&mut app, &dir);
        let items = ids(&app);
        for id in [TRANSCRIBE, VOICE_OVER, SETTINGS] {
            let (_, path) = items.iter().find(|(i, _)| i == id).unwrap();
            assert_eq!(path, &vec!["Sequence".to_string(), "AI".to_string()]);
        }
        let (_, path) = items.iter().find(|(i, _)| i == PREFERENCES).unwrap();
        assert_eq!(path, &vec!["Edit".to_string(), "Preferences".to_string()]);
        // the items open windows; nothing is sent
        let ctx = egui::Context::default();
        assert_eq!(crate::menus::invoke(&mut app, &ctx, TRANSCRIBE, json!({})), Ok(json!({"dialog": "aiTranscribe"})));
        assert_eq!(crate::menus::invoke(&mut app, &ctx, VOICE_OVER, json!({})), Ok(json!({"dialog": "aiVoiceOver"})));
        let ai = app.ai.as_ref().unwrap();
        assert!(ai.transcribe_window && ai.voice_window && !ai.working());
        let r = crate::menus::invoke(&mut app, &ctx, PREFERENCES, json!({})).unwrap();
        assert_eq!(r["page"], crate::panels::settings::AI_PAGE);
        assert_eq!(app.ui.settings.as_ref().map(|d| d.page.as_str()), Some(crate::panels::settings::AI_PAGE));
    }

    #[test]
    fn the_first_launch_asks_once_and_never_under_control() {
        let (mut app, dir) = demo_app("first-run");
        with_ai(&mut app, &dir);
        let ctx = egui::Context::default();
        for _ in 0..5 {
            frame(&mut app, &ctx);
        }
        let ai = app.ai.as_mut().unwrap();
        // a credential for the test slug may not exist: the setup window opens once
        assert_eq!(ai.setup_window, ai.state.wants_first_run() || ai.state.owes_first_card());
        let (mut app, dir) = demo_app("first-run-control");
        with_ai(&mut app, &dir);
        app.ai.as_mut().unwrap().skip_first_run();
        for _ in 0..5 {
            frame(&mut app, &ctx);
        }
        assert!(!app.ai.as_ref().unwrap().setup_window);
    }
}
