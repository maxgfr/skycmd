//! The `recall` probe of s6-sleep (`docs/s6-sleep-prereg.md` §3) played by a model in the browser
//! (the Memory page and the Space of s6-sleep3-nano).
//!
//! It is the evaluation's probe, not a copy of it: an episode is `sky_games::make_game("recall",
//! seed)`, the function `sky-train s2-probe` calls; the model reads the game's own state text with
//! the question's canonical instructions and options (`QuestionSpec::instructions`,
//! `canonical_options`), as `s2-probe` builds its records; it answers at the last exit (τ = 1); the
//! argmax is played (first on ties) and an ask is scored by the game itself (`correct` in its
//! snapshot's `asks`, the field `s2-probe` reads). What differs is the engine: `s2-probe` runs the
//! training checkpoint in mlx-rs, this runs the exported GGUF in `sky-infer`, whose memory matches
//! the training code on the golden sequences (`crates/sky-infer/tests/golden_s6m.rs`, natively and
//! on wasm32).
//!
//! A model with a memory plays an episode as one stream, as `s2-probe` and `sky-arena
//! --memory-passes` do: a fresh zero memory, one frame per decision, the frame written at full
//! depth when the next one starts, a sleep of `N − 1` passes over the last frames every
//! `SLEEP_EVERY` frames (`recall` never pauses). With the memory **zeroed**, the engine resets it
//! before every frame (`SleepOptions::memory_off`, `s2-probe --memory-off`). A model without a
//! memory (the control) answers every frame alone.

use serde::Serialize;
use serde_json::{json, Value};
use sky_games::recall::{BUCKETS, FRAMES};
use sky_games::{make_game, DecisionGame};
use sky_infer::s2::memory::{SleepOptions, SLEEP_EVERY};
use sky_infer::QType;

use crate::clock;
use crate::model::Model;

/// The game, by its name in `sky_games::make_game`.
pub const GAME: &str = sky_games::recall::NAME;
/// The one stream of a probe.
const STREAM: u64 = 1;
/// Sleep passes accepted (`N`, the wake pass included).
pub const MAX_PASSES: usize = 8;

/// What a frame shows, read from the game's state text
/// (`show <door> colour <colour> ask <door> noise <n>`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FrameView {
    pub frame: u32,
    /// The text the model reads.
    pub state: String,
    /// `show`, `ask` or `idle`.
    pub kind: &'static str,
    /// The door shown or asked, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub door: Option<String>,
    /// The colour shown (a show frame only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub colour: Option<String>,
    /// The right answer: `wait`, or the colour of an ask (the game's oracle).
    pub answer: &'static str,
    /// An ask: the frames since its door was shown, and the index of its bucket in `buckets`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delay: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket: Option<usize>,
}

/// One answered ask of an episode (as `sky_games::recall::Ask`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AskResult {
    pub frame: u32,
    pub delay: u32,
    pub bucket: usize,
    pub correct: bool,
}

/// One frame played by the model.
#[derive(Clone, Debug, Serialize)]
pub struct StepView {
    #[serde(flatten)]
    pub view: FrameView,
    /// The model's probability of each answer, in the order of `keys`.
    pub probs: Vec<f32>,
    /// The answer played (the argmax).
    pub choice: &'static str,
    /// An ask: whether the game scored it right.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correct: Option<bool>,
    /// Wall time of the decision (µs).
    pub micros: f64,
    pub done: bool,
}

/// A whole episode played by the model.
#[derive(Clone, Debug, Serialize)]
pub struct EpisodeResult {
    pub seed: u64,
    pub asks: Vec<AskResult>,
    pub decisions: usize,
    /// Sum of the decision times (µs).
    pub micros: f64,
}

/// The probe: a model, how it reads its memory, and the episode in progress.
pub struct Probe {
    model: Model,
    memory: bool,
    zeroed: bool,
    passes: usize,
    instructions: &'static str,
    options: Vec<String>,
    keys: Vec<&'static str>,
    game: Option<Box<dyn DecisionGame>>,
    seed: u64,
}

fn argmax(v: &[f32]) -> usize {
    // `s2-probe`'s: the first of equal values.
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

/// The words of a `recall` state text: `(shown door, shown colour, asked door)`.
fn words(state: &str) -> (Option<String>, Option<String>, Option<String>) {
    let mut it = state.split(' ');
    let mut get = |key: &str| -> Option<String> {
        let mut out = None;
        while let Some(k) = it.next() {
            if k == key {
                out = it.next().filter(|v| *v != "none").map(str::to_string);
                break;
            }
        }
        out
    };
    let show = get("show");
    let colour = get("colour");
    let ask = get("ask");
    (show, colour, ask)
}

/// The asks a game has scored so far, from its snapshot (what `s2-probe` reads).
fn asks_of(g: &dyn DecisionGame) -> Vec<AskResult> {
    let snap = g.snapshot_json();
    snap["asks"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .iter()
        .filter_map(|x| {
            Some(AskResult {
                frame: x["frame"].as_u64()? as u32,
                delay: x["delay"].as_u64()? as u32,
                bucket: x["bucket"].as_u64()? as usize,
                correct: x["correct"].as_bool()?,
            })
        })
        .collect()
}

fn new_game(seed: u64) -> Box<dyn DecisionGame> {
    make_game(GAME, seed).expect("recall is a probe of sky-games")
}

/// The episode of `seed` frame by frame, as the game's oracle plays it: what each frame shows, its
/// right answer, and the delay of each ask (from the game's own record). Independent of any model.
pub fn describe(seed: u64) -> Vec<FrameView> {
    let mut g = new_game(seed);
    let keys = g.questions()[0].keys();
    let mut frames = Vec::with_capacity(FRAMES as usize);
    while !g.done() {
        let t = g.stats().steps;
        let state = g.state_text();
        let answer = keys[argmax(&g.oracle()[0])];
        frames.push(view(t, state, answer));
        let choice = argmax(&g.oracle()[0]);
        g.apply(choice, false);
    }
    for a in asks_of(&*g) {
        if let Some(f) = frames.get_mut(a.frame as usize) {
            f.delay = Some(a.delay);
            f.bucket = Some(a.bucket);
        }
    }
    frames
}

fn view(frame: u32, state: String, answer: &'static str) -> FrameView {
    let (show, colour, ask) = words(&state);
    let (kind, door) = match (show, ask) {
        (Some(d), _) => ("show", Some(d)),
        (None, Some(d)) => ("ask", Some(d)),
        (None, None) => ("idle", None),
    };
    FrameView {
        frame,
        state,
        kind,
        door,
        colour,
        answer,
        delay: None,
        bucket: None,
    }
}

impl Probe {
    /// A probe of `gguf` (an s2 model, with or without a memory), memory on, `N` = 1.
    pub fn load(gguf: &[u8]) -> Result<Probe, String> {
        let mut model = Model::load(gguf)?;
        if model.engine().s2_model().is_none() {
            return Err("the recall probe needs an s2 model".into());
        }
        // τ = 1: every answer at the last exit, as s2-probe reads it.
        model.set_exit_threshold(Some(1.0));
        let memory = model.engine().has_memory();
        let g = new_game(0);
        let q = &g.questions()[0];
        if QType::from_name(q.qtype.as_str()) != Some(QType::Choice) {
            return Err("recall's question is a choice".into());
        }
        let mut p = Probe {
            model,
            memory,
            zeroed: false,
            passes: 1,
            instructions: q.instructions(),
            options: q.canonical_options(),
            keys: q.keys(),
            game: None,
            seed: 0,
        };
        p.apply_sleep();
        Ok(p)
    }

    pub fn has_memory(&self) -> bool {
        self.memory
    }

    /// `on` or `zeroed` (the memory reset before every frame). A model without a memory accepts
    /// `on` only. Takes effect at the next episode.
    pub fn set_memory(&mut self, mode: &str) -> Result<(), String> {
        self.zeroed = match mode {
            "on" => false,
            "zeroed" if self.memory => true,
            "zeroed" => return Err("this model has no memory to zero".into()),
            _ => return Err(format!("memory mode {mode:?}: on or zeroed")),
        };
        self.apply_sleep();
        Ok(())
    }

    pub fn memory_mode(&self) -> &'static str {
        if self.zeroed {
            "zeroed"
        } else {
            "on"
        }
    }

    /// `N`, the passes of a sleep, the wake pass included (1: no sleep). Takes effect at the next
    /// episode.
    pub fn set_sleep_passes(&mut self, n: usize) -> Result<(), String> {
        if !(1..=MAX_PASSES).contains(&n) {
            return Err(format!("sleep passes {n}: between 1 and {MAX_PASSES}"));
        }
        self.passes = n;
        self.apply_sleep();
        Ok(())
    }

    pub fn sleep_passes(&self) -> usize {
        self.passes
    }

    fn apply_sleep(&mut self) {
        let sleep = SleepOptions {
            passes: self.passes,
            every: SLEEP_EVERY,
            memory_off: self.zeroed,
        };
        let e = self.model.engine_mut();
        e.set_sleep(sleep);
        e.reset_stream(STREAM);
    }

    /// The question and answers the model reads, and the probe's constants.
    pub fn info(&mut self) -> Value {
        json!({
            "name": self.model.name(),
            "n_params": self.model.n_params(),
            "file_bytes": self.model.file_bytes(),
            "memory": self.memory,
            "memory_mode": self.memory_mode(),
            "sleep_passes": self.passes,
            "sleep_every": SLEEP_EVERY,
            "frames": FRAMES,
            "buckets": BUCKETS,
            "instructions": self.instructions,
            "options": self.options,
            "keys": self.keys,
            "exit_threshold": 1.0,
        })
    }

    /// Starts the episode of `seed` from a zero memory and describes it (see [`describe`]).
    pub fn start(&mut self, seed: u64) -> Value {
        self.model.engine_mut().reset_stream(STREAM);
        self.game = Some(new_game(seed));
        self.seed = seed;
        json!({
            "seed": seed,
            "frames": describe(seed),
            "memory": self.memory,
            "memory_mode": self.memory_mode(),
            "sleep_passes": self.passes,
        })
    }

    /// Whether the episode in progress has ended (true when none was started).
    pub fn done(&self) -> bool {
        self.game.as_ref().is_none_or(|g| g.done())
    }

    /// The model's probabilities for the current frame of `g` (one frame of the stream).
    fn decide(&mut self, g: &dyn DecisionGame) -> Result<(Vec<f32>, f64), String> {
        let state = g.state_text();
        let t0 = clock::now_us();
        let p = if self.memory {
            let e = self.model.engine_mut();
            // One frame per decision point, even on a state text equal to the last one.
            e.new_frame(STREAM, g.pause());
            e.decide_raw_stream(
                STREAM,
                QType::Choice,
                self.instructions,
                &self.options,
                &state,
            )
            .map_err(|e| e.to_string())?
        } else {
            self.model
                .probs(QType::Choice, self.instructions, &self.options, &state)?
        };
        Ok((p, clock::now_us() - t0))
    }

    /// Plays the current frame of the episode in progress and moves to the next one.
    pub fn step(&mut self) -> Result<StepView, String> {
        let mut g = self
            .game
            .take()
            .ok_or("no episode in progress: call start(seed)")?;
        if g.done() {
            self.game = Some(g);
            return Err("the episode has ended".into());
        }
        let r = self.step_game(&mut *g);
        self.game = Some(g);
        r
    }

    fn step_game(&mut self, g: &mut dyn DecisionGame) -> Result<StepView, String> {
        let t = g.stats().steps;
        let state = g.state_text();
        let answer = self.keys[argmax(&g.oracle()[0])];
        let (probs, micros) = self.decide(&*g)?;
        let choice = argmax(&probs);
        let before = asks_of(&*g).len();
        g.apply(choice, false);
        let mut v = view(t, state, answer);
        let mut correct = None;
        if let Some(a) = asks_of(&*g).get(before) {
            v.delay = Some(a.delay);
            v.bucket = Some(a.bucket);
            correct = Some(a.correct);
        }
        Ok(StepView {
            view: v,
            probs,
            choice: self.keys[choice],
            correct,
            micros,
            done: g.done(),
        })
    }

    /// Plays the whole episode of `seed` from a zero memory (the episode in progress, if any, is
    /// dropped).
    pub fn run_episode(&mut self, seed: u64) -> Result<EpisodeResult, String> {
        self.game = None;
        self.model.engine_mut().reset_stream(STREAM);
        let mut g = new_game(seed);
        let (mut decisions, mut micros) = (0usize, 0.0f64);
        while !g.done() {
            if g.decision_due() {
                let (p, us) = self.decide(&*g)?;
                g.apply(argmax(&p), false);
                decisions += 1;
                micros += us;
            } else {
                g.tick();
            }
        }
        Ok(EpisodeResult {
            seed,
            asks: asks_of(&*g),
            decisions,
            micros,
        })
    }

    /// The seed of the last episode started.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// A seed given as a JavaScript number: a whole number in `[0, 2^53]`.
pub fn seed_from_f64(seed: f64) -> Result<u64, String> {
    if seed.is_finite() && seed >= 0.0 && seed.fract() == 0.0 && seed <= 9_007_199_254_740_992.0 {
        Ok(seed as u64)
    } else {
        Err(format!("seed {seed}: a whole number between 0 and 2^53"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_words_of_a_state_text() {
        assert_eq!(
            words("show a colour red ask none noise 46"),
            (Some("a".into()), Some("red".into()), None)
        );
        assert_eq!(
            words("show none colour none ask b noise 3"),
            (None, None, Some("b".into()))
        );
        assert_eq!(
            words("show none colour none ask none noise 0"),
            (None, None, None)
        );
    }

    #[test]
    fn seeds_are_whole_javascript_numbers() {
        assert_eq!(seed_from_f64(7_000_000_018.0), Ok(7_000_000_018));
        assert!(seed_from_f64(-1.0).is_err());
        assert!(seed_from_f64(1.5).is_err());
        assert!(seed_from_f64(f64::NAN).is_err());
    }
}
