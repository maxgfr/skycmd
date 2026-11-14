//! "Recall", the primary memory probe of `docs/s6-sleep-prereg.md` §3.
//!
//! Three doors each have a colour. A *show* frame gives a door its colour for one frame; an *ask*
//! frame names a door, and the right answer is the colour it was last given. Every other frame
//! only carries a distractor number, and its answer is `wait`. Nothing on an ask frame tells the
//! colour, so a policy that reads one frame at a time is right one time in four.

use crate::common::{DecisionGame, EpisodeStats, Event, OptionSpec, QuestionSpec, StepOutcome};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use sky_schema::QType;

pub const NAME: &str = "recall";
/// Frames of an episode; every frame asks for a decision.
pub const FRAMES: u32 = 64;
pub const DOORS: [&str; 3] = ["a", "b", "c"];
pub const COLOURS: [&str; 4] = ["red", "green", "blue", "yellow"];
/// Delay buckets (inclusive): frames between the show of a door and its ask.
pub const BUCKETS: [(u32, u32); 4] = [(1, 4), (5, 8), (9, 16), (17, 32)];
/// Asks planned per bucket and per episode.
pub const ASKS_PER_BUCKET: usize = 2;
/// Placement attempts per ask before it is given up (rare: see the tests).
const ATTEMPTS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    Idle,
    Show { door: u8, colour: u8 },
    Ask { door: u8, colour: u8, delay: u8 },
}

/// One answered ask frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Ask {
    pub frame: u32,
    /// Frames since the door was shown.
    pub delay: u32,
    /// Index into [`BUCKETS`].
    pub bucket: usize,
    pub correct: bool,
}

/// Index into [`BUCKETS`] of a delay.
pub fn bucket_of(delay: u32) -> usize {
    BUCKETS
        .iter()
        .position(|&(lo, hi)| (lo..=hi).contains(&delay))
        .expect("a delay between 1 and 32")
}

#[derive(Debug, Clone)]
pub struct Recall {
    rng: SmallRng,
    frames: Vec<Frame>,
    noise: Vec<u8>,
    t: u32,
    asks: Vec<Ask>,
}

impl Recall {
    pub fn new(seed: u64) -> Self {
        let mut g = Recall {
            rng: SmallRng::seed_from_u64(seed ^ 0x7265_6361_6c6c),
            frames: Vec::new(),
            noise: Vec::new(),
            t: 0,
            asks: Vec::new(),
        };
        g.reset();
        g
    }

    /// Draws a new episode.
    pub fn reset(&mut self) {
        self.frames = plan(&mut self.rng);
        self.noise = (0..FRAMES).map(|_| self.rng.gen_range(0..100)).collect();
        self.t = 0;
        self.asks.clear();
    }

    /// The asks answered so far.
    pub fn asks(&self) -> &[Ask] {
        &self.asks
    }

    /// Ask frames of the whole episode.
    pub fn planned_asks(&self) -> usize {
        self.frames
            .iter()
            .filter(|f| matches!(f, Frame::Ask { .. }))
            .count()
    }

    fn frame(&self) -> Frame {
        self.frames[(self.t as usize).min(self.frames.len() - 1)]
    }

    pub fn state_text(&self) -> String {
        let (show, colour, ask) = match self.frame() {
            Frame::Idle => ("none", "none", "none"),
            Frame::Show { door, colour } => {
                (DOORS[door as usize], COLOURS[colour as usize], "none")
            }
            Frame::Ask { door, .. } => ("none", "none", DOORS[door as usize]),
        };
        let noise = self.noise[(self.t as usize).min(self.noise.len() - 1)];
        format!("show {show} colour {colour} ask {ask} noise {noise}")
    }

    /// One-hot over `wait` and the four colours.
    pub fn oracle(&self) -> Vec<Vec<f32>> {
        let mut t = vec![0.0; 1 + COLOURS.len()];
        match self.frame() {
            Frame::Ask { colour, .. } => t[1 + colour as usize] = 1.0,
            _ => t[0] = 1.0,
        }
        vec![t]
    }

    /// Answers the current frame (`0` is `wait`, `1..=4` a colour) and moves to the next one.
    pub fn step(&mut self, choice: usize) -> StepOutcome {
        if self.t >= FRAMES {
            return StepOutcome::new(0.0, true, Event::None, false);
        }
        let mut reward = 0.0;
        if let Frame::Ask { colour, delay, .. } = self.frame() {
            let correct = choice == 1 + colour as usize;
            reward = correct as u8 as f32;
            self.asks.push(Ask {
                frame: self.t,
                delay: delay as u32,
                bucket: bucket_of(delay as u32),
                correct,
            });
        }
        self.t += 1;
        let done = self.t >= FRAMES;
        let event = if done { Event::Completed } else { Event::None };
        StepOutcome::new(reward, done, event, false)
    }
}

/// Plans an episode: [`ASKS_PER_BUCKET`] show/ask pairs per delay bucket, the longest delays
/// placed first. Between the show of a door and its ask, the door is not shown again, so the
/// delay of an ask is the time since the last show of its door.
fn plan(rng: &mut SmallRng) -> Vec<Frame> {
    let mut frames = vec![Frame::Idle; FRAMES as usize];
    let mut spans: [Vec<(u32, u32)>; 3] = Default::default();
    for &(lo, hi) in BUCKETS.iter().rev() {
        for _ in 0..ASKS_PER_BUCKET {
            for _ in 0..ATTEMPTS {
                let delay = rng.gen_range(lo..=hi);
                let show = rng.gen_range(0..FRAMES - delay);
                let ask = show + delay;
                let door = rng.gen_range(0..DOORS.len());
                let free =
                    frames[show as usize] == Frame::Idle && frames[ask as usize] == Frame::Idle;
                let clear = spans[door].iter().all(|&(s, a)| ask < s || a < show);
                if !(free && clear) {
                    continue;
                }
                let colour = rng.gen_range(0..COLOURS.len()) as u8;
                let door8 = door as u8;
                frames[show as usize] = Frame::Show {
                    door: door8,
                    colour,
                };
                frames[ask as usize] = Frame::Ask {
                    door: door8,
                    colour,
                    delay: delay as u8,
                };
                spans[door].push((show, ask));
                break;
            }
        }
    }
    frames
}

impl DecisionGame for Recall {
    fn name(&self) -> &'static str {
        NAME
    }
    fn reset(&mut self) {
        Recall::reset(self)
    }
    fn done(&self) -> bool {
        self.t >= FRAMES
    }
    fn decision_due(&self) -> bool {
        true
    }
    fn questions(&self) -> &'static [QuestionSpec] {
        QUESTIONS
    }
    fn state_text(&self) -> String {
        Recall::state_text(self)
    }
    fn oracle(&self) -> Vec<Vec<f32>> {
        Recall::oracle(self)
    }
    fn apply(&mut self, choice: usize, _abort: bool) -> StepOutcome {
        self.step(choice)
    }
    fn tick(&mut self) -> StepOutcome {
        self.step(0)
    }
    fn stats(&self) -> EpisodeStats {
        EpisodeStats {
            score: self.asks.iter().filter(|a| a.correct).count() as f32,
            max_score: self.planned_asks() as f32,
            success: self.t >= FRAMES,
            steps: self.t,
            ..EpisodeStats::default()
        }
    }
    fn snapshot_json(&self) -> serde_json::Value {
        serde_json::json!({ "frame": self.t, "state": self.state_text(), "asks": self.asks })
    }
}

const fn colour(
    key: &'static str,
    en: &'static [&'static str],
    fr: &'static [&'static str],
) -> OptionSpec {
    // No synonyms: the option must keep the word the show frame used.
    OptionSpec {
        key,
        synonyms: &[],
        desc_en: en,
        desc_fr: fr,
    }
}

pub static QUESTIONS: &[QuestionSpec] = &[QuestionSpec {
    id: "action",
    qtype: QType::Choice,
    ins_en: &[
        "Which key opens the door that is asked for?",
        "Name the colour of the door being asked, or wait when no door is asked.",
        "A door is asked: which colour was it given? Wait otherwise.",
    ],
    ins_fr: &[
        "Quelle clé ouvre la porte demandée ?",
        "Donne la couleur de la porte demandée, ou attends si aucune porte n'est demandée.",
    ],
    options: &[
        OptionSpec {
            key: "wait",
            synonyms: &["hold", "idle", "attendre"],
            desc_en: &["no door is asked", "nothing to answer on this frame"],
            desc_fr: &[
                "aucune porte n'est demandée",
                "rien à répondre sur cette image",
            ],
        },
        colour(
            "red",
            &["the door was given red", "the red key"],
            &["la porte a reçu le rouge", "la clé rouge"],
        ),
        colour(
            "green",
            &["the door was given green", "the green key"],
            &["la porte a reçu le vert", "la clé verte"],
        ),
        colour(
            "blue",
            &["the door was given blue", "the blue key"],
            &["la porte a reçu le bleu", "la clé bleue"],
        ),
        colour(
            "yellow",
            &["the door was given yellow", "the yellow key"],
            &["la porte a reçu le jaune", "la clé jaune"],
        ),
    ],
}];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::argmax;

    #[test]
    fn plans_two_asks_per_bucket_and_keeps_the_delay_honest() {
        let mut full = 0;
        for seed in 0..500 {
            let g = Recall::new(seed);
            // Replays the plan: the colour of an ask is the last one its door was given, and
            // the delay is the time since that show.
            let mut last: [Option<(u32, u8)>; 3] = [None; 3];
            let mut per_bucket = [0usize; 4];
            for (t, f) in g.frames.iter().enumerate() {
                match *f {
                    Frame::Show { door, colour } => last[door as usize] = Some((t as u32, colour)),
                    Frame::Ask {
                        door,
                        colour,
                        delay,
                    } => {
                        let (shown, c) = last[door as usize].expect("shown before asked");
                        assert_eq!(c, colour);
                        assert_eq!(t as u32 - shown, delay as u32);
                        per_bucket[bucket_of(delay as u32)] += 1;
                    }
                    Frame::Idle => {}
                }
            }
            assert!(per_bucket.iter().all(|&n| n <= ASKS_PER_BUCKET));
            full += (g.planned_asks() == BUCKETS.len() * ASKS_PER_BUCKET) as usize;
        }
        assert!(full >= 495, "{full} of 500 episodes have every ask");
    }

    #[test]
    fn the_oracle_answers_every_ask_and_waiting_answers_none() {
        let mut g = Recall::new(7);
        while !DecisionGame::done(&g) {
            let t = g.oracle();
            g.step(argmax(&t[0]));
        }
        let s = DecisionGame::stats(&g);
        assert_eq!(s.score, s.max_score);
        assert!(s.max_score >= 6.0 && s.success && s.steps == FRAMES);
        assert!(g.asks().iter().all(|a| a.correct));

        let mut g = Recall::new(7);
        while !DecisionGame::done(&g) {
            g.step(0);
        }
        assert_eq!(DecisionGame::stats(&g).score, 0.0);
    }

    #[test]
    fn an_ask_frame_does_not_show_the_colour() {
        let mut g = Recall::new(3);
        let mut asks = 0;
        while !DecisionGame::done(&g) {
            let text = g.state_text();
            let keys: Vec<&str> = text.split(' ').step_by(2).collect();
            assert_eq!(keys, ["show", "colour", "ask", "noise"]);
            let t = g.oracle();
            if t[0][0] == 0.0 {
                asks += 1;
                assert!(text.starts_with("show none colour none ask "), "{text}");
            }
            g.step(argmax(&t[0]));
        }
        assert!(asks >= 6);
    }

    #[test]
    fn same_seed_same_episode_and_reset_draws_another() {
        let (a, mut b) = (Recall::new(11), Recall::new(11));
        assert_eq!(a.frames, b.frames);
        assert_eq!(a.noise, b.noise);
        b.reset();
        assert_ne!(a.frames, b.frames);
        assert_ne!(Recall::new(12).frames, a.frames);
    }

    #[test]
    fn the_question_has_one_option_per_answer() {
        let q = &QUESTIONS[0];
        assert_eq!(q.keys(), ["wait", "red", "green", "blue", "yellow"]);
        assert_eq!(q.canonical_options().len(), 5);
        assert_eq!(Recall::new(1).oracle()[0].len(), 5);
    }
}
