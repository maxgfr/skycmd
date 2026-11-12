//! "Rule 110", the secondary memory probe of `docs/s6-sleep-prereg.md` §3: the sanity check of
//! arXiv 2605.26099 at a small scale.
//!
//! Frame 0 shows the cells of a ring and a number of steps, then the game reports a pause. Each
//! of the next frames hides the cells and asks the value of one cell after that many steps of
//! Rule 110. The answer has to be worked out from what was kept of frame 0.

use crate::common::{DecisionGame, EpisodeStats, Event, OptionSpec, QuestionSpec, StepOutcome};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sky_schema::QType;

pub const NAME: &str = "rule110";
/// Cells of the ring.
pub const CELLS: usize = 16;
/// Question frames after the cue.
pub const QUERIES: usize = 7;
/// Steps of the automaton an episode can ask for.
pub const STEPS: [u32; 2] = [2, 4];
/// Names of the cells in the state text.
const KEYS: [&str; CELLS] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
];

/// One step of Rule 110 on a ring: the new cell is bit `left·4 + centre·2 + right` of 110.
pub fn step_ring(cells: &[bool; CELLS]) -> [bool; CELLS] {
    let mut out = [false; CELLS];
    for (i, o) in out.iter_mut().enumerate() {
        let l = cells[(i + CELLS - 1) % CELLS] as u32;
        let c = cells[i] as u32;
        let r = cells[(i + 1) % CELLS] as u32;
        *o = (110u32 >> (l << 2 | c << 1 | r)) & 1 == 1;
    }
    out
}

#[derive(Debug, Clone)]
pub struct Rule110 {
    rng: SmallRng,
    cells: [bool; CELLS],
    after: [bool; CELLS],
    steps: u32,
    asked: [usize; QUERIES],
    t: u32,
    correct: u32,
}

impl Rule110 {
    pub fn new(seed: u64) -> Self {
        let mut g = Rule110 {
            rng: SmallRng::seed_from_u64(seed ^ 0x7231_3130),
            cells: [false; CELLS],
            after: [false; CELLS],
            steps: 0,
            asked: [0; QUERIES],
            t: 0,
            correct: 0,
        };
        g.reset();
        g
    }

    /// Draws a new episode.
    pub fn reset(&mut self) {
        for c in &mut self.cells {
            *c = self.rng.gen_bool(0.5);
        }
        self.steps = STEPS[self.rng.gen_range(0..STEPS.len())];
        self.after = self.cells;
        for _ in 0..self.steps {
            self.after = step_ring(&self.after);
        }
        let mut order: Vec<usize> = (0..CELLS).collect();
        order.shuffle(&mut self.rng);
        self.asked.copy_from_slice(&order[..QUERIES]);
        self.t = 0;
        self.correct = 0;
    }

    /// The cell the current frame asks, if it is a question frame.
    fn asked_cell(&self) -> Option<usize> {
        (self.t >= 1 && self.t as usize <= QUERIES).then(|| self.asked[self.t as usize - 1])
    }

    pub fn state_text(&self) -> String {
        match self.asked_cell() {
            Some(cell) => format!("ask {} steps {}", KEYS[cell], self.steps),
            None => {
                let mut s = String::new();
                for (k, c) in KEYS.iter().zip(self.cells) {
                    s.push_str(&format!("{k} {} ", c as u8));
                }
                s.push_str(&format!("ask none steps {}", self.steps));
                s
            }
        }
    }

    /// One-hot over `wait`, `off`, `on`.
    pub fn oracle(&self) -> Vec<Vec<f32>> {
        let mut t = vec![0.0; 3];
        match self.asked_cell() {
            Some(cell) => t[1 + self.after[cell] as usize] = 1.0,
            None => t[0] = 1.0,
        }
        vec![t]
    }

    /// Answers the current frame (`0` wait, `1` off, `2` on) and moves to the next one.
    pub fn step(&mut self, choice: usize) -> StepOutcome {
        if self.t as usize > QUERIES {
            return StepOutcome::new(0.0, true, Event::None, false);
        }
        let mut reward = 0.0;
        if let Some(cell) = self.asked_cell() {
            if choice == 1 + self.after[cell] as usize {
                self.correct += 1;
                reward = 1.0;
            }
        }
        self.t += 1;
        let done = self.t as usize > QUERIES;
        let event = if done { Event::Completed } else { Event::None };
        StepOutcome::new(reward, done, event, false)
    }
}

impl DecisionGame for Rule110 {
    fn name(&self) -> &'static str {
        NAME
    }
    fn reset(&mut self) {
        Rule110::reset(self)
    }
    fn done(&self) -> bool {
        self.t as usize > QUERIES
    }
    fn decision_due(&self) -> bool {
        true
    }
    /// The pause comes right after the cue: the cells are gone and no question was asked yet.
    fn pause(&self) -> bool {
        self.t == 1
    }
    fn questions(&self) -> &'static [QuestionSpec] {
        QUESTIONS
    }
    fn state_text(&self) -> String {
        Rule110::state_text(self)
    }
    fn oracle(&self) -> Vec<Vec<f32>> {
        Rule110::oracle(self)
    }
    fn apply(&mut self, choice: usize, _abort: bool) -> StepOutcome {
        self.step(choice)
    }
    fn tick(&mut self) -> StepOutcome {
        self.step(0)
    }
    fn stats(&self) -> EpisodeStats {
        EpisodeStats {
            score: self.correct as f32,
            max_score: QUERIES as f32,
            success: self.t as usize > QUERIES,
            steps: self.t,
            ..EpisodeStats::default()
        }
    }
    fn snapshot_json(&self) -> serde_json::Value {
        serde_json::json!({
            "frame": self.t, "state": self.state_text(), "steps": self.steps, "correct": self.correct,
        })
    }
}

pub static QUESTIONS: &[QuestionSpec] = &[QuestionSpec {
    id: "action",
    qtype: QType::Choice,
    ins_en: &[
        "What is the asked cell after the given steps of Rule 110?",
        "Give the value of the cell that is asked once the automaton has run, or wait when no cell is asked.",
    ],
    ins_fr: &[
        "Que vaut la cellule demandée après les pas indiqués de la règle 110 ?",
        "Donne la valeur de la cellule demandée une fois l'automate déroulé, ou attends si aucune n'est demandée.",
    ],
    options: &[
        OptionSpec {
            key: "wait",
            synonyms: &["hold", "idle", "attendre"],
            desc_en: &["no cell is asked", "nothing to answer on this frame"],
            desc_fr: &["aucune cellule n'est demandée", "rien à répondre sur cette image"],
        },
        OptionSpec {
            key: "off",
            synonyms: &["zero", "dead", "éteinte"],
            desc_en: &["the cell is 0", "the cell ends off"],
            desc_fr: &["la cellule vaut 0", "la cellule finit éteinte"],
        },
        OptionSpec {
            key: "on",
            synonyms: &["one", "alive", "allumée"],
            desc_en: &["the cell is 1", "the cell ends on"],
            desc_fr: &["la cellule vaut 1", "la cellule finit allumée"],
        },
    ],
}];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::argmax;

    fn ring(bits: &str) -> [bool; CELLS] {
        let mut c = [false; CELLS];
        for (i, b) in bits.bytes().enumerate() {
            c[i] = b == b'1';
        }
        c
    }

    #[test]
    fn one_live_cell_grows_to_the_left() {
        // Rule 110 from a single 1: 1, 11, 111, 1101 (reading leftward growth on a ring).
        let s0 = ring("0000000000000001");
        let s1 = step_ring(&s0);
        assert_eq!(s1, ring("0000000000000011"));
        let s2 = step_ring(&s1);
        assert_eq!(s2, ring("0000000000000111"));
        let s3 = step_ring(&s2);
        assert_eq!(s3, ring("0000000000001101"));
    }

    #[test]
    fn the_oracle_answers_every_question_and_the_pause_follows_the_cue() {
        for seed in 0..50 {
            let mut g = Rule110::new(seed);
            assert!(STEPS.contains(&g.steps));
            let mut pauses = Vec::new();
            let mut frame = 0;
            while !DecisionGame::done(&g) {
                if DecisionGame::pause(&g) {
                    pauses.push(frame);
                }
                let text = g.state_text();
                let t = g.oracle();
                if frame == 0 {
                    assert_eq!(t[0][0], 1.0);
                    assert_eq!(text.split(' ').count(), 2 * CELLS + 4, "{text}");
                } else {
                    assert_eq!(t[0][0], 0.0);
                    assert_eq!(text.split(' ').count(), 4, "{text}");
                }
                g.step(argmax(&t[0]));
                frame += 1;
            }
            assert_eq!(frame, 1 + QUERIES);
            assert_eq!(pauses, [1]);
            let s = DecisionGame::stats(&g);
            assert_eq!((s.score, s.max_score), (QUERIES as f32, QUERIES as f32));
        }
    }

    #[test]
    fn the_questions_ask_distinct_cells_and_seeds_differ() {
        let g = Rule110::new(5);
        let mut asked = g.asked.to_vec();
        asked.sort_unstable();
        asked.dedup();
        assert_eq!(asked.len(), QUERIES);
        assert_eq!(Rule110::new(5).cells, g.cells);
        assert_ne!(Rule110::new(6).cells, g.cells);
    }

    #[test]
    fn both_answers_occur() {
        let (mut on, mut total) = (0, 0);
        for seed in 0..200 {
            let g = Rule110::new(seed);
            on += g.after.iter().filter(|&&c| c).count();
            total += CELLS;
        }
        let share = on as f64 / total as f64;
        assert!((0.3..0.75).contains(&share), "{share}");
    }
}
