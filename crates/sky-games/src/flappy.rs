//! "Flappy Drone": a drone at a fixed x, gravity, an upward `thrust` impulse versus `glide`,
//! and scrolling pipe pairs. A decision is taken every frame.
//!
//! Coordinates: pixels, `y` is the altitude above the ground (up is positive), the world is
//! [`WIDTH`] x [`HEIGHT`]. Pipes are stored in world coordinates and scroll with the frame
//! counter, so a lookahead at frame `f + k` is exact.

use crate::common::{
    bin, mix_seed, sigmoid, DecisionGame, EpisodeStats, Event, OptionSpec, QuestionSpec,
    StepOutcome, DT,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use sky_schema::QType;

pub const NAME: &str = "flappy";
pub const WIDTH: f32 = 480.0;
pub const HEIGHT: f32 = 512.0;
pub const DRONE_X: f32 = 120.0;
/// Drone half width / half height.
pub const DRONE_HW: f32 = 14.0;
pub const DRONE_HH: f32 = 10.0;
pub const GRAVITY: f32 = 900.0;
/// Vertical speed set by a thrust impulse.
pub const THRUST_V: f32 = 320.0;
/// Terminal falling speed.
pub const MAX_FALL: f32 = -380.0;
pub const SCROLL: f32 = 130.0;
pub const PIPE_W: f32 = 64.0;
pub const GAP_H: f32 = 150.0;
pub const PIPE_SPACING: f32 = 230.0;
const FIRST_PIPE_X: f32 = 420.0;
const GAP_MIN_C: f32 = GAP_H / 2.0 + 40.0;
const GAP_MAX_C: f32 = HEIGHT - GAP_H / 2.0 - 40.0;
const GAP_MAX_DELTA: f32 = 150.0;
const N_PIPES: usize = 4;

/// State-text scale: one unit is 8 px.
const UNIT: f32 = 8.0;

// Oracle search parameters.
const HORIZON: usize = 40;
/// Minimum frames between two thrusts inside a lookahead plan (after the first action).
const MIN_THRUST_GAP: u32 = 8;
const MARGIN_CAP: f32 = 30.0;
const W_CENTER: f32 = 0.08;
const W_THRUST: f32 = 0.1;
const TARGET_T: f32 = 2.5;
/// The danger question looks half a second ahead (gliding for a full second almost always
/// ends on the ground under this gravity, which would make the label nearly constant).
const DANGER_FRAMES: f32 = 30.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FlappyAction {
    Thrust,
    Glide,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Pipe {
    /// Left edge in world coordinates.
    pub wx: f32,
    /// Gap top (altitude of the lower edge of the upper pipe).
    pub gap_top: f32,
    /// Gap bottom (altitude of the upper edge of the lower pipe).
    pub gap_bottom: f32,
    pub scored: bool,
}

impl Pipe {
    fn center(&self) -> f32 {
        0.5 * (self.gap_top + self.gap_bottom)
    }
}

/// Render data for a canvas: screen coordinates (x right, y = altitude).
#[derive(Debug, Clone, Serialize)]
pub struct FlappySnapshot {
    pub width: f32,
    pub height: f32,
    pub drone: DroneView,
    pub pipes: Vec<PipeView>,
    pub pipes_passed: u32,
    pub frame: u32,
    pub alive: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DroneView {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub vy: f32,
    /// True on the frame of a thrust (for the rotor flame effect).
    pub thrusting: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PipeView {
    pub x: f32,
    pub w: f32,
    pub gap_top: f32,
    pub gap_bottom: f32,
}

#[derive(Debug, Clone)]
pub struct Flappy {
    seed: u64,
    episode: u64,
    rng: SmallRng,
    pub y: f32,
    pub vy: f32,
    pub frame: u32,
    pub pipes_passed: u32,
    pub alive: bool,
    pipes: [Pipe; N_PIPES],
    thrusting: bool,
}

/// Outcome of a frame in the lookahead: `None` on crash, else the safety margin.
#[inline]
fn integrate(y: f32, vy: f32, thrust: bool) -> (f32, f32) {
    let vy = if thrust { THRUST_V } else { vy };
    let vy = (vy - GRAVITY * DT).max(MAX_FALL);
    (y + vy * DT, vy)
}

impl Flappy {
    pub fn new(seed: u64) -> Self {
        let mut g = Flappy {
            seed,
            episode: 0,
            rng: SmallRng::seed_from_u64(mix_seed(seed, 0)),
            y: 0.0,
            vy: 0.0,
            frame: 0,
            pipes_passed: 0,
            alive: true,
            pipes: [Pipe {
                wx: 0.0,
                gap_top: 0.0,
                gap_bottom: 0.0,
                scored: false,
            }; N_PIPES],
            thrusting: false,
        };
        g.start();
        g
    }

    /// Starts the next episode (a new course).
    pub fn reset(&mut self) {
        self.episode += 1;
        self.rng = SmallRng::seed_from_u64(mix_seed(self.seed, self.episode));
        self.start();
    }

    fn start(&mut self) {
        self.y = HEIGHT * 0.5;
        self.vy = 0.0;
        self.frame = 0;
        self.pipes_passed = 0;
        self.alive = true;
        self.thrusting = false;
        let mut c = HEIGHT * 0.5;
        for i in 0..N_PIPES {
            c = self.next_center(c);
            self.pipes[i] = Self::make_pipe(FIRST_PIPE_X + i as f32 * PIPE_SPACING, c);
        }
    }

    fn next_center(&mut self, prev: f32) -> f32 {
        let lo = (prev - GAP_MAX_DELTA).max(GAP_MIN_C);
        let hi = (prev + GAP_MAX_DELTA).min(GAP_MAX_C);
        self.rng.gen_range(lo..=hi)
    }

    fn make_pipe(wx: f32, c: f32) -> Pipe {
        Pipe {
            wx,
            gap_top: c + GAP_H * 0.5,
            gap_bottom: c - GAP_H * 0.5,
            scored: false,
        }
    }

    #[inline]
    fn scroll_at(frame: u32) -> f32 {
        frame as f32 * SCROLL * DT
    }

    /// Screen x of a pipe's left edge at a given frame.
    #[inline]
    fn pipe_x(p: &Pipe, frame: u32) -> f32 {
        p.wx - Self::scroll_at(frame)
    }

    /// Safety margin of the drone at altitude `y` at `frame`: negative means a collision.
    #[inline]
    fn margin(&self, y: f32, frame: u32) -> f32 {
        let mut m = (y - DRONE_HH).min(HEIGHT - DRONE_HH - y);
        for p in &self.pipes {
            let x = Self::pipe_x(p, frame);
            if x < DRONE_X + DRONE_HW && x + PIPE_W > DRONE_X - DRONE_HW {
                m = m.min((y - DRONE_HH - p.gap_bottom).min(p.gap_top - (y + DRONE_HH)));
            }
        }
        m
    }

    /// The pipe the drone is in or approaching at `frame`.
    fn upcoming(&self, frame: u32) -> &Pipe {
        self.pipes
            .iter()
            .find(|p| Self::pipe_x(p, frame) + PIPE_W > DRONE_X - DRONE_HW)
            .unwrap_or(&self.pipes[N_PIPES - 1])
    }

    pub fn step(&mut self, action: FlappyAction) -> StepOutcome {
        if !self.alive {
            return StepOutcome::new(0.0, true, Event::Crashed, false);
        }
        self.thrusting = action == FlappyAction::Thrust;
        let (y, vy) = integrate(self.y, self.vy, self.thrusting);
        self.y = y;
        self.vy = vy;
        self.frame += 1;

        let mut reward = 0.01;
        let mut event = Event::None;
        for p in self.pipes.iter_mut() {
            if !p.scored && Self::pipe_x(p, self.frame) + PIPE_W < DRONE_X - DRONE_HW {
                p.scored = true;
                self.pipes_passed += 1;
                reward += 1.0;
                event = Event::Passed;
            }
        }
        if Self::pipe_x(&self.pipes[0], self.frame) + PIPE_W < -16.0 {
            self.pipes.rotate_left(1);
            let last = self.pipes[N_PIPES - 2];
            let c = self.next_center(last.center());
            self.pipes[N_PIPES - 1] = Self::make_pipe(last.wx + PIPE_SPACING, c);
        }
        if self.margin(self.y, self.frame) < 0.0 {
            self.alive = false;
            return StepOutcome::new(-1.0, true, Event::Crashed, false);
        }
        StepOutcome::new(reward, false, event, false)
    }

    pub fn score(&self) -> u32 {
        self.pipes_passed
    }

    pub fn snapshot(&self) -> FlappySnapshot {
        FlappySnapshot {
            width: WIDTH,
            height: HEIGHT,
            drone: DroneView {
                x: DRONE_X,
                y: self.y,
                w: 2.0 * DRONE_HW,
                h: 2.0 * DRONE_HH,
                vy: self.vy,
                thrusting: self.thrusting,
            },
            pipes: self
                .pipes
                .iter()
                .map(|p| PipeView {
                    x: Self::pipe_x(p, self.frame),
                    w: PIPE_W,
                    gap_top: p.gap_top,
                    gap_bottom: p.gap_bottom,
                })
                .collect(),
            pipes_passed: self.pipes_passed,
            frame: self.frame,
            alive: self.alive,
        }
    }

    pub fn questions() -> &'static [QuestionSpec] {
        QUESTIONS
    }

    /// `vy dx up dn nxt alt`: vertical speed, distance to the next gap, clearance above and
    /// below the drone inside that gap, offset of the following gap's center, and altitude,
    /// in 8 px units (vy in 25 px/s units).
    pub fn state_text(&self) -> String {
        let i = self
            .pipes
            .iter()
            .position(|p| Self::pipe_x(p, self.frame) + PIPE_W > DRONE_X - DRONE_HW)
            .unwrap_or(N_PIPES - 2);
        let p = &self.pipes[i];
        let next = &self.pipes[(i + 1).min(N_PIPES - 1)];
        let dx = Self::pipe_x(p, self.frame) - (DRONE_X + DRONE_HW);
        // Clearances are relative to the drone, so the model reads "room above / below" directly
        // instead of subtracting absolute heights (the code measures, the model decides).
        format!(
            "vy {} dx {} up {} dn {} nxt {} alt {}",
            bin(self.vy, 25.0, -16, 16),
            bin(dx, UNIT, -12, 60),
            bin(p.gap_top - (self.y + DRONE_HH), UNIT, -20, 20),
            bin((self.y - DRONE_HH) - p.gap_bottom, UNIT, -20, 20),
            bin(next.center() - p.center(), UNIT, -20, 20),
            bin(self.y, UNIT, 0, 64),
        )
    }

    /// Best plan value over the lookahead tree, starting with `first`.
    fn plan_value(&self, first: FlappyAction) -> f32 {
        let thrust = first == FlappyAction::Thrust;
        let (y, vy) = integrate(self.y, self.vy, thrust);
        let m = self.margin(y, self.frame + 1);
        if m < 0.0 {
            return -1000.0;
        }
        let mut best = f32::NEG_INFINITY;
        let since = if thrust { 0 } else { MIN_THRUST_GAP };
        self.search(y, vy, 1, since, m, thrust as u32, &mut best);
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        &self,
        y: f32,
        vy: f32,
        k: usize,
        since: u32,
        run_min: f32,
        thrusts: u32,
        best: &mut f32,
    ) {
        let frame = self.frame + k as u32;
        if k == HORIZON {
            let c = self.upcoming(frame).center();
            let v = run_min.min(MARGIN_CAP) - W_CENTER * (y - c).abs() - W_THRUST * thrusts as f32;
            if v > *best {
                *best = v;
            }
            return;
        }
        if run_min.min(MARGIN_CAP) <= *best {
            return;
        }
        // Explore the action that moves toward the upcoming gap center first (better pruning).
        let c = self.upcoming(frame).center();
        let order = if y < c - 20.0 {
            [true, false]
        } else {
            [false, true]
        };
        for thrust in order {
            if thrust && since < MIN_THRUST_GAP {
                continue;
            }
            let (y2, vy2) = integrate(y, vy, thrust);
            let m = self.margin(y2, frame + 1);
            if m < 0.0 {
                let v = -1000.0 + k as f32;
                if v > *best {
                    *best = v;
                }
                continue;
            }
            let since2 = if thrust { 0 } else { since.saturating_add(1) };
            self.search(
                y2,
                vy2,
                k + 1,
                since2,
                run_min.min(m),
                thrusts + thrust as u32,
                best,
            );
        }
    }

    /// Frames until a crash when gliding from now on (`None` within the horizon).
    fn glide_crash_frame(&self, horizon: u32) -> Option<u32> {
        let (mut y, mut vy) = (self.y, self.vy);
        for k in 1..=horizon {
            (y, vy) = integrate(y, vy, false);
            if self.margin(y, self.frame + k) < 0.0 {
                return Some(k);
            }
        }
        None
    }

    /// Lookahead values of `[thrust, glide]`.
    pub fn action_values(&self) -> [f32; 2] {
        [
            self.plan_value(FlappyAction::Thrust),
            self.plan_value(FlappyAction::Glide),
        ]
    }

    /// Soft targets: `action` over `[thrust, glide]`, `danger` over `[false, true]`.
    pub fn oracle(&self) -> Vec<Vec<f32>> {
        let [vt, vg] = self.action_values();
        let p_thrust = sigmoid(((vt - vg) / TARGET_T).clamp(-3.0, 3.0));
        let crash = self.glide_crash_frame(60).map_or(60.0, |f| f as f32);
        let p_danger = sigmoid(((DANGER_FRAMES + 0.5 - crash) / 2.0).clamp(-3.0, 3.0));
        vec![
            vec![p_thrust, 1.0 - p_thrust],
            vec![1.0 - p_danger, p_danger],
        ]
    }

    pub fn oracle_action(&self) -> FlappyAction {
        let [vt, vg] = self.action_values();
        if vt > vg {
            FlappyAction::Thrust
        } else {
            FlappyAction::Glide
        }
    }
}

impl DecisionGame for Flappy {
    fn name(&self) -> &'static str {
        NAME
    }
    fn reset(&mut self) {
        Flappy::reset(self)
    }
    fn done(&self) -> bool {
        !self.alive
    }
    fn decision_due(&self) -> bool {
        true
    }
    fn questions(&self) -> &'static [QuestionSpec] {
        QUESTIONS
    }
    fn state_text(&self) -> String {
        Flappy::state_text(self)
    }
    fn oracle(&self) -> Vec<Vec<f32>> {
        Flappy::oracle(self)
    }
    fn apply(&mut self, choice: usize, _abort: bool) -> StepOutcome {
        self.step(if choice == 0 {
            FlappyAction::Thrust
        } else {
            FlappyAction::Glide
        })
    }
    fn tick(&mut self) -> StepOutcome {
        self.step(FlappyAction::Glide)
    }
    fn stats(&self) -> EpisodeStats {
        EpisodeStats {
            score: self.pipes_passed as f32,
            success: false,
            crashed: !self.alive,
            steps: self.frame,
            ..EpisodeStats::default()
        }
    }
    fn snapshot_json(&self) -> serde_json::Value {
        serde_json::to_value(self.snapshot()).expect("snapshot serializes")
    }
}

pub static QUESTIONS: &[QuestionSpec] = &[
    QuestionSpec {
        id: "action",
        qtype: QType::Choice,
        ins_en: &[
            "Which input keeps the drone flying through the next gap?",
            "Pick the control input for this frame so the drone clears the pipes.",
            "What should the drone do right now to pass the next gap safely?",
            "Choose the action that keeps the drone centered in the upcoming gap.",
        ],
        ins_fr: &[
            "Quelle commande garde le drone dans le prochain passage ?",
            "Choisis l'action de cette image pour que le drone franchisse les tuyaux.",
            "Que doit faire le drone maintenant pour passer le prochain trou sans danger ?",
        ],
        options: &[
            OptionSpec {
                key: "thrust",
                synonyms: &["climb", "up", "boost", "monter"],
                desc_en: &[
                    "fire the rotors for an upward impulse",
                    "push the drone upward",
                    "gain altitude now",
                ],
                desc_fr: &["donner une impulsion vers le haut", "faire monter le drone"],
            },
            OptionSpec {
                key: "glide",
                synonyms: &["fall", "coast", "drift", "planer"],
                desc_en: &[
                    "let gravity pull the drone down",
                    "do nothing and sink",
                    "keep coasting without thrust",
                ],
                desc_fr: &[
                    "laisser la gravité faire descendre le drone",
                    "ne rien faire et descendre",
                ],
            },
        ],
    },
    QuestionSpec {
        id: "danger",
        qtype: QType::Noul,
        ins_en: &[
            "Will the drone crash within the next half second if it keeps gliding?",
            "If no thrust is applied for half a second, does the drone hit something?",
            "The drone collides within half a second unless it thrusts.",
        ],
        ins_fr: &[
            "Le drone va-t-il s'écraser dans la demi-seconde s'il continue de planer ?",
            "Sans poussée pendant une demi-seconde, le drone heurte-t-il un obstacle ?",
        ],
        options: &[
            OptionSpec {
                key: "false",
                synonyms: &[],
                desc_en: &["", "the drone stays clear", "no crash ahead"],
                desc_fr: &["le drone reste dégagé", "pas de collision"],
            },
            OptionSpec {
                key: "true",
                synonyms: &[],
                desc_en: &["", "the drone crashes", "a collision is coming"],
                desc_fr: &["le drone s'écrase", "une collision approche"],
            },
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_without_thrust() {
        let mut g = Flappy::new(1);
        let mut frames = 0;
        while !g.step(FlappyAction::Glide).done {
            frames += 1;
            assert!(frames < 200);
        }
        assert!(!g.alive);
    }

    #[test]
    fn oracle_prefers_thrust_when_low() {
        let mut g = Flappy::new(3);
        g.y = 40.0;
        g.vy = -200.0;
        let t = g.oracle();
        // Thrusting one frame later still survives, so the target is soft but favors thrust.
        assert!(t[0][0] > 0.75, "{t:?}");
        assert!(t[1][1] > 0.9, "{t:?}");
    }

    #[test]
    fn oracle_prefers_glide_when_high() {
        let mut g = Flappy::new(3);
        g.y = HEIGHT - 40.0;
        g.vy = 200.0;
        let t = g.oracle();
        assert!(t[0][1] > 0.9, "{t:?}");
    }
}
