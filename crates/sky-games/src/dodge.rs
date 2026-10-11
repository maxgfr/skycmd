//! "Obstacle Dodge": a side-scroller where the drone flies forward through a course of
//! obstacles. Each obstacle has a height, may sit under a deck (a low ceiling such as a bridge)
//! and may leave a lateral gap on one side. The agent picks a maneuver every
//! [`DECISION_EVERY`] frames: fly `over`, go `around` through the gap, `brake`, or `continue`
//! straight and level. A secondary `abort` question stops the mission in front of an
//! impassable obstacle. A coded safety reflex ([`Dodge::veto`]) overrides maneuvers that would
//! collide within the stopping horizon.
//!
//! Units: meters and seconds. `x` forward, `y` lateral (corridor `|y| <= Y_MAX`), `z` up.

use crate::common::{
    bin, bin_floor, mix_seed, sigmoid, with_floor, DecisionGame, EpisodeStats, Event, OptionSpec,
    QuestionSpec, StepOutcome, DT,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use sky_schema::QType;

pub const NAME: &str = "dodge";
pub const CRUISE_V: f32 = 11.0;
pub const ACCEL: f32 = 3.0;
pub const DECEL: f32 = 7.0;
/// Maximum vertical and lateral speeds.
pub const CLIMB: f32 = 4.0;
pub const LAT: f32 = 4.0;
/// Position gain of the altitude/lateral autopilot.
const K_POS: f32 = 3.0;
/// Drone half extents: length (x), width (y), height (z).
pub const HL: f32 = 0.5;
pub const HW: f32 = 0.5;
pub const HH: f32 = 0.3;
/// Required clearance around obstacles.
pub const MARGIN: f32 = 0.5;
pub const Y_MAX: f32 = 5.0;
/// Airspace ceiling (used as the ceiling of obstacles without a deck).
pub const Z_MAX: f32 = 18.0;
pub const Z_CRUISE: f32 = 2.0;
const Z_TARGET_MIN: f32 = 1.0;
const Z_TARGET_MAX: f32 = Z_MAX - HH - 0.5;
/// Decks overhang their obstacle by this much on both sides.
const DECK_OVERHANG: f32 = 1.0;
pub const N_OBSTACLES: usize = 10;
pub const DECISION_EVERY: u32 = 6;
pub const MAX_FRAMES: u32 = 60 * 120;
/// Reaction allowance added to maneuver time estimates.
const T_REACT: f32 = 0.35;
/// Below this speed, time estimates use this speed (the drone can always creep forward).
const V_EFF_MIN: f32 = 4.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DodgeAction {
    Over,
    Around,
    Brake,
    Continue,
    /// Answer "yes" to the abort question: brake to a stop and land.
    Abort,
}

impl DodgeAction {
    /// Canonical `maneuver` options, in order.
    pub const MANEUVERS: [DodgeAction; 4] = [
        DodgeAction::Over,
        DodgeAction::Around,
        DodgeAction::Brake,
        DodgeAction::Continue,
    ];

    pub fn from_index(i: usize) -> DodgeAction {
        Self::MANEUVERS[i.min(3)]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Obstacle {
    /// Front face.
    pub x0: f32,
    /// Depth along x.
    pub len: f32,
    /// Height of the obstacle.
    pub top: f32,
    /// Underside of the deck above the obstacle ([`Z_MAX`] without a deck).
    pub ceil: f32,
    pub deck: bool,
    /// Width of the lateral gap (0 = none).
    pub gap: f32,
    /// Side of the gap: +1 left (`y > 0`), -1 right.
    pub side: f32,
}

impl Obstacle {
    fn x1(&self) -> f32 {
        self.x0 + self.len
    }
    /// End of the obstacle including its deck overhang: the obstacle is passed beyond it.
    fn x_end(&self) -> f32 {
        self.x1() + if self.deck { DECK_OVERHANG } else { 0.0 }
    }
    fn gap_range(&self) -> (f32, f32) {
        if self.side > 0.0 {
            (Y_MAX - self.gap, Y_MAX)
        } else {
            (-Y_MAX, -Y_MAX + self.gap)
        }
    }
    fn gap_center(&self) -> f32 {
        self.side * (Y_MAX - self.gap * 0.5)
    }
    /// Free vertical slot between obstacle + margin and deck - margin (negative = too tight).
    fn slot(&self) -> f32 {
        (self.ceil - MARGIN - HH) - (self.top + MARGIN + HH)
    }
    fn over_ok(&self) -> bool {
        self.slot() >= 0.0
    }
    fn around_ok(&self) -> bool {
        self.gap >= 2.0 * (HW + MARGIN)
    }
    pub fn impassable(&self) -> bool {
        !self.over_ok() && !self.around_ok()
    }
    /// Target altitude to fly over it.
    fn z_over(&self) -> f32 {
        let lo = self.top + MARGIN + HH;
        let hi = self.ceil - MARGIN - HH;
        (lo + 0.25)
            .min(0.5 * (lo + hi))
            .clamp(Z_TARGET_MIN, Z_TARGET_MAX)
    }
}

/// Render data for a canvas (side view `x`/`z` plus a top strip `x`/`y`).
#[derive(Debug, Clone, Serialize)]
pub struct DodgeSnapshot {
    pub drone: DroneView,
    pub obstacles: Vec<ObstacleView>,
    pub finish_x: f32,
    pub y_max: f32,
    pub z_max: f32,
    pub frame: u32,
    pub passed: u32,
    /// The maneuver being executed.
    pub action: DodgeAction,
    /// True when the last decision was overridden by the safety reflex.
    pub vetoed: bool,
    /// What the agent asked for when the reflex fired.
    pub vetoed_request: Option<DodgeAction>,
    pub crashed: bool,
    pub finished: bool,
    pub aborted: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DroneView {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub v: f32,
    pub hl: f32,
    pub hw: f32,
    pub hh: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ObstacleView {
    pub x0: f32,
    pub x1: f32,
    pub top: f32,
    /// Deck span and underside, when there is a deck.
    pub deck: Option<(f32, f32, f32)>,
    /// Lateral gap `(y_lo, y_hi)`, when there is one.
    pub gap: Option<(f32, f32)>,
    pub impassable: bool,
    pub passed: bool,
}

#[derive(Debug, Clone)]
pub struct Dodge {
    seed: u64,
    episode: u64,
    rng: SmallRng,
    obstacles: [Obstacle; N_OBSTACLES],
    pub finish_x: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub v: f32,
    tz: f32,
    ty: f32,
    action: DodgeAction,
    /// Index of the first obstacle not yet passed.
    next: usize,
    pub frame: u32,
    pub crashed: bool,
    pub finished: bool,
    pub aborted: bool,
    pub safe_abort: bool,
    pub timed_out: bool,
    pub vetoes: u32,
    /// When false, the safety reflex is disabled.
    pub veto_enabled: bool,
    last_vetoed: Option<DodgeAction>,
}

/// Geometry of the next obstacle relative to the drone.
#[derive(Debug, Clone, Copy)]
struct Geo {
    t_avail: f32,
    clear: bool,
    slack_over: Option<f32>,
    slack_around: Option<f32>,
    can_stop: bool,
}

impl Dodge {
    pub fn new(seed: u64) -> Self {
        let blank = Obstacle {
            x0: 0.0,
            len: 0.0,
            top: 0.0,
            ceil: Z_MAX,
            deck: false,
            gap: 0.0,
            side: 1.0,
        };
        let mut g = Dodge {
            seed,
            episode: 0,
            rng: SmallRng::seed_from_u64(mix_seed(seed, 0)),
            obstacles: [blank; N_OBSTACLES],
            finish_x: 0.0,
            x: 0.0,
            y: 0.0,
            z: Z_CRUISE,
            v: CRUISE_V,
            tz: Z_CRUISE,
            ty: 0.0,
            action: DodgeAction::Continue,
            next: 0,
            frame: 0,
            crashed: false,
            finished: false,
            aborted: false,
            safe_abort: false,
            timed_out: false,
            vetoes: 0,
            veto_enabled: true,
            last_vetoed: None,
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
        self.generate_course();
        self.x = 0.0;
        self.y = 0.0;
        self.z = Z_CRUISE;
        self.v = CRUISE_V;
        self.tz = Z_CRUISE;
        self.ty = 0.0;
        self.action = DodgeAction::Continue;
        self.next = 0;
        self.frame = 0;
        self.crashed = false;
        self.finished = false;
        self.aborted = false;
        self.safe_abort = false;
        self.timed_out = false;
        self.vetoes = 0;
        self.last_vetoed = None;
    }

    fn generate_course(&mut self) {
        let r = &mut self.rng;
        let wall_at = r.gen_bool(0.2).then(|| r.gen_range(3..N_OBSTACLES));
        let mut x = 35.0;
        for i in 0..N_OBSTACLES {
            let len = r.gen_range(3.0..6.0);
            let side = if r.gen_bool(0.5) { 1.0 } else { -1.0 };
            let (top, ceil, deck, gap) = if wall_at == Some(i) {
                // Impassable: deck too low to fly over, gap too narrow (or none).
                let top = r.gen_range(12.0..16.0);
                let gap = if r.gen_bool(0.5) {
                    r.gen_range(0.8..1.4)
                } else {
                    0.0
                };
                (top, top + r.gen_range(0.3..1.0), true, gap)
            } else {
                match r.gen_range(0..100) {
                    // Tower: fly over (sometimes with a gap too narrow to use).
                    0..=29 => {
                        let gap = if r.gen_bool(0.3) {
                            r.gen_range(1.0..1.8)
                        } else {
                            0.0
                        };
                        (r.gen_range(3.0..10.0), Z_MAX, false, gap)
                    }
                    // Tall wall under a deck: go around.
                    30..=54 => {
                        let top = r.gen_range(9.0..15.0);
                        (
                            top,
                            top + r.gen_range(0.4..1.4),
                            true,
                            r.gen_range(2.6..4.0),
                        )
                    }
                    // Low block with a gap: both maneuvers work.
                    55..=74 => {
                        let top = r.gen_range(2.5..6.0);
                        let (ceil, deck) = if r.gen_bool(0.5) {
                            (top + r.gen_range(3.0..6.0), true)
                        } else {
                            (Z_MAX, false)
                        };
                        (top, ceil, deck, r.gen_range(2.6..4.0))
                    }
                    // Low bridge: squeeze over the block under the deck.
                    _ => {
                        let top = r.gen_range(1.2..3.0);
                        (top, top + r.gen_range(2.2..3.5), true, 0.0)
                    }
                }
            };
            self.obstacles[i] = Obstacle {
                x0: x,
                len,
                top,
                ceil,
                deck,
                gap,
                side,
            };
            let spacing = if r.gen_bool(0.25) {
                r.gen_range(16.0..22.0)
            } else {
                r.gen_range(26.0..44.0)
            };
            x += len + spacing;
        }
        self.finish_x = self.obstacles[N_OBSTACLES - 1].x1() + 25.0;
    }

    pub fn obstacles(&self) -> &[Obstacle] {
        &self.obstacles
    }

    pub fn next_obstacle(&self) -> Option<&Obstacle> {
        self.obstacles.get(self.next)
    }

    pub fn done(&self) -> bool {
        self.crashed || self.finished || self.aborted || self.timed_out
    }

    pub fn decision_due(&self) -> bool {
        self.frame.is_multiple_of(DECISION_EVERY)
    }

    /// Sets the held maneuver and the autopilot targets.
    fn set_action(&mut self, action: DodgeAction) {
        self.action = action;
        let o = self.next_obstacle().copied();
        match action {
            DodgeAction::Over => {
                if let Some(o) = o {
                    self.tz = o.z_over();
                }
            }
            DodgeAction::Around => {
                if let Some(o) = o.filter(|o| o.gap > 0.0) {
                    self.ty = o
                        .gap_center()
                        .clamp(-(Y_MAX - HW - 0.05), Y_MAX - HW - 0.05);
                    self.tz = self.tz.min(o.ceil - MARGIN - HH - 0.25).max(Z_TARGET_MIN);
                }
            }
            DodgeAction::Continue => {
                self.tz = self.z;
                self.ty = self.y;
            }
            DodgeAction::Brake | DodgeAction::Abort => {}
        }
    }

    fn collides(&self) -> bool {
        if self.z - HH < 0.0 || self.z + HH > Z_MAX || self.y.abs() + HW > Y_MAX {
            return true;
        }
        let lo = self.next.saturating_sub(1);
        let hi = (self.next + 2).min(N_OBSTACLES);
        self.obstacles[lo..hi].iter().any(|o| {
            let body = self.x + HL > o.x0 && self.x - HL < o.x1() && self.z - HH < o.top && {
                let (g0, g1) = o.gap_range();
                !(o.gap > 0.0 && self.y - HW >= g0 && self.y + HW <= g1)
            };
            let deck = o.deck
                && self.x + HL > o.x0 - DECK_OVERHANG
                && self.x - HL < o.x1() + DECK_OVERHANG
                && self.z + HH > o.ceil;
            body || deck
        })
    }

    /// One physics frame with the held maneuver.
    fn physics(&mut self) -> StepOutcome {
        match self.action {
            DodgeAction::Brake | DodgeAction::Abort => self.v = (self.v - DECEL * DT).max(0.0),
            DodgeAction::Continue => self.v = (self.v + ACCEL * DT).min(CRUISE_V),
            DodgeAction::Over | DodgeAction::Around => {}
        }
        self.z += (K_POS * (self.tz - self.z)).clamp(-CLIMB, CLIMB) * DT;
        self.y += (K_POS * (self.ty - self.y)).clamp(-LAT, LAT) * DT;
        self.x += self.v * DT;
        self.frame += 1;

        if self.collides() {
            self.crashed = true;
            return StepOutcome::new(-10.0, true, Event::Crashed, false);
        }
        let mut reward = 0.0;
        let mut event = Event::None;
        while self.next < N_OBSTACLES && self.obstacles[self.next].x_end() < self.x - HL {
            self.next += 1;
            reward += 1.0;
            event = Event::Passed;
        }
        if self.x >= self.finish_x {
            self.finished = true;
            return StepOutcome::new(reward + 5.0, true, Event::Completed, false);
        }
        if self.action == DodgeAction::Abort && self.v == 0.0 {
            self.aborted = true;
            self.safe_abort = self.next_obstacle().is_some_and(Obstacle::impassable);
            let r = if self.safe_abort { 3.0 } else { -3.0 };
            return StepOutcome::new(reward + r, true, Event::Aborted, false);
        }
        if self.frame >= MAX_FRAMES {
            self.timed_out = true;
            return StepOutcome::new(reward, true, Event::Timeout, false);
        }
        StepOutcome::new(reward, false, event, false)
    }

    /// Advances one frame. When a decision is due, `action` becomes the new maneuver (after the
    /// safety reflex); otherwise the held maneuver continues and `action` is ignored.
    pub fn step(&mut self, action: DodgeAction) -> StepOutcome {
        if self.done() {
            return StepOutcome::new(0.0, true, Event::None, false);
        }
        let mut vetoed = false;
        if self.decision_due() && self.action != DodgeAction::Abort {
            let chosen = match self.veto_enabled.then(|| self.veto(action)).flatten() {
                Some(replacement) => {
                    vetoed = true;
                    self.vetoes += 1;
                    self.last_vetoed = Some(action);
                    replacement
                }
                None => {
                    self.last_vetoed = None;
                    action
                }
            };
            self.set_action(chosen);
        }
        let mut out = self.physics();
        out.info.vetoed = vetoed;
        out
    }

    /// Holds the current maneuver for one frame.
    pub fn tick(&mut self) -> StepOutcome {
        let a = self.action;
        if self.decision_due() {
            // Re-issuing the held maneuver re-targets the next obstacle, without the reflex.
            if self.done() {
                return StepOutcome::new(0.0, true, Event::None, false);
            }
            self.set_action(a);
            return self.physics();
        }
        self.step(a)
    }

    /// True when repeating `action` at every decision stays collision-free for `frames`, or
    /// until the obstacle the decision is about has been passed (later obstacles get their own
    /// decisions).
    fn safe_hold(&self, action: DodgeAction, frames: u32) -> bool {
        let mut sim = self.clone();
        sim.veto_enabled = false;
        for f in 0..frames {
            if f == 0 || sim.decision_due() {
                sim.set_action(action);
            }
            sim.physics();
            if sim.crashed {
                return false;
            }
            if sim.done() || sim.next > self.next {
                return true;
            }
        }
        true
    }

    /// The safety reflex: when `action`, held through the stopping horizon, would collide with
    /// the obstacle ahead, returns the first safe alternative among brake, over, around,
    /// continue (brake if none is safe). Abort is never vetoed.
    pub fn veto(&self, action: DodgeAction) -> Option<DodgeAction> {
        if action == DodgeAction::Abort {
            return None;
        }
        let frames = (((self.v / DECEL) + 0.6) * 60.0).max(30.0) as u32;
        if self.safe_hold(action, frames) {
            return None;
        }
        let alternatives = [
            DodgeAction::Brake,
            DodgeAction::Over,
            DodgeAction::Around,
            DodgeAction::Continue,
        ];
        Some(
            alternatives
                .into_iter()
                .filter(|a| *a != action)
                .find(|a| self.safe_hold(*a, frames))
                .unwrap_or(DodgeAction::Brake),
        )
    }

    fn geo(&self, o: &Obstacle) -> Geo {
        let d = (o.x0 - (self.x + HL)).max(0.0);
        let t_avail = d / self.v.max(V_EFF_MIN);
        let clear_over =
            self.z - HH >= o.top + 0.6 * MARGIN && self.z + HH <= o.ceil - 0.6 * MARGIN;
        let clear_gap = o.around_ok()
            && (self.y - o.gap_center()).abs() <= o.gap * 0.5 - HW - 0.6 * MARGIN
            && self.z + HH <= o.ceil - 0.6 * MARGIN;
        let slack_over = o
            .over_ok()
            .then(|| t_avail - ((o.z_over() - self.z).abs() / CLIMB + T_REACT));
        let slack_around = o.around_ok().then(|| {
            let z_ar = self.z.min(o.ceil - MARGIN - HH - 0.25).max(Z_TARGET_MIN);
            let t = ((o.gap_center() - self.y).abs() / LAT).max((z_ar - self.z).abs() / CLIMB);
            t_avail - (t + T_REACT)
        });
        Geo {
            t_avail,
            clear: clear_over || clear_gap,
            slack_over,
            slack_around,
            can_stop: d > self.v * self.v / (2.0 * DECEL) + 0.5,
        }
    }

    /// Soft targets: `maneuver` over `[over, around, brake, continue]`, `abort` over
    /// `[false, true]`.
    pub fn oracle(&self) -> Vec<Vec<f32>> {
        let Some(o) = self.next_obstacle() else {
            return vec![with_floor(vec![0.0, 0.0, 0.0, 1.0], 0.01), vec![0.95, 0.05]];
        };
        let p_abort = if o.impassable() { 0.95 } else { 0.05 };
        let abort = vec![1.0 - p_abort, p_abort];
        if o.impassable() {
            return vec![with_floor(vec![0.0, 0.0, 1.0, 0.0], 0.01), abort];
        }
        let g = self.geo(o);
        if g.clear {
            return vec![with_floor(vec![0.0, 0.0, 0.0, 1.0], 0.01), abort];
        }
        let best = g
            .slack_over
            .unwrap_or(f32::MIN)
            .max(g.slack_around.unwrap_or(f32::MIN));
        let p_cont = sigmoid((best - 1.0) / 0.15) * sigmoid((g.t_avail - 3.0) / 0.15);
        let p_brake = if g.can_stop && self.v > V_EFF_MIN {
            (1.0 - p_cont) * sigmoid(-best / 0.1)
        } else {
            0.0
        };
        let p_man = 1.0 - p_cont - p_brake;
        let p_over = match (g.slack_over, g.slack_around) {
            (Some(so), Some(sa)) => sigmoid((so - sa) / 0.2),
            (Some(_), None) => 1.0,
            _ => 0.0,
        };
        vec![
            with_floor(
                vec![p_man * p_over, p_man * (1.0 - p_over), p_brake, p_cont],
                0.01,
            ),
            abort,
        ]
    }

    /// The oracle's decision: the argmax maneuver, or abort when the abort target says so.
    pub fn oracle_action(&self) -> DodgeAction {
        let t = self.oracle();
        if t[1][1] > 0.5 {
            DodgeAction::Abort
        } else {
            DodgeAction::from_index(crate::common::argmax(&t[0]))
        }
    }

    /// Obstacles passed.
    pub fn score(&self) -> u32 {
        self.next as u32
    }

    pub fn snapshot(&self) -> DodgeSnapshot {
        DodgeSnapshot {
            drone: DroneView {
                x: self.x,
                y: self.y,
                z: self.z,
                v: self.v,
                hl: HL,
                hw: HW,
                hh: HH,
            },
            obstacles: self
                .obstacles
                .iter()
                .enumerate()
                .map(|(i, o)| ObstacleView {
                    x0: o.x0,
                    x1: o.x1(),
                    top: o.top,
                    deck: o
                        .deck
                        .then(|| (o.x0 - DECK_OVERHANG, o.x1() + DECK_OVERHANG, o.ceil)),
                    gap: (o.gap > 0.0).then(|| o.gap_range()),
                    impassable: o.impassable(),
                    passed: i < self.next,
                })
                .collect(),
            finish_x: self.finish_x,
            y_max: Y_MAX,
            z_max: Z_MAX,
            frame: self.frame,
            passed: self.next as u32,
            action: self.action,
            vetoed: self.last_vetoed.is_some(),
            vetoed_request: self.last_vetoed,
            crashed: self.crashed,
            finished: self.finished,
            aborted: self.aborted,
        }
    }

    pub fn questions() -> &'static [QuestionSpec] {
        QUESTIONS
    }

    /// `d v dz hr gw dy cl` for the next obstacle: distance (m), speed (m/s), climb needed to
    /// fly over (m), free slot over it (m, negative = too tight), usable gap width (m, negative =
    /// too narrow), lateral offset to the gap center (m), clearance under the deck (m).
    pub fn state_text(&self) -> String {
        let Some(o) = self.next_obstacle() else {
            return format!(
                "d 60 v {} dz -20 hr 20 gw 9 dy 0 cl 20",
                bin(self.v, 1.0, 0, 15)
            );
        };
        let d = (o.x0 - (self.x + HL)).max(0.0);
        let dy = if o.gap > 0.0 {
            (o.gap_center() - self.y).abs()
        } else {
            0.0
        };
        format!(
            "d {} v {} dz {} hr {} gw {} dy {} cl {}",
            bin(d, 1.0, 0, 60),
            bin(self.v, 1.0, 0, 15),
            bin(o.z_over() - self.z, 1.0, -20, 20),
            bin_floor(o.slot(), 1.0, -9, 20),
            bin_floor(o.gap - 2.0 * (HW + MARGIN), 1.0, -3, 9),
            bin(dy, 1.0, 0, 10),
            bin_floor(o.ceil - MARGIN - HH - self.z, 1.0, -9, 20),
        )
    }
}

impl DecisionGame for Dodge {
    fn name(&self) -> &'static str {
        NAME
    }
    fn reset(&mut self) {
        Dodge::reset(self)
    }
    fn done(&self) -> bool {
        Dodge::done(self)
    }
    fn decision_due(&self) -> bool {
        Dodge::decision_due(self)
    }
    fn questions(&self) -> &'static [QuestionSpec] {
        QUESTIONS
    }
    fn state_text(&self) -> String {
        Dodge::state_text(self)
    }
    fn oracle(&self) -> Vec<Vec<f32>> {
        Dodge::oracle(self)
    }
    fn apply(&mut self, choice: usize, abort: bool) -> StepOutcome {
        let a = if abort {
            DodgeAction::Abort
        } else {
            DodgeAction::from_index(choice)
        };
        self.step(a)
    }
    fn tick(&mut self) -> StepOutcome {
        Dodge::tick(self)
    }
    fn stats(&self) -> EpisodeStats {
        EpisodeStats {
            score: self.next as f32,
            success: self.finished,
            crashed: self.crashed,
            aborted: self.aborted,
            safe_abort: self.safe_abort,
            vetoes: self.vetoes,
            steps: self.frame,
        }
    }
    fn set_veto(&mut self, on: bool) {
        self.veto_enabled = on;
    }
    fn snapshot_json(&self) -> serde_json::Value {
        serde_json::to_value(self.snapshot()).expect("snapshot serializes")
    }
}

pub static QUESTIONS: &[QuestionSpec] = &[
    QuestionSpec {
        id: "maneuver",
        qtype: QType::Choice,
        ins_en: &[
            "Which maneuver gets the drone safely past the next obstacle?",
            "Pick the maneuver for the obstacle ahead.",
            "How should the drone handle the obstacle in front of it?",
            "Choose how the drone deals with the next obstacle on its path.",
        ],
        ins_fr: &[
            "Quelle manœuvre permet au drone de passer le prochain obstacle sans danger ?",
            "Choisis la manœuvre face à l'obstacle devant.",
            "Comment le drone doit-il gérer l'obstacle devant lui ?",
        ],
        options: &[
            OptionSpec {
                key: "over",
                synonyms: &["climb_over", "above", "dessus"],
                desc_en: &[
                    "climb and fly over it",
                    "pass above the obstacle",
                    "gain height to clear it",
                ],
                desc_fr: &["monter et passer par-dessus", "survoler l'obstacle"],
            },
            OptionSpec {
                key: "around",
                synonyms: &["sidestep", "gap", "contourner"],
                desc_en: &[
                    "slide sideways through the gap",
                    "go around it through the side gap",
                    "use the lateral opening",
                ],
                desc_fr: &[
                    "passer sur le côté par l'ouverture",
                    "contourner par le côté",
                ],
            },
            OptionSpec {
                key: "brake",
                synonyms: &["slow", "stop", "freiner"],
                desc_en: &["slow down to buy time", "decelerate now", "reduce speed"],
                desc_fr: &["ralentir pour gagner du temps", "freiner maintenant"],
            },
            OptionSpec {
                key: "continue",
                synonyms: &["straight", "proceed", "continuer"],
                desc_en: &[
                    "keep flying straight and level",
                    "carry on unchanged",
                    "the path is clear, keep going",
                ],
                desc_fr: &["continuer tout droit", "garder le cap"],
            },
        ],
    },
    QuestionSpec {
        id: "abort",
        qtype: QType::Noul,
        ins_en: &[
            "Should the drone abort the mission because no safe way past the next obstacle exists?",
            "There is no safe maneuver past the obstacle ahead, so the mission must be aborted.",
            "Is the obstacle ahead impassable, requiring the drone to stop and land?",
        ],
        ins_fr: &[
            "Le drone doit-il annuler la mission faute de passage sûr ?",
            "L'obstacle devant est-il infranchissable, obligeant le drone à se poser ?",
        ],
        options: &[
            OptionSpec {
                key: "false",
                synonyms: &[],
                desc_en: &["", "a safe way past exists", "keep the mission going"],
                desc_fr: &["un passage sûr existe", "poursuivre la mission"],
            },
            OptionSpec {
                key: "true",
                synonyms: &[],
                desc_en: &["", "abort and land", "the way is blocked"],
                desc_fr: &["annuler et se poser", "le passage est bloqué"],
            },
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn single(o: Obstacle) -> Dodge {
        let mut g = Dodge::new(1);
        g.obstacles = [Obstacle { x0: 1000.0, ..o }; N_OBSTACLES];
        g.obstacles[0] = o;
        for i in 1..N_OBSTACLES {
            g.obstacles[i].x0 = 1000.0 + 50.0 * i as f32;
        }
        g
    }

    #[test]
    fn tower_means_over_and_wall_means_abort() {
        let tower = Obstacle {
            x0: 20.0,
            len: 4.0,
            top: 6.0,
            ceil: Z_MAX,
            deck: false,
            gap: 0.0,
            side: 1.0,
        };
        let g = single(tower);
        assert_eq!(g.oracle_action(), DodgeAction::Over);
        let wall = Obstacle {
            top: 14.0,
            ceil: 14.5,
            deck: true,
            ..tower
        };
        let g = single(wall);
        assert_eq!(g.oracle_action(), DodgeAction::Abort);
        let gapwall = Obstacle { gap: 3.5, ..wall };
        let g = single(gapwall);
        assert_eq!(g.oracle_action(), DodgeAction::Around);
    }

    #[test]
    fn veto_fires_on_continue_into_a_tower() {
        let tower = Obstacle {
            x0: 14.0,
            len: 4.0,
            top: 6.0,
            ceil: Z_MAX,
            deck: false,
            gap: 0.0,
            side: 1.0,
        };
        let g = single(tower);
        assert!(g.veto(DodgeAction::Continue).is_some());
        assert_eq!(g.veto(DodgeAction::Abort), None);
        let far = single(Obstacle { x0: 80.0, ..tower });
        assert_eq!(far.veto(DodgeAction::Continue), None);
    }
}
