//! "Rescue Run": a 12x12 grid with rocks, wind cells, a base and 3 hikers. Every move drains
//! the battery (3 units against the wind, 1 otherwise, hovering 1); the base recharges it.
//! Entering a hiker's cell rescues the hiker; the mission ends when all hikers are rescued and
//! the drone is back at base. The `sensor` sometimes drops out and hides the hiker data.

use crate::common::{
    mix_seed, with_floor, DecisionGame, EpisodeStats, Event, OptionSpec, QuestionSpec, StepOutcome,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use sky_schema::QType;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub const NAME: &str = "rescue";
pub const N: usize = 12;
const CELLS: usize = N * N;
pub const N_HIKERS: usize = 3;
pub const BATTERY_MAX: i32 = 40;
/// Battery reserve the oracle keeps on top of a hiker trip.
pub const MARGIN: i32 = 3;
pub const MAX_STEPS: u32 = 240;
/// Probability that the hiker sensor drops out on a given step.
pub const DROPOUT: f32 = 0.12;
const INF: i16 = 999;

/// Cell codes of [`RescueSnapshot::cells`].
pub const EMPTY: u8 = 0;
pub const ROCK: u8 = 1;
/// Wind blowing toward north/south/east/west (codes 2..=5).
pub const WIND_N: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Dir {
    N,
    S,
    E,
    W,
}

const DIRS: [Dir; 4] = [Dir::N, Dir::S, Dir::E, Dir::W];

impl Dir {
    fn opposite(self) -> Dir {
        match self {
            Dir::N => Dir::S,
            Dir::S => Dir::N,
            Dir::E => Dir::W,
            Dir::W => Dir::E,
        }
    }
    fn index(self) -> usize {
        self as usize
    }
    fn letter(self) -> &'static str {
        ["n", "s", "e", "w"][self as usize]
    }
}

/// Canonical options of the `move` question, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RescueAction {
    North,
    South,
    East,
    West,
    Hover,
    ReturnHome,
    Unknown,
}

impl RescueAction {
    pub const ALL: [RescueAction; 7] = [
        RescueAction::North,
        RescueAction::South,
        RescueAction::East,
        RescueAction::West,
        RescueAction::Hover,
        RescueAction::ReturnHome,
        RescueAction::Unknown,
    ];

    pub fn from_index(i: usize) -> RescueAction {
        Self::ALL[i.min(6)]
    }

    pub fn index(self) -> usize {
        self as usize
    }

    fn dir(self) -> Option<Dir> {
        match self {
            RescueAction::North => Some(Dir::N),
            RescueAction::South => Some(Dir::S),
            RescueAction::East => Some(Dir::E),
            RescueAction::West => Some(Dir::W),
            _ => None,
        }
    }
}

/// Render data for a canvas.
#[derive(Debug, Clone, Serialize)]
pub struct RescueSnapshot {
    pub n: usize,
    /// Row-major cell codes: 0 empty, 1 rock, 2..=5 wind toward N/S/E/W.
    pub cells: Vec<u8>,
    /// (row, col) of the drone, and where it was before the last step (for tweening).
    pub drone: (usize, usize),
    pub prev: (usize, usize),
    pub base: (usize, usize),
    /// (row, col, rescued).
    pub hikers: Vec<(usize, usize, bool)>,
    pub battery: i32,
    pub battery_max: i32,
    pub sensor_ok: bool,
    pub rescued: u32,
    pub steps: u32,
    pub last_action: RescueAction,
    pub vetoed: bool,
    pub crashed: bool,
    pub completed: bool,
}

#[derive(Debug, Clone, Copy)]
struct MoveFeature {
    blocked: bool,
    /// Cost of the cheapest "rescue a hiker then fly home" trip through this move.
    trip: i32,
    /// Cost of flying home through this move.
    home: i32,
}

#[derive(Debug, Clone)]
pub struct Rescue {
    seed: u64,
    episode: u64,
    rng: SmallRng,
    cells: [u8; CELLS],
    base: usize,
    hikers: [usize; N_HIKERS],
    rescued: [bool; N_HIKERS],
    pub pos: usize,
    prev: usize,
    pub battery: i32,
    pub steps: u32,
    pub sensor_ok: bool,
    /// Cost to reach the base from each cell.
    base_map: [i16; CELLS],
    /// Cost to reach each hiker from each cell.
    hiker_maps: [[i16; CELLS]; N_HIKERS],
    pub crashed: bool,
    pub completed: bool,
    pub timed_out: bool,
    pub vetoes: u32,
    /// When false, the safety reflex is disabled (unsafe actions are executed).
    pub veto_enabled: bool,
    last_action: RescueAction,
    last_vetoed: bool,
}

fn rc(i: usize) -> (usize, usize) {
    (i / N, i % N)
}

fn neighbor(i: usize, d: Dir) -> Option<usize> {
    let (r, c) = rc(i);
    match d {
        Dir::N if r > 0 => Some(i - N),
        Dir::S if r + 1 < N => Some(i + N),
        Dir::E if c + 1 < N => Some(i + 1),
        Dir::W if c > 0 => Some(i - 1),
        _ => None,
    }
}

fn wind_of(cell: u8) -> Option<Dir> {
    (WIND_N..WIND_N + 4)
        .contains(&cell)
        .then(|| DIRS[(cell - WIND_N) as usize])
}

impl Rescue {
    pub fn new(seed: u64) -> Self {
        let mut g = Rescue {
            seed,
            episode: 0,
            rng: SmallRng::seed_from_u64(mix_seed(seed, 0)),
            cells: [EMPTY; CELLS],
            base: 0,
            hikers: [0; N_HIKERS],
            rescued: [false; N_HIKERS],
            pos: 0,
            prev: 0,
            battery: BATTERY_MAX,
            steps: 0,
            sensor_ok: true,
            base_map: [INF; CELLS],
            hiker_maps: [[INF; CELLS]; N_HIKERS],
            crashed: false,
            completed: false,
            timed_out: false,
            vetoes: 0,
            veto_enabled: true,
            last_action: RescueAction::Hover,
            last_vetoed: false,
        };
        g.start();
        g
    }

    /// Starts the next episode (a new map).
    pub fn reset(&mut self) {
        self.episode += 1;
        self.rng = SmallRng::seed_from_u64(mix_seed(self.seed, self.episode));
        self.start();
    }

    fn start(&mut self) {
        loop {
            self.generate_map();
            if self.map_is_feasible() {
                break;
            }
        }
        self.rescued = [false; N_HIKERS];
        self.pos = self.base;
        self.prev = self.base;
        self.battery = BATTERY_MAX;
        self.steps = 0;
        self.sensor_ok = true;
        self.crashed = false;
        self.completed = false;
        self.timed_out = false;
        self.vetoes = 0;
        self.last_action = RescueAction::Hover;
        self.last_vetoed = false;
    }

    fn generate_map(&mut self) {
        self.cells = [EMPTY; CELLS];
        // Wind streaks along rows or columns.
        for _ in 0..3 {
            let d = DIRS[self.rng.gen_range(0..4)];
            let len = self.rng.gen_range(4..=8);
            let mut i = self.rng.gen_range(0..CELLS);
            let along = if self.rng.gen_bool(0.5) {
                d
            } else {
                [Dir::E, Dir::S][(d.index() / 2) % 2]
            };
            for _ in 0..len {
                self.cells[i] = WIND_N + d.index() as u8;
                match neighbor(i, along) {
                    Some(j) => i = j,
                    None => break,
                }
            }
        }
        // Rocks: small clusters and single boulders.
        for _ in 0..4 {
            let mut i = self.rng.gen_range(0..CELLS);
            for _ in 0..self.rng.gen_range(2..=4) {
                self.cells[i] = ROCK;
                match neighbor(i, DIRS[self.rng.gen_range(0..4)]) {
                    Some(j) => i = j,
                    None => break,
                }
            }
        }
        for _ in 0..8 {
            let i = self.rng.gen_range(0..CELLS);
            self.cells[i] = ROCK;
        }
        self.base = self.random_free(&[]);
        for h in 0..N_HIKERS {
            let taken: Vec<usize> = std::iter::once(self.base)
                .chain(self.hikers[..h].iter().copied())
                .collect();
            let mut cell = self.random_free(&taken);
            for _ in 0..20 {
                let (r0, c0) = rc(self.base);
                let (r1, c1) = rc(cell);
                if r0.abs_diff(r1) + c0.abs_diff(c1) >= 4 {
                    break;
                }
                cell = self.random_free(&taken);
            }
            self.hikers[h] = cell;
        }
    }

    fn random_free(&mut self, taken: &[usize]) -> usize {
        loop {
            let i = self.rng.gen_range(0..CELLS);
            if self.cells[i] == EMPTY && !taken.contains(&i) {
                return i;
            }
        }
    }

    fn map_is_feasible(&mut self) -> bool {
        self.base_map = self.cost_to(self.base);
        for h in 0..N_HIKERS {
            self.hiker_maps[h] = self.cost_to(self.hikers[h]);
            let trip = self.hiker_maps[h][self.base] as i32 + self.base_map[self.hikers[h]] as i32;
            if trip + MARGIN > BATTERY_MAX {
                return false;
            }
        }
        true
    }

    /// Battery cost of entering `to` while moving in direction `d`.
    fn enter_cost(&self, to: usize, d: Dir) -> i32 {
        if wind_of(self.cells[to]) == Some(d.opposite()) {
            3
        } else {
            1
        }
    }

    fn blocked(&self, from: usize, d: Dir) -> Option<usize> {
        match neighbor(from, d) {
            Some(j) if self.cells[j] != ROCK => Some(j),
            _ => None,
        }
    }

    /// Cheapest battery cost from every cell to `target` (Dijkstra on the reversed graph).
    fn cost_to(&self, target: usize) -> [i16; CELLS] {
        let mut dist = [INF; CELLS];
        let mut heap = BinaryHeap::new();
        dist[target] = 0;
        heap.push(Reverse((0i16, target)));
        while let Some(Reverse((du, u))) = heap.pop() {
            if du > dist[u] {
                continue;
            }
            for d in DIRS {
                // A predecessor v reaches u by moving in direction d.
                if let Some(v) = neighbor(u, d.opposite()) {
                    if self.cells[v] == ROCK {
                        continue;
                    }
                    let nd = du + self.enter_cost(u, d) as i16;
                    if nd < dist[v] {
                        dist[v] = nd;
                        heap.push(Reverse((nd, v)));
                    }
                }
            }
        }
        dist
    }

    pub fn hikers_left(&self) -> usize {
        self.rescued.iter().filter(|r| !**r).count()
    }

    fn trip_from(&self, cell: usize) -> i32 {
        (0..N_HIKERS)
            .filter(|&h| !self.rescued[h])
            .map(|h| self.hiker_maps[h][cell] as i32 + self.base_map[self.hikers[h]] as i32)
            .min()
            .unwrap_or(INF as i32)
    }

    fn features(&self) -> [MoveFeature; 4] {
        DIRS.map(|d| match self.blocked(self.pos, d) {
            Some(j) => {
                let step = self.enter_cost(j, d);
                MoveFeature {
                    blocked: false,
                    trip: step + self.trip_from(j),
                    home: step + self.base_map[j] as i32,
                }
            }
            None => MoveFeature {
                blocked: true,
                trip: INF as i32,
                home: INF as i32,
            },
        })
    }

    /// Cost to fly home from the current cell.
    pub fn home_cost(&self) -> i32 {
        self.base_map[self.pos] as i32
    }

    fn home_dir(&self) -> Option<Dir> {
        if self.pos == self.base {
            return None;
        }
        let f = self.features();
        DIRS.into_iter()
            .filter(|d| !f[d.index()].blocked)
            .min_by_key(|d| f[d.index()].home)
    }

    pub fn done(&self) -> bool {
        self.crashed || self.completed || self.timed_out
    }

    /// The safety reflex: returns a replacement when `action` flies into a rock or leaves too
    /// little battery to get home.
    pub fn veto(&self, action: RescueAction) -> Option<RescueAction> {
        let (after_pos, cost) = match action {
            RescueAction::ReturnHome => return None,
            RescueAction::Hover | RescueAction::Unknown => (self.pos, 1),
            a => {
                let d = a.dir().expect("move action");
                match self.blocked(self.pos, d) {
                    Some(j) => (j, self.enter_cost(j, d)),
                    None => {
                        let hover_ok = self.pos == self.base || self.battery > self.home_cost();
                        return Some(if hover_ok {
                            RescueAction::Hover
                        } else {
                            RescueAction::ReturnHome
                        });
                    }
                }
            }
        };
        if after_pos == self.base && self.pos == self.base {
            return None;
        }
        if self.battery - cost < self.base_map[after_pos] as i32 {
            return Some(RescueAction::ReturnHome);
        }
        None
    }

    pub fn step(&mut self, action: RescueAction) -> StepOutcome {
        if self.done() {
            return StepOutcome::new(0.0, true, Event::None, false);
        }
        let (action, vetoed) = match self.veto_enabled.then(|| self.veto(action)).flatten() {
            Some(a) => (a, true),
            None => (action, false),
        };
        self.vetoes += vetoed as u32;
        self.last_action = action;
        self.last_vetoed = vetoed;
        self.prev = self.pos;
        self.steps += 1;

        let dir = match action {
            RescueAction::ReturnHome => self.home_dir(),
            a => a.dir(),
        };
        let mut reward = -0.1;
        let mut event = Event::None;
        let cost = match dir {
            Some(d) => match self.blocked(self.pos, d) {
                Some(j) => {
                    let c = self.enter_cost(j, d);
                    self.pos = j;
                    c
                }
                None => {
                    self.crashed = true;
                    return StepOutcome::new(-30.0, true, Event::Crashed, vetoed);
                }
            },
            None => i32::from(self.pos != self.base),
        };
        self.battery -= cost;
        if self.battery < 0 || (self.battery == 0 && self.pos != self.base) {
            self.battery = self.battery.max(0);
            self.crashed = true;
            return StepOutcome::new(-30.0, true, Event::Crashed, vetoed);
        }
        for h in 0..N_HIKERS {
            if !self.rescued[h] && self.hikers[h] == self.pos {
                self.rescued[h] = true;
                reward += 10.0;
                event = Event::Rescued;
            }
        }
        if self.pos == self.base {
            if self.battery < BATTERY_MAX && event == Event::None {
                event = Event::Recharged;
            }
            self.battery = BATTERY_MAX;
            if self.hikers_left() == 0 {
                self.completed = true;
                return StepOutcome::new(reward + 10.0, true, Event::Completed, vetoed);
            }
        }
        self.sensor_ok = self.rng.gen::<f32>() >= DROPOUT;
        if self.steps >= MAX_STEPS {
            self.timed_out = true;
            return StepOutcome::new(reward, true, Event::Timeout, vetoed);
        }
        StepOutcome::new(reward, false, event, vetoed)
    }

    /// Hikers rescued so far.
    pub fn score(&self) -> u32 {
        (N_HIKERS - self.hikers_left()) as u32
    }

    pub fn snapshot(&self) -> RescueSnapshot {
        RescueSnapshot {
            n: N,
            cells: self.cells.to_vec(),
            drone: rc(self.pos),
            prev: rc(self.prev),
            base: rc(self.base),
            hikers: (0..N_HIKERS)
                .map(|h| {
                    let (r, c) = rc(self.hikers[h]);
                    (r, c, self.rescued[h])
                })
                .collect(),
            battery: self.battery,
            battery_max: BATTERY_MAX,
            sensor_ok: self.sensor_ok,
            rescued: self.score(),
            steps: self.steps,
            last_action: self.last_action,
            vetoed: self.last_vetoed,
            crashed: self.crashed,
            completed: self.completed,
        }
    }

    pub fn questions() -> &'static [QuestionSpec] {
        QUESTIONS
    }

    /// For each move `n s e w`: `<m>h` = cost of rescuing the cheapest hiker through that move
    /// and flying home (`?` on sensor dropout, `-` when no hiker is left), `<m>b` = cost of
    /// flying home through that move (`x` = blocked); then battery, wind at the drone, hikers
    /// left and sensor status.
    pub fn state_text(&self) -> String {
        let f = self.features();
        let hk = self.hikers_left();
        let mut s = String::with_capacity(96);
        for d in DIRS {
            let m = f[d.index()];
            let l = d.letter();
            if m.blocked {
                s.push_str(&format!("{l}h x {l}b x "));
                continue;
            }
            let h = if hk == 0 {
                "-".to_string()
            } else if !self.sensor_ok {
                "?".to_string()
            } else {
                m.trip.min(99).to_string()
            };
            s.push_str(&format!("{l}h {h} {l}b {} ", m.home.min(99)));
        }
        let wind = wind_of(self.cells[self.pos]).map_or("0", Dir::letter);
        let sen = if self.sensor_ok { "ok" } else { "?" };
        s.push_str(&format!(
            "bat {} wind {wind} hk {hk} sen {sen}",
            self.battery
        ));
        s
    }

    /// Soft targets: `move` over the 7 canonical options, `risk` over 5 levels.
    pub fn oracle(&self) -> Vec<Vec<f32>> {
        vec![self.move_target(), self.risk_target()]
    }

    fn move_target(&self) -> Vec<f32> {
        let mut p = vec![0.0f32; 7];
        let f = self.features();
        let ret = RescueAction::ReturnHome.index();
        if self.hikers_left() == 0 {
            p[ret] = 1.0;
            return with_floor(p, 0.01);
        }
        let home_here = self.home_cost();
        if !self.sensor_ok {
            if home_here + MARGIN > self.battery {
                p[ret] = 1.0;
            } else {
                p[RescueAction::Unknown.index()] = 1.0;
            }
            return with_floor(p, 0.01);
        }
        let best = f
            .iter()
            .filter(|m| !m.blocked)
            .map(|m| m.trip)
            .min()
            .unwrap_or(INF as i32);
        let mut moves = [0.0f32; 4];
        for (i, m) in f.iter().enumerate() {
            if !m.blocked && m.trip < INF as i32 {
                moves[i] = (-((m.trip - best) as f32) / 0.6).exp();
            }
        }
        let z: f32 = moves.iter().sum();
        let go_margin = (self.battery - best - MARGIN) as f32;
        let p_go = if z > 0.0 {
            crate::common::sigmoid((go_margin + 0.5) / 0.5)
        } else {
            0.0
        };
        for i in 0..4 {
            if z > 0.0 {
                p[i] = p_go * moves[i] / z;
            }
        }
        if self.pos == self.base {
            // Already home: hovering (recharging) is the conservative alternative.
            p[RescueAction::Hover.index()] = 1.0 - p_go;
        } else {
            p[ret] = 1.0 - p_go;
        }
        with_floor(p, 0.01)
    }

    fn risk_target(&self) -> Vec<f32> {
        let margin = (self.battery - self.home_cost()) as f32;
        // Piecewise-linear map from battery margin to a continuous level in [0, 4].
        const KNOTS: [(f32, f32); 5] =
            [(24.0, 0.0), (14.0, 1.0), (8.0, 2.0), (4.0, 3.0), (0.0, 4.0)];
        let level = if margin >= KNOTS[0].0 {
            0.0
        } else if margin <= 0.0 {
            4.0
        } else {
            let k = KNOTS
                .windows(2)
                .find(|w| margin <= w[0].0 && margin >= w[1].0)
                .expect("in range");
            let t = (k[0].0 - margin) / (k[0].0 - k[1].0);
            k[0].1 + t * (k[1].1 - k[0].1)
        };
        let p: Vec<f32> = (0..5)
            .map(|i| (-(i as f32 - level).powi(2) / (2.0 * 0.45 * 0.45)).exp())
            .collect();
        with_floor(p, 0.005)
    }

    pub fn oracle_action(&self) -> RescueAction {
        RescueAction::from_index(crate::common::argmax(&self.move_target()))
    }
}

impl DecisionGame for Rescue {
    fn name(&self) -> &'static str {
        NAME
    }
    fn reset(&mut self) {
        Rescue::reset(self)
    }
    fn done(&self) -> bool {
        Rescue::done(self)
    }
    fn decision_due(&self) -> bool {
        true
    }
    fn questions(&self) -> &'static [QuestionSpec] {
        QUESTIONS
    }
    fn state_text(&self) -> String {
        Rescue::state_text(self)
    }
    fn oracle(&self) -> Vec<Vec<f32>> {
        Rescue::oracle(self)
    }
    fn apply(&mut self, choice: usize, _abort: bool) -> StepOutcome {
        self.step(RescueAction::from_index(choice))
    }
    fn tick(&mut self) -> StepOutcome {
        self.step(RescueAction::Hover)
    }
    fn stats(&self) -> EpisodeStats {
        EpisodeStats {
            score: self.score() as f32,
            success: self.completed,
            crashed: self.crashed,
            vetoes: self.vetoes,
            steps: self.steps,
            ..EpisodeStats::default()
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
        id: "move",
        qtype: QType::Choice,
        ins_en: &[
            "Which move should the rescue drone make next?",
            "Pick the next step of the rescue drone: reach hikers but never run out of battery.",
            "Where should the drone go now to rescue the hikers and still get home?",
            "Choose the drone's next action on the rescue grid.",
        ],
        ins_fr: &[
            "Quel déplacement le drone de secours doit-il faire ?",
            "Choisis la prochaine action du drone : sauver les randonneurs sans vider la batterie.",
            "Où le drone doit-il aller pour secourir les randonneurs et rentrer à la base ?",
        ],
        options: &[
            OptionSpec {
                key: "north",
                synonyms: &["n", "up", "nord"],
                desc_en: &["move one cell north", "fly north", "go one square up"],
                desc_fr: &["avancer d'une case vers le nord", "voler vers le nord"],
            },
            OptionSpec {
                key: "south",
                synonyms: &["s", "down", "sud"],
                desc_en: &["move one cell south", "fly south", "go one square down"],
                desc_fr: &["avancer d'une case vers le sud", "voler vers le sud"],
            },
            OptionSpec {
                key: "east",
                synonyms: &["e", "right", "est"],
                desc_en: &["move one cell east", "fly east", "go one square right"],
                desc_fr: &["avancer d'une case vers l'est", "voler vers l'est"],
            },
            OptionSpec {
                key: "west",
                synonyms: &["w", "left", "ouest"],
                desc_en: &["move one cell west", "fly west", "go one square left"],
                desc_fr: &["avancer d'une case vers l'ouest", "voler vers l'ouest"],
            },
            OptionSpec {
                key: "hover",
                synonyms: &["hold", "wait", "attendre"],
                desc_en: &["stay in place for one step", "hold position", "wait here"],
                desc_fr: &["rester sur place", "tenir la position"],
            },
            OptionSpec {
                key: "return_home",
                synonyms: &["rtb", "go_home", "retour"],
                desc_en: &[
                    "fly back toward the base to recharge",
                    "head home now",
                    "return to base",
                ],
                desc_fr: &["rentrer à la base pour recharger", "retourner à la base"],
            },
            OptionSpec {
                key: "unknown",
                synonyms: &["unsure", "cannot_tell", "inconnu"],
                desc_en: &[
                    "the sensor data is missing, the right move cannot be decided",
                    "not enough information",
                    "cannot tell from the readings",
                ],
                desc_fr: &["les données du capteur manquent", "impossible de décider"],
            },
        ],
    },
    QuestionSpec {
        id: "risk",
        qtype: QType::Score,
        ins_en: &[
            "How risky is the drone's battery margin for getting home?",
            "Rate the battery risk: can the drone still make it back to base?",
            "Assess the danger of running out of battery before reaching base.",
        ],
        ins_fr: &[
            "Quel est le risque lié à la marge de batterie pour rentrer ?",
            "Évalue le risque de tomber en panne de batterie avant la base.",
        ],
        options: &[
            OptionSpec {
                key: "0",
                synonyms: &[],
                desc_en: &["negligible", "no concern", "plenty of battery"],
                desc_fr: &["négligeable", "batterie largement suffisante"],
            },
            OptionSpec {
                key: "1",
                synonyms: &[],
                desc_en: &["low", "minor concern", "comfortable margin"],
                desc_fr: &["faible", "marge confortable"],
            },
            OptionSpec {
                key: "2",
                synonyms: &[],
                desc_en: &["moderate", "watch the battery", "margin shrinking"],
                desc_fr: &["modéré", "marge qui diminue"],
            },
            OptionSpec {
                key: "3",
                synonyms: &[],
                desc_en: &["high", "serious concern", "barely enough battery"],
                desc_fr: &["élevé", "batterie juste suffisante"],
            },
            OptionSpec {
                key: "4",
                synonyms: &[],
                desc_en: &["critical", "emergency", "battery about to run out"],
                desc_fr: &["critique", "urgence"],
            },
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dijkstra_respects_wind() {
        let mut g = Rescue::new(5);
        g.cells = [EMPTY; CELLS];
        // Wind blowing west at (0, 1): entering it eastward costs 3.
        g.cells[1] = WIND_N + Dir::W.index() as u8;
        let m = g.cost_to(2);
        assert_eq!(m[0], 4);
        assert_eq!(m[1], 1);
    }

    #[test]
    fn veto_blocks_rocks_and_low_battery() {
        let mut g = Rescue::new(7);
        g.cells = [EMPTY; CELLS];
        g.base = 0;
        g.base_map = g.cost_to(0);
        g.pos = 1;
        g.cells[2] = ROCK;
        g.battery = 10;
        assert_eq!(g.veto(RescueAction::East), Some(RescueAction::Hover));
        g.battery = 1;
        assert_eq!(g.veto(RescueAction::South), Some(RescueAction::ReturnHome));
        assert_eq!(g.veto(RescueAction::West), None);
    }
}
