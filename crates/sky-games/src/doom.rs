//! "Doom-like": a deterministic, grid-based first-person shooter in the spirit of Doom (1993).
//!
//! The player walks a level of rooms, corridors and doors at Doom's fixed 35 Hz tick, fights
//! imp-like monsters (melee claw + dodgeable fireballs) and zombieman-like monsters (hitscan),
//! picks up health and ammo, and presses the exit switch. The agent decides every
//! [`DECISION_EVERY`] ticks (~9 Hz) and the action is held in between, like a human's reaction
//! time in Doom.
//!
//! Levels are either hand-designed (mirrored at random) or generated from the episode seed.
//! The code measures (A* waypoint, enemy bearings, wall distances, incoming projectiles) and
//! writes small integers in the state text; the model decides. A coded safety reflex
//! ([`Doom::veto`]) never lets the player stand in a known fireball path when a dodge is free,
//! and never fires an empty weapon.
//!
//! Units: one grid cell is one unit (64 Doom map units). `x` points east, `y` south (row
//! index); the facing vector is `(cos angle, sin angle)`, so turning right increases `angle`
//! and positive bearings are on the player's right.

use crate::common::{
    argmax, bin, mix_seed, sigmoid, with_floor, DecisionGame, EpisodeStats, Event, OptionSpec,
    QuestionSpec, StepOutcome,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use sky_schema::QType;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f32::consts::{PI, TAU};
use std::sync::Arc;

pub const NAME: &str = "doom";
/// Doom's tick rate.
pub const TICRATE: u32 = 35;
/// Ticks between two decisions (the action is held in between).
pub const DECISION_EVERY: u32 = 4;
/// Episode budget: 2.5 minutes.
pub const MAX_TICKS: u32 = TICRATE * 150;
pub const PLAYER_RADIUS: f32 = 0.25;
/// Movement per tick, in cells.
pub const MOVE_SPEED: f32 = 0.11;
pub const STRAFE_SPEED: f32 = 0.10;
pub const BACK_SPEED: f32 = 0.08;
/// Turn per tick (4 degrees: 16 degrees per decision).
pub const TURN_RATE: f32 = 4.0 * PI / 180.0;
/// Ticks between two shots.
pub const FIRE_COOLDOWN: u32 = 18;
/// Half-angle of the aim cone (autoaim). At least half a decision's turn, so that turning
/// toward an enemy always lands inside the cone.
pub const AIM_CONE: f32 = 9.0 * PI / 180.0;
pub const SHOT_RANGE: f32 = 24.0;
/// Enemies farther than this are not reported in the state text.
pub const SIGHT_RANGE: f32 = 14.0;
/// Reach of `use`, from the player's center.
pub const USE_RANGE: f32 = 1.0;
pub const MAX_HEALTH: i32 = 100;
pub const MAX_AMMO: i32 = 50;
pub const MONSTER_RADIUS: f32 = 0.3;
pub const PROJ_SPEED: f32 = 0.17;
pub const PROJ_RADIUS: f32 = 0.12;
const DOOR_SPEED: f32 = 1.0 / 24.0;
/// A door is passable from this opening on, and stops blocking sight from `DOOR_SIGHT`.
const DOOR_PASS: f32 = 0.95;
const DOOR_SIGHT: f32 = 0.6;
const PICKUP_RADIUS: f32 = 0.55;
/// Firing wakes monsters within this many cells of open path.
const NOISE_RANGE: u16 = 12;
const WAKE_RANGE: f32 = 16.0;
const PAIN_TICKS: u32 = 8;
const ATTACK_TICKS: u32 = 12;
/// Lookahead of the safety reflex and of the oracle's dodge check (ticks).
const VETO_HORIZON: u32 = 12;
const DODGE_HORIZON: u32 = 16;
/// The danger question looks one second ahead.
const DANGER_TICKS: u32 = TICRATE;
/// Health at or below which the navigation goal becomes a health pickup.
const LOW_HEALTH: i32 = 40;
/// Ammo at or below which the navigation goal becomes an ammo pickup.
const LOW_AMMO: i32 = 2;
/// The oracle turns toward the waypoint beyond this bearing (degrees).
const NAV_TURN_DEG: f32 = 22.0;
/// The oracle opens a door ahead when the waypoint is within this bearing (degrees).
const DOOR_USE_DEG: f32 = 35.0;
const UNREACHED: u16 = u16::MAX;

/// The eight actions of the `action` question, in canonical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DoomAction {
    Forward,
    TurnLeft,
    TurnRight,
    StrafeLeft,
    StrafeRight,
    Shoot,
    Use,
    Back,
}

impl DoomAction {
    pub const ALL: [DoomAction; 8] = [
        DoomAction::Forward,
        DoomAction::TurnLeft,
        DoomAction::TurnRight,
        DoomAction::StrafeLeft,
        DoomAction::StrafeRight,
        DoomAction::Shoot,
        DoomAction::Use,
        DoomAction::Back,
    ];

    pub fn from_index(i: usize) -> DoomAction {
        Self::ALL[i.min(7)]
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn key(self) -> &'static str {
        QUESTIONS[0].options[self.index()].key
    }

    pub fn from_key(key: &str) -> Option<DoomAction> {
        Self::ALL.into_iter().find(|a| a.key() == key)
    }
}

/// Dodge candidates of the reflex and the oracle, in preference order.
const DODGES: [DoomAction; 4] = [
    DoomAction::StrafeLeft,
    DoomAction::StrafeRight,
    DoomAction::Back,
    DoomAction::Forward,
];

/// Difficulty. `Easy` is the default game (`doom`); `Normal` hits harder, attacks more often
/// and puts more monsters in generated levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Skill {
    Easy,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MonsterKind {
    /// Imp-like: melee claw up close, fireballs at range.
    Imp,
    /// Zombieman-like: hitscan rifle, inaccurate at range.
    Zombie,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MonsterState {
    Idle,
    Chase,
    Attack,
    Pain,
    Dead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ItemKind {
    /// Stimpack: +10 health.
    Health,
    /// Medikit: +25 health.
    Medikit,
    /// Box of shells: +10 ammo.
    Ammo,
}

/// A map cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Floor,
    /// A wall with a texture id in `1..=5`.
    Wall(u8),
    Door,
    /// The exit switch: a solid wall cell that ends the level when used.
    Exit,
}

impl Cell {
    fn code(self) -> char {
        match self {
            Cell::Floor => '.',
            Cell::Wall(t) => (b'0' + t.clamp(1, 5)) as char,
            Cell::Door => 'D',
            Cell::Exit => 'X',
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Player {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub health: i32,
    pub ammo: i32,
    cooldown: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Monster {
    pub id: u32,
    pub kind: MonsterKind,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub hp: i32,
    pub state: MonsterState,
    state_ticks: u32,
    attack_cd: u32,
}

impl Monster {
    pub fn alive(&self) -> bool {
        self.state != MonsterState::Dead
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projectile {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    damage: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Item {
    pub id: u32,
    pub kind: ItemKind,
    pub x: f32,
    pub y: f32,
    pub taken: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DoorState {
    cell: usize,
    open: f32,
    opening: bool,
}

/// The static part of a level (shared between the game and its lookahead copies).
#[derive(Debug)]
struct MapData {
    name: String,
    w: usize,
    h: usize,
    cells: Vec<Cell>,
    /// Index into `Doom::doors` of each door cell (`u16::MAX` elsewhere).
    door_of: Vec<u16>,
    exit: usize,
}

/// A level as built by a map source, before play starts.
struct Level {
    map: MapData,
    start: (f32, f32, f32),
    monsters: Vec<(MonsterKind, f32, f32, f32)>,
    items: Vec<(ItemKind, f32, f32)>,
}

/// What lies within reach straight ahead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ahead {
    Nothing,
    /// A closed door that is not opening yet.
    Door,
    Exit,
}

/// The navigation goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Goal {
    Exit,
    Item(usize),
}

/// The next waypoint toward the goal (A* path, string-pulled).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nav {
    pub goal: Goal,
    pub x: f32,
    pub y: f32,
    /// Bearing from the player's facing, radians (positive = right).
    pub bearing: f32,
    pub dist: f32,
    /// Length of the A* path to the goal, in cells (`None` when unreachable).
    pub path_len: Option<f32>,
}

/// Everything the state text reports, measured from the full simulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Features {
    /// Nearest visible enemy: index, bearing (radians, positive = right), distance.
    pub enemy: Option<(usize, f32, f32)>,
    pub aligned: bool,
    pub n_visible: u32,
    pub wall_ahead: f32,
    pub wall_left: f32,
    pub wall_right: f32,
    pub ahead: Ahead,
    pub nav: Nav,
    /// Nearest incoming projectile: bearing, distance, ticks to impact.
    pub proj: Option<(f32, f32, f32)>,
}

/// Render data for the browser raycaster.
#[derive(Debug, Clone, Serialize)]
pub struct DoomSnapshot {
    pub map: String,
    pub skill: Skill,
    pub width: usize,
    pub height: usize,
    /// `width * height` cell codes, row-major: `.` floor, `1`..`5` walls, `D` door, `X` exit.
    pub cells: String,
    pub doors: Vec<DoorView>,
    pub exit: ExitView,
    pub player: PlayerView,
    pub sprites: Vec<SpriteView>,
    pub waypoint: Option<PointView>,
    pub tick: u32,
    pub kills: u32,
    pub monsters: u32,
    pub items: u32,
    pub items_total: u32,
    pub score: u32,
    /// Ticks since the last shot (large when none).
    pub fire_tick: u32,
    /// Ticks since the player last took damage (large when none).
    pub hurt_tick: u32,
    pub action: Option<DoomAction>,
    pub vetoed: bool,
    pub vetoed_request: Option<DoomAction>,
    pub dead: bool,
    pub completed: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct DoorView {
    pub x: usize,
    pub y: usize,
    pub open: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ExitView {
    pub x: usize,
    pub y: usize,
    pub pressed: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PlayerView {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub health: i32,
    pub ammo: i32,
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PointView {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SpriteView {
    pub id: u32,
    pub kind: &'static str,
    pub x: f32,
    pub y: f32,
    pub state: &'static str,
    pub angle: f32,
    pub anim: u32,
    pub visible: bool,
}

/// Monster and damage parameters of a skill.
#[derive(Debug, Clone, Copy)]
struct SkillParams {
    start_ammo: i32,
    fireball_dmg: (i32, i32),
    claw_dmg: (i32, i32),
    rifle_dmg: (i32, i32),
    imp_cd: (u32, u32),
    zombie_cd: (u32, u32),
    /// Generated levels: monster count range.
    monsters: (usize, usize),
}

impl Skill {
    fn params(self) -> SkillParams {
        match self {
            Skill::Easy => SkillParams {
                start_ammo: 25,
                fireball_dmg: (5, 12),
                claw_dmg: (4, 10),
                rifle_dmg: (3, 8),
                imp_cd: (50, 80),
                zombie_cd: (55, 85),
                monsters: (3, 5),
            },
            Skill::Normal => SkillParams {
                start_ammo: 20,
                fireball_dmg: (8, 20),
                claw_dmg: (6, 16),
                rifle_dmg: (5, 12),
                imp_cd: (35, 60),
                zombie_cd: (40, 65),
                monsters: (5, 8),
            },
        }
    }
}

/// Wraps an angle to `(-PI, PI]`.
fn wrap(a: f32) -> f32 {
    let mut a = a % TAU;
    if a > PI {
        a -= TAU;
    } else if a <= -PI {
        a += TAU;
    }
    a
}

fn deg(a: f32) -> f32 {
    a * 180.0 / PI
}

/// Half-angle within which a shot hits a monster at distance `d`.
fn cone(d: f32) -> f32 {
    AIM_CONE.max((MONSTER_RADIUS / d.max(0.1)).atan())
}

#[derive(Debug, Clone)]
pub struct Doom {
    seed: u64,
    episode: u64,
    rng: SmallRng,
    skill: Skill,
    map: Arc<MapData>,
    doors: Vec<DoorState>,
    door_version: u32,
    pub player: Player,
    monsters: Vec<Monster>,
    projectiles: Vec<Projectile>,
    items: Vec<Item>,
    action: Option<DoomAction>,
    use_pending: bool,
    pub tick: u32,
    pub kills: u32,
    pub items_taken: u32,
    pub dead: bool,
    pub completed: bool,
    pub timed_out: bool,
    pub vetoes: u32,
    /// When false, the safety reflex is disabled.
    pub veto_enabled: bool,
    last_vetoed: Option<DoomAction>,
    last_fire: Option<u32>,
    last_hurt: Option<u32>,
    /// Hits taken from projectiles (lookahead bookkeeping).
    proj_hits: u32,
    /// Lookahead copies: monsters neither move nor attack.
    frozen: bool,
    next_id: u32,
    /// Monster flow field toward the player, keyed by (player cell, door version).
    flow: Arc<Vec<u16>>,
    flow_key: (usize, u32),
    nav_cache: RefCell<Option<(u64, u32, Nav)>>,
}

impl Doom {
    /// The default game: easy skill.
    pub fn new(seed: u64) -> Self {
        Self::with_skill(seed, Skill::Easy)
    }

    pub fn with_skill(seed: u64, skill: Skill) -> Self {
        let empty = MapData {
            name: String::new(),
            w: 1,
            h: 1,
            cells: vec![Cell::Floor],
            door_of: vec![u16::MAX],
            exit: 0,
        };
        let mut g = Doom {
            seed,
            episode: 0,
            rng: SmallRng::seed_from_u64(mix_seed(seed, 0)),
            skill,
            map: Arc::new(empty),
            doors: Vec::new(),
            door_version: 0,
            player: Player {
                x: 0.5,
                y: 0.5,
                angle: 0.0,
                health: MAX_HEALTH,
                ammo: 0,
                cooldown: 0,
            },
            monsters: Vec::new(),
            projectiles: Vec::new(),
            items: Vec::new(),
            action: None,
            use_pending: false,
            tick: 0,
            kills: 0,
            items_taken: 0,
            dead: false,
            completed: false,
            timed_out: false,
            vetoes: 0,
            veto_enabled: true,
            last_vetoed: None,
            last_fire: None,
            last_hurt: None,
            proj_hits: 0,
            frozen: false,
            next_id: 0,
            flow: Arc::new(Vec::new()),
            flow_key: (usize::MAX, u32::MAX),
            nav_cache: RefCell::new(None),
        };
        g.start();
        g
    }

    /// Starts the next episode (a new level).
    pub fn reset(&mut self) {
        self.episode += 1;
        self.rng = SmallRng::seed_from_u64(mix_seed(self.seed, self.episode));
        self.start();
    }

    pub fn skill(&self) -> Skill {
        self.skill
    }

    pub fn map_name(&self) -> &str {
        &self.map.name
    }

    fn start(&mut self) {
        let level = if self.rng.gen_bool(0.3) {
            let i = self.rng.gen_range(0..HAND_MAPS.len());
            let (fx, fy) = (self.rng.gen_bool(0.5), self.rng.gen_bool(0.5));
            parse_map(HAND_MAPS[i].0, HAND_MAPS[i].1, fx, fy)
        } else {
            generate_level(&mut self.rng, self.skill)
        };
        let Level {
            map,
            start,
            monsters,
            items,
        } = level;
        self.doors = map
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == Cell::Door)
            .map(|(i, _)| DoorState {
                cell: i,
                open: 0.0,
                opening: false,
            })
            .collect();
        self.map = Arc::new(map);
        self.door_version = 0;
        let p = self.skill.params();
        self.player = Player {
            x: start.0,
            y: start.1,
            angle: start.2,
            health: MAX_HEALTH,
            ammo: p.start_ammo,
            cooldown: 0,
        };
        self.next_id = 0;
        self.monsters = monsters
            .into_iter()
            .map(|(kind, x, y, angle)| {
                self.next_id += 1;
                Monster {
                    id: self.next_id,
                    kind,
                    x,
                    y,
                    // Hand-designed levels give a facing; jitter it so every episode differs.
                    angle: wrap(angle + self.rng.gen_range(-0.4..0.4)),
                    hp: match kind {
                        MonsterKind::Imp => 60,
                        MonsterKind::Zombie => 20,
                    },
                    state: MonsterState::Idle,
                    state_ticks: 0,
                    attack_cd: 0,
                }
            })
            .collect();
        self.items = items
            .into_iter()
            .map(|(kind, x, y)| {
                self.next_id += 1;
                Item {
                    id: self.next_id,
                    kind,
                    x,
                    y,
                    taken: false,
                }
            })
            .collect();
        self.projectiles.clear();
        self.action = None;
        self.use_pending = false;
        self.tick = 0;
        self.kills = 0;
        self.items_taken = 0;
        self.dead = false;
        self.completed = false;
        self.timed_out = false;
        self.vetoes = 0;
        self.last_vetoed = None;
        self.last_fire = None;
        self.last_hurt = None;
        self.proj_hits = 0;
        self.frozen = false;
        self.flow = Arc::new(Vec::new());
        self.flow_key = (usize::MAX, u32::MAX);
        *self.nav_cache.borrow_mut() = None;
    }

    pub fn done(&self) -> bool {
        self.dead || self.completed || self.timed_out
    }

    pub fn decision_due(&self) -> bool {
        !self.done() && self.tick.is_multiple_of(DECISION_EVERY)
    }

    pub fn monsters(&self) -> &[Monster] {
        &self.monsters
    }

    pub fn projectiles(&self) -> &[Projectile] {
        &self.projectiles
    }

    pub fn items(&self) -> &[Item] {
        &self.items
    }

    pub fn n_monsters(&self) -> u32 {
        self.monsters.len() as u32
    }

    // ---------------------------------------------------------------- grid queries

    fn idx(&self, x: i32, y: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && (x as usize) < self.map.w && (y as usize) < self.map.h)
            .then(|| y as usize * self.map.w + x as usize)
    }

    pub fn cell_at(&self, x: i32, y: i32) -> Option<Cell> {
        self.idx(x, y).map(|i| self.map.cells[i])
    }

    fn door_open(&self, i: usize) -> f32 {
        match self.map.door_of.get(i) {
            Some(&d) if d != u16::MAX => self.doors[d as usize].open,
            _ => 0.0,
        }
    }

    fn center(&self, i: usize) -> (f32, f32) {
        ((i % self.map.w) as f32 + 0.5, (i / self.map.w) as f32 + 0.5)
    }

    fn cell_of(&self, x: f32, y: f32) -> usize {
        let cx = (x.floor() as i32).clamp(0, self.map.w as i32 - 1);
        let cy = (y.floor() as i32).clamp(0, self.map.h as i32 - 1);
        cy as usize * self.map.w + cx as usize
    }

    /// Blocks movement (walls, the exit switch, doors that are not open yet, outside).
    fn solid_move(&self, x: i32, y: i32) -> bool {
        match self.idx(x, y) {
            None => true,
            Some(i) => match self.map.cells[i] {
                Cell::Floor => false,
                Cell::Door => self.door_open(i) < DOOR_PASS,
                _ => true,
            },
        }
    }

    /// Blocks sight and projectiles.
    fn solid_sight(&self, x: i32, y: i32) -> bool {
        match self.idx(x, y) {
            None => true,
            Some(i) => match self.map.cells[i] {
                Cell::Floor => false,
                Cell::Door => self.door_open(i) < DOOR_SIGHT,
                _ => true,
            },
        }
    }

    /// True when a circle overlaps a solid cell (`through` is treated as open).
    fn circle_hits_walls(&self, x: f32, y: f32, r: f32, through: Option<usize>) -> bool {
        let (x0, x1) = ((x - r).floor() as i32, (x + r).floor() as i32);
        let (y0, y1) = ((y - r).floor() as i32, (y + r).floor() as i32);
        for cy in y0..=y1 {
            for cx in x0..=x1 {
                if !self.solid_move(cx, cy) || (through.is_some() && self.idx(cx, cy) == through) {
                    continue;
                }
                let nx = x.clamp(cx as f32, cx as f32 + 1.0);
                let ny = y.clamp(cy as f32, cy as f32 + 1.0);
                if (x - nx).powi(2) + (y - ny).powi(2) < r * r {
                    return true;
                }
            }
        }
        false
    }

    /// Line of sight between two points (grid traversal; the end cell itself is not tested).
    fn los(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> bool {
        let (mut cx, mut cy) = (x0.floor() as i32, y0.floor() as i32);
        let (tx, ty) = (x1.floor() as i32, y1.floor() as i32);
        let (dx, dy) = (x1 - x0, y1 - y0);
        let step_x = if dx > 0.0 { 1 } else { -1 };
        let step_y = if dy > 0.0 { 1 } else { -1 };
        let t_dx = if dx != 0.0 {
            1.0 / dx.abs()
        } else {
            f32::INFINITY
        };
        let t_dy = if dy != 0.0 {
            1.0 / dy.abs()
        } else {
            f32::INFINITY
        };
        let mut t_x = if dx > 0.0 {
            (cx as f32 + 1.0 - x0) * t_dx
        } else {
            (x0 - cx as f32) * t_dx
        };
        let mut t_y = if dy > 0.0 {
            (cy as f32 + 1.0 - y0) * t_dy
        } else {
            (y0 - cy as f32) * t_dy
        };
        for _ in 0..256 {
            if cx == tx && cy == ty {
                return true;
            }
            if t_x < t_y {
                cx += step_x;
                t_x += t_dx;
            } else {
                cy += step_y;
                t_y += t_dy;
            }
            if (cx != tx || cy != ty) && self.solid_sight(cx, cy) {
                return false;
            }
        }
        true
    }

    /// Distance along a ray to the first cell that blocks movement (capped at `max`), and that
    /// cell.
    fn ray(&self, x0: f32, y0: f32, angle: f32, max: f32) -> (f32, Option<usize>) {
        let (dx, dy) = (angle.cos(), angle.sin());
        let (mut cx, mut cy) = (x0.floor() as i32, y0.floor() as i32);
        let step_x = if dx > 0.0 { 1 } else { -1 };
        let step_y = if dy > 0.0 { 1 } else { -1 };
        let t_dx = if dx.abs() > 1e-9 {
            1.0 / dx.abs()
        } else {
            f32::INFINITY
        };
        let t_dy = if dy.abs() > 1e-9 {
            1.0 / dy.abs()
        } else {
            f32::INFINITY
        };
        let mut t_x = if dx > 0.0 {
            (cx as f32 + 1.0 - x0) * t_dx
        } else {
            (x0 - cx as f32) * t_dx
        };
        let mut t_y = if dy > 0.0 {
            (cy as f32 + 1.0 - y0) * t_dy
        } else {
            (y0 - cy as f32) * t_dy
        };
        loop {
            let t = if t_x < t_y {
                cx += step_x;
                let t = t_x;
                t_x += t_dx;
                t
            } else {
                cy += step_y;
                let t = t_y;
                t_y += t_dy;
                t
            };
            if t > max {
                return (max, None);
            }
            if self.solid_move(cx, cy) {
                return (t, self.idx(cx, cy));
            }
        }
    }

    // ---------------------------------------------------------------- navigation

    /// Cells the planner may cross: floor and doors (open or not; doors cost extra).
    fn nav_passable(&self, i: usize) -> bool {
        matches!(self.map.cells[i], Cell::Floor | Cell::Door)
    }

    /// A* over the grid (8-connected, no corner cutting, closed doors cost two extra cells).
    pub fn astar(&self, start: usize, goals: &[usize]) -> Option<Vec<usize>> {
        let (w, h) = (self.map.w, self.map.h);
        let n = w * h;
        if goals.is_empty() || start >= n {
            return None;
        }
        let heur = |i: usize| -> u32 {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            goals
                .iter()
                .map(|&g| {
                    let (dx, dy) = (((g % w) as i32 - x).abs(), ((g / w) as i32 - y).abs());
                    (10 * dx.max(dy) + 4 * dx.min(dy)) as u32
                })
                .min()
                .unwrap_or(0)
        };
        let mut cost = vec![u32::MAX; n];
        let mut parent = vec![usize::MAX; n];
        let mut closed = vec![false; n];
        let mut heap = BinaryHeap::new();
        cost[start] = 0;
        heap.push(Reverse((heur(start), start)));
        while let Some(Reverse((_, c))) = heap.pop() {
            if closed[c] {
                continue;
            }
            closed[c] = true;
            if goals.contains(&c) {
                let mut path = vec![c];
                let mut k = c;
                while parent[k] != usize::MAX {
                    k = parent[k];
                    path.push(k);
                }
                path.reverse();
                return Some(path);
            }
            let (x, y) = ((c % w) as i32, (c / w) as i32);
            for (dx, dy) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let Some(ni) = self.idx(x + dx, y + dy) else {
                    continue;
                };
                if !self.nav_passable(ni) {
                    continue;
                }
                let diag = dx != 0 && dy != 0;
                if diag {
                    let floor =
                        |i: Option<usize>| i.is_some_and(|i| self.map.cells[i] == Cell::Floor);
                    if !(floor(Some(c))
                        && floor(Some(ni))
                        && floor(self.idx(x + dx, y))
                        && floor(self.idx(x, y + dy)))
                    {
                        continue;
                    }
                }
                let mut step = if diag { 14 } else { 10 };
                if self.map.cells[ni] == Cell::Door && self.door_open(ni) < DOOR_PASS {
                    step += 20;
                }
                let nc = cost[c] + step;
                if nc < cost[ni] {
                    cost[ni] = nc;
                    parent[ni] = c;
                    heap.push(Reverse((nc + heur(ni), ni)));
                }
            }
        }
        None
    }

    fn exit_goal_cells(&self) -> Vec<usize> {
        let e = self.map.exit;
        let (x, y) = ((e % self.map.w) as i32, (e / self.map.w) as i32);
        [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .into_iter()
            .filter_map(|(dx, dy)| self.idx(x + dx, y + dy))
            .filter(|&i| self.map.cells[i] == Cell::Floor)
            .collect()
    }

    /// The navigation goal: a health pickup when hurt, ammo when (nearly) empty, else the exit.
    fn choose_goal(&self) -> Goal {
        let p = &self.player;
        let want = |k: ItemKind| match k {
            ItemKind::Health | ItemKind::Medikit => p.health <= LOW_HEALTH,
            ItemKind::Ammo => p.ammo <= LOW_AMMO && p.health > LOW_HEALTH,
        };
        self.items
            .iter()
            .enumerate()
            .filter(|(_, it)| !it.taken && want(it.kind))
            .min_by(|a, b| {
                let da = (a.1.x - p.x).hypot(a.1.y - p.y);
                let db = (b.1.x - p.x).hypot(b.1.y - p.y);
                da.total_cmp(&db)
            })
            .map_or(Goal::Exit, |(i, _)| Goal::Item(i))
    }

    /// True when the player's circle can slide straight from `a` to `b`.
    fn clear_walk(&self, a: (f32, f32), b: (f32, f32), through: Option<usize>) -> bool {
        let d = (b.0 - a.0).hypot(b.1 - a.1);
        let n = (d / 0.1).ceil().max(1.0) as usize;
        (1..=n).all(|k| {
            let t = k as f32 / n as f32;
            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            !self.circle_hits_walls(x, y, PLAYER_RADIUS * 0.85, through)
        })
    }

    fn compute_nav(&self) -> Nav {
        let p = self.player;
        let mut goal = self.choose_goal();
        let start = self.cell_of(p.x, p.y);
        let mut path = None;
        if let Goal::Item(i) = goal {
            let it = self.items[i];
            path = self.astar(start, &[self.cell_of(it.x, it.y)]);
            if path.is_none() {
                goal = Goal::Exit;
            }
        }
        if goal == Goal::Exit {
            path = self.astar(start, &self.exit_goal_cells());
        }
        let goal_point = match goal {
            Goal::Exit => self.center(self.map.exit),
            Goal::Item(i) => (self.items[i].x, self.items[i].y),
        };
        let (wx, wy) = match &path {
            None => goal_point,
            Some(path) if path.len() == 1 => goal_point,
            Some(path) => {
                let closed_door =
                    |i: usize| self.map.cells[i] == Cell::Door && self.door_open(i) < DOOR_PASS;
                let door_k = path
                    .iter()
                    .position(|&c| closed_door(c))
                    .unwrap_or(path.len());
                let max_k = (path.len() - 1).min(door_k).min(8);
                let mut best = 1;
                for k in (1..=max_k).rev() {
                    let c = path[k];
                    let through = closed_door(c).then_some(c);
                    if self.clear_walk((p.x, p.y), self.center(c), through) {
                        best = k;
                        break;
                    }
                }
                if best == path.len() - 1 {
                    goal_point
                } else {
                    self.center(path[best])
                }
            }
        };
        let (dx, dy) = (wx - p.x, wy - p.y);
        Nav {
            goal,
            x: wx,
            y: wy,
            bearing: wrap(dy.atan2(dx) - p.angle),
            dist: dx.hypot(dy),
            path_len: path.map(|p| p.len() as f32),
        }
    }

    /// The next waypoint toward the goal (cached per tick).
    pub fn nav(&self) -> Nav {
        if let Some((ep, t, nav)) = *self.nav_cache.borrow() {
            if ep == self.episode && t == self.tick {
                return nav;
            }
        }
        let nav = self.compute_nav();
        *self.nav_cache.borrow_mut() = Some((self.episode, self.tick, nav));
        nav
    }

    // ---------------------------------------------------------------- measurements

    fn bearing_to(&self, x: f32, y: f32) -> f32 {
        wrap((y - self.player.y).atan2(x - self.player.x) - self.player.angle)
    }

    /// Live monsters the player can see within [`SIGHT_RANGE`], with bearing and distance.
    fn visible_enemies(&self) -> Vec<(usize, f32, f32)> {
        let p = self.player;
        self.monsters
            .iter()
            .enumerate()
            .filter(|(_, m)| m.alive())
            .filter_map(|(i, m)| {
                let d = (m.x - p.x).hypot(m.y - p.y);
                (d <= SIGHT_RANGE && self.los(p.x, p.y, m.x, m.y))
                    .then(|| (i, self.bearing_to(m.x, m.y), d))
            })
            .collect()
    }

    /// The projectile on a collision course that arrives first: bearing, distance, ticks.
    fn incoming(&self) -> Option<(f32, f32, f32)> {
        let p = self.player;
        let mut best: Option<(f32, f32, f32)> = None;
        for pr in &self.projectiles {
            let (rx, ry) = (pr.x - p.x, pr.y - p.y);
            let vv = pr.vx * pr.vx + pr.vy * pr.vy;
            let t = -(rx * pr.vx + ry * pr.vy) / vv.max(1e-9);
            if t <= 0.0 || t > 60.0 {
                continue;
            }
            let miss = (rx + pr.vx * t).hypot(ry + pr.vy * t);
            if miss > PLAYER_RADIUS + PROJ_RADIUS + 0.1 || !self.los(pr.x, pr.y, p.x, p.y) {
                continue;
            }
            if best.is_none_or(|b| t < b.2) {
                best = Some((self.bearing_to(pr.x, pr.y), rx.hypot(ry), t));
            }
        }
        best
    }

    /// What is within `use` reach straight ahead.
    fn ahead(&self) -> Ahead {
        let p = self.player;
        match self.ray(p.x, p.y, p.angle, USE_RANGE) {
            (_, Some(i)) => match self.map.cells[i] {
                Cell::Exit => Ahead::Exit,
                Cell::Door => {
                    let d = self.map.door_of[i] as usize;
                    if self.doors[d].opening {
                        Ahead::Nothing
                    } else {
                        Ahead::Door
                    }
                }
                _ => Ahead::Nothing,
            },
            _ => Ahead::Nothing,
        }
    }

    /// Every measurement of the state text.
    pub fn features(&self) -> Features {
        let p = self.player;
        let vis = self.visible_enemies();
        let enemy = vis.iter().copied().min_by(|a, b| a.2.total_cmp(&b.2));
        let aligned = enemy.is_some_and(|(_, b, d)| b.abs() <= cone(d));
        Features {
            enemy,
            aligned,
            n_visible: vis.len() as u32,
            wall_ahead: self.ray(p.x, p.y, p.angle, 3.0).0,
            wall_left: self.ray(p.x, p.y, p.angle - PI / 2.0, 3.0).0,
            wall_right: self.ray(p.x, p.y, p.angle + PI / 2.0, 3.0).0,
            ahead: self.ahead(),
            nav: self.nav(),
            proj: self.incoming(),
        }
    }

    /// `ea ed al en hp am wf wl wr dr gb gd pb pd`: nearest visible enemy bearing (10 degree
    /// bins, positive = right) and distance (cells, 0 = none), aligned with the crosshair (0/1),
    /// visible enemies, health (tens), ammo, wall distance ahead / left / right (quarter cells),
    /// what is in use reach ahead (0 nothing, 1 closed door, 2 exit switch), waypoint bearing
    /// and distance (cells), incoming projectile bearing and distance (half cells, 0 = none).
    pub fn state_text(&self) -> String {
        let f = self.features();
        let p = self.player;
        let (ea, ed) = f.enemy.map_or((0, 0), |(_, b, d)| {
            (bin(deg(b), 10.0, -18, 18), bin(d, 1.0, 1, 20))
        });
        let (pb, pd) = f.proj.map_or((0, 0), |(b, d, _)| {
            (bin(deg(b), 10.0, -18, 18), bin(d, 0.5, 1, 20))
        });
        format!(
            "ea {ea} ed {ed} al {} en {} hp {} am {} wf {} wl {} wr {} dr {} gb {} gd {} pb {pb} pd {pd}",
            u8::from(f.aligned),
            f.n_visible.min(9),
            ((p.health.max(0) + 9) / 10).min(10),
            p.ammo.clamp(0, 30),
            bin(f.wall_ahead, 0.25, 0, 12),
            bin(f.wall_left, 0.25, 0, 12),
            bin(f.wall_right, 0.25, 0, 12),
            match f.ahead {
                Ahead::Nothing => 0,
                Ahead::Door => 1,
                Ahead::Exit => 2,
            },
            bin(deg(f.nav.bearing), 10.0, -18, 18),
            bin(f.nav.dist, 1.0, 0, 20),
        )
    }

    // ---------------------------------------------------------------- simulation

    /// Applies a decision (after the safety reflex) and advances one tick. Between decisions
    /// the held action continues and `action` is ignored.
    pub fn step(&mut self, action: DoomAction) -> StepOutcome {
        if self.done() {
            return StepOutcome::new(0.0, true, Event::None, false);
        }
        let mut vetoed = false;
        if self.decision_due() {
            vetoed = self.set_action(action);
        }
        let mut out = self.advance();
        out.info.vetoed = vetoed;
        out
    }

    /// Human control: the action changes on every tick (the reflex still applies); `None`
    /// stands still.
    pub fn step_human(&mut self, action: Option<DoomAction>) -> StepOutcome {
        if self.done() {
            return StepOutcome::new(0.0, true, Event::None, false);
        }
        let vetoed = match action {
            Some(a) => self.set_action(a),
            None => {
                self.action = None;
                self.use_pending = false;
                self.last_vetoed = None;
                false
            }
        };
        let mut out = self.advance();
        out.info.vetoed = vetoed;
        out
    }

    /// Sets the held action through the reflex; returns true when it was replaced.
    fn set_action(&mut self, action: DoomAction) -> bool {
        let chosen = match self.veto_enabled.then(|| self.veto(action)).flatten() {
            Some(r) => {
                self.vetoes += 1;
                self.last_vetoed = Some(action);
                r
            }
            None => {
                self.last_vetoed = None;
                action
            }
        };
        self.action = Some(chosen);
        self.use_pending = chosen == DoomAction::Use;
        self.last_vetoed.is_some()
    }

    /// Holds the current action for one tick.
    pub fn hold(&mut self) -> StepOutcome {
        self.advance()
    }

    fn advance(&mut self) -> StepOutcome {
        if self.done() {
            return StepOutcome::new(0.0, true, Event::None, false);
        }
        self.tick += 1;
        let (hp0, kills0, items0) = (self.player.health, self.kills, self.items_taken);
        self.update_doors();
        self.player_act();
        self.pickups();
        if !self.frozen {
            self.monsters_think();
        }
        self.move_projectiles();

        let hurt = (hp0 - self.player.health).max(0);
        let mut reward = (self.kills - kills0) as f32 + 0.2 * (self.items_taken - items0) as f32
            - 0.01 * hurt as f32;
        let mut event = if self.kills > kills0 {
            Event::Kill
        } else if hurt > 0 {
            Event::Hurt
        } else if self.items_taken > items0 {
            Event::Pickup
        } else {
            Event::None
        };
        if self.player.health <= 0 {
            self.dead = true;
            return StepOutcome::new(reward - 10.0, true, Event::Died, false);
        }
        if self.completed {
            reward += 10.0;
            event = Event::Completed;
        } else if self.tick >= MAX_TICKS {
            self.timed_out = true;
            event = Event::Timeout;
        }
        StepOutcome::new(reward, self.done(), event, false)
    }

    fn update_doors(&mut self) {
        for d in self.doors.iter_mut() {
            if d.opening && d.open < 1.0 {
                let before = d.open;
                d.open = (d.open + DOOR_SPEED).min(1.0);
                if (before < DOOR_SIGHT) != (d.open < DOOR_SIGHT)
                    || (before < DOOR_PASS) != (d.open < DOOR_PASS)
                {
                    self.door_version += 1;
                }
            }
        }
    }

    fn player_blocked(&self, x: f32, y: f32) -> bool {
        self.circle_hits_walls(x, y, PLAYER_RADIUS, None)
            || self
                .monsters
                .iter()
                .any(|m| m.alive() && (m.x - x).hypot(m.y - y) < PLAYER_RADIUS + MONSTER_RADIUS)
    }

    /// Pushes a circle out of the solid cells it overlaps (so movement slides around corners).
    fn push_out(&self, mut x: f32, mut y: f32, r: f32) -> (f32, f32) {
        for _ in 0..3 {
            let mut moved = false;
            let (x0, x1) = ((x - r).floor() as i32, (x + r).floor() as i32);
            let (y0, y1) = ((y - r).floor() as i32, (y + r).floor() as i32);
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    if !self.solid_move(cx, cy) {
                        continue;
                    }
                    let nx = x.clamp(cx as f32, cx as f32 + 1.0);
                    let ny = y.clamp(cy as f32, cy as f32 + 1.0);
                    let d = (x - nx).hypot(y - ny);
                    if d < r && d > 1e-6 {
                        let k = (r + 1e-4 - d) / d;
                        x += (x - nx) * k;
                        y += (y - ny) * k;
                        moved = true;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        (x, y)
    }

    fn move_player(&mut self, dx: f32, dy: f32) {
        let (x, y) = (self.player.x, self.player.y);
        let (nx, ny) = self.push_out(x + dx, y + dy, PLAYER_RADIUS);
        if (nx - x).hypot(ny - y) <= 2.0 * dx.hypot(dy) + 0.02 && !self.player_blocked(nx, ny) {
            self.player.x = nx;
            self.player.y = ny;
            return;
        }
        if !self.player_blocked(x + dx, y) {
            self.player.x = x + dx;
        }
        if !self.player_blocked(self.player.x, y + dy) {
            self.player.y = y + dy;
        }
    }

    fn player_act(&mut self) {
        self.player.cooldown = self.player.cooldown.saturating_sub(1);
        let a = self.player.angle;
        let (fx, fy) = (a.cos(), a.sin());
        let (rx, ry) = (-fy, fx);
        match self.action {
            Some(DoomAction::Forward) => self.move_player(fx * MOVE_SPEED, fy * MOVE_SPEED),
            Some(DoomAction::Back) => self.move_player(-fx * BACK_SPEED, -fy * BACK_SPEED),
            Some(DoomAction::StrafeLeft) => {
                self.move_player(-rx * STRAFE_SPEED, -ry * STRAFE_SPEED)
            }
            Some(DoomAction::StrafeRight) => self.move_player(rx * STRAFE_SPEED, ry * STRAFE_SPEED),
            Some(DoomAction::TurnLeft) => self.player.angle = wrap(a - TURN_RATE),
            Some(DoomAction::TurnRight) => self.player.angle = wrap(a + TURN_RATE),
            Some(DoomAction::Shoot) => self.fire(),
            Some(DoomAction::Use) | None => {}
        }
        if self.use_pending {
            self.use_pending = false;
            self.do_use();
        }
    }

    fn do_use(&mut self) {
        let p = self.player;
        if let (_, Some(i)) = self.ray(p.x, p.y, p.angle, USE_RANGE) {
            match self.map.cells[i] {
                Cell::Exit => self.completed = true,
                Cell::Door => {
                    let d = self.map.door_of[i] as usize;
                    self.doors[d].opening = true;
                }
                _ => {}
            }
        }
    }

    /// The monster a shot fired now would hit: the nearest one inside the aim cone.
    fn shot_target(&self) -> Option<usize> {
        let p = self.player;
        self.monsters
            .iter()
            .enumerate()
            .filter(|(_, m)| m.alive())
            .filter_map(|(i, m)| {
                let d = (m.x - p.x).hypot(m.y - p.y);
                let b = self.bearing_to(m.x, m.y);
                (d <= SHOT_RANGE && b.abs() <= cone(d) && self.los(p.x, p.y, m.x, m.y))
                    .then_some((i, d))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    fn fire(&mut self) {
        if self.player.cooldown > 0 || self.player.ammo <= 0 {
            return;
        }
        self.player.ammo -= 1;
        self.player.cooldown = FIRE_COOLDOWN;
        self.last_fire = Some(self.tick);
        if let Some(i) = self.shot_target() {
            let dmg = self.rng.gen_range(25..=45);
            self.damage_monster(i, dmg);
        }
        if !self.frozen {
            self.make_noise();
        }
    }

    fn damage_monster(&mut self, i: usize, dmg: i32) {
        let pain = self.rng.gen_bool(0.78);
        let react = self.rng.gen_range(8..=20);
        let m = &mut self.monsters[i];
        m.hp -= dmg;
        if m.hp <= 0 {
            m.state = MonsterState::Dead;
            m.state_ticks = 0;
            self.kills += 1;
            return;
        }
        if m.state == MonsterState::Idle {
            m.attack_cd = react;
        }
        if pain {
            m.state = MonsterState::Pain;
            m.state_ticks = 0;
        } else if m.state == MonsterState::Idle {
            m.state = MonsterState::Chase;
            m.state_ticks = 0;
        }
    }

    /// Gunfire wakes the monsters within [`NOISE_RANGE`] cells of open path.
    fn make_noise(&mut self) {
        let (w, n) = (self.map.w, self.map.w * self.map.h);
        let mut dist = vec![UNREACHED; n];
        let start = self.cell_of(self.player.x, self.player.y);
        dist[start] = 0;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(c) = queue.pop_front() {
            if dist[c] >= NOISE_RANGE {
                continue;
            }
            let (x, y) = ((c % w) as i32, (c / w) as i32);
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                if let Some(ni) = self.idx(x + dx, y + dy) {
                    if dist[ni] == UNREACHED && !self.solid_sight(x + dx, y + dy) {
                        dist[ni] = dist[c] + 1;
                        queue.push_back(ni);
                    }
                }
            }
        }
        for i in 0..self.monsters.len() {
            let m = self.monsters[i];
            if m.state == MonsterState::Idle && dist[self.cell_of(m.x, m.y)] != UNREACHED {
                self.wake(i);
            }
        }
    }

    fn wake(&mut self, i: usize) {
        let react = self.rng.gen_range(12..=30);
        let m = &mut self.monsters[i];
        if m.state == MonsterState::Idle {
            m.state = MonsterState::Chase;
            m.state_ticks = 0;
            m.attack_cd = react;
        }
    }

    fn pickups(&mut self) {
        let p = self.player;
        for it in self.items.iter_mut().filter(|it| !it.taken) {
            if (it.x - p.x).hypot(it.y - p.y) > PICKUP_RADIUS {
                continue;
            }
            let used = match it.kind {
                ItemKind::Health | ItemKind::Medikit if self.player.health < MAX_HEALTH => {
                    let add = if it.kind == ItemKind::Health { 10 } else { 25 };
                    self.player.health = (self.player.health + add).min(MAX_HEALTH);
                    true
                }
                ItemKind::Ammo if self.player.ammo < MAX_AMMO => {
                    self.player.ammo = (self.player.ammo + 10).min(MAX_AMMO);
                    true
                }
                _ => false,
            };
            if used {
                it.taken = true;
                self.items_taken += 1;
            }
        }
    }

    fn hurt_player(&mut self, dmg: i32) {
        self.player.health -= dmg;
        self.last_hurt = Some(self.tick);
    }

    /// Breadth-first distances toward the player over cells monsters can walk.
    fn ensure_flow(&mut self) {
        let start = self.cell_of(self.player.x, self.player.y);
        let key = (start, self.door_version);
        if self.flow_key == key && !self.flow.is_empty() {
            return;
        }
        let w = self.map.w;
        let mut dist = vec![UNREACHED; w * self.map.h];
        dist[start] = 0;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(c) = queue.pop_front() {
            let (x, y) = ((c % w) as i32, (c / w) as i32);
            for (dx, dy) in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ] {
                let Some(ni) = self.idx(x + dx, y + dy) else {
                    continue;
                };
                if dist[ni] != UNREACHED || self.solid_move(x + dx, y + dy) {
                    continue;
                }
                if dx != 0 && dy != 0 && (self.solid_move(x + dx, y) || self.solid_move(x, y + dy))
                {
                    continue;
                }
                dist[ni] = dist[c] + 1;
                queue.push_back(ni);
            }
        }
        self.flow = Arc::new(dist);
        self.flow_key = key;
    }

    /// Where a monster without sight of the player walks next (a neighbor cell center).
    fn flow_target(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let c = self.cell_of(x, y);
        let here = *self.flow.get(c)?;
        if here == UNREACHED || here == 0 {
            return None;
        }
        let (cx, cy) = ((c % self.map.w) as i32, (c / self.map.w) as i32);
        let mut best = (here, usize::MAX);
        for (dx, dy) in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ] {
            if let Some(ni) = self.idx(cx + dx, cy + dy) {
                if self.flow[ni] < best.0 {
                    best = (self.flow[ni], ni);
                }
            }
        }
        (best.1 != usize::MAX).then(|| self.center(best.1))
    }

    fn monster_blocked(&self, i: usize, x: f32, y: f32) -> bool {
        let p = self.player;
        self.circle_hits_walls(x, y, MONSTER_RADIUS, None)
            || (p.x - x).hypot(p.y - y) < PLAYER_RADIUS + MONSTER_RADIUS
            || self.monsters.iter().enumerate().any(|(j, o)| {
                j != i && o.alive() && (o.x - x).hypot(o.y - y) < 2.0 * MONSTER_RADIUS * 0.9
            })
    }

    fn monsters_think(&mut self) {
        let params = self.skill.params();
        let p = self.player;
        for i in 0..self.monsters.len() {
            let mut m = self.monsters[i];
            if !m.alive() {
                m.state_ticks += 1;
                self.monsters[i] = m;
                continue;
            }
            let (dx, dy) = (p.x - m.x, p.y - m.y);
            let d = dx.hypot(dy);
            m.state_ticks += 1;
            match m.state {
                MonsterState::Dead => {}
                MonsterState::Idle => {
                    let facing = m.angle.cos() * dx + m.angle.sin() * dy >= 0.0;
                    if (self.tick + m.id).is_multiple_of(4)
                        && d <= WAKE_RANGE
                        && (facing || d < 5.0)
                        && self.los(m.x, m.y, p.x, p.y)
                    {
                        m.state = MonsterState::Chase;
                        m.state_ticks = 0;
                        m.attack_cd = self.rng.gen_range(12..=30);
                    }
                }
                MonsterState::Pain => {
                    if m.state_ticks >= PAIN_TICKS {
                        m.state = MonsterState::Chase;
                        m.state_ticks = 0;
                    }
                }
                MonsterState::Attack => {
                    m.angle = dy.atan2(dx);
                    let fire_at = match m.kind {
                        MonsterKind::Imp => 6,
                        MonsterKind::Zombie => 5,
                    };
                    if m.state_ticks == fire_at && self.los(m.x, m.y, p.x, p.y) {
                        self.monster_attack(&m, d);
                    }
                    if m.state_ticks >= ATTACK_TICKS {
                        m.state = MonsterState::Chase;
                        m.state_ticks = 0;
                        let (lo, hi) = match m.kind {
                            MonsterKind::Imp => params.imp_cd,
                            MonsterKind::Zombie => params.zombie_cd,
                        };
                        m.attack_cd = self.rng.gen_range(lo..=hi);
                    }
                }
                MonsterState::Chase => {
                    m.attack_cd = m.attack_cd.saturating_sub(1);
                    let sees = self.los(m.x, m.y, p.x, p.y);
                    if sees {
                        m.angle = dy.atan2(dx);
                        if m.attack_cd == 0 && d <= SIGHT_RANGE {
                            m.state = MonsterState::Attack;
                            m.state_ticks = 0;
                            self.monsters[i] = m;
                            continue;
                        }
                    }
                    let (speed, keep) = match m.kind {
                        MonsterKind::Imp => (0.06, PLAYER_RADIUS + MONSTER_RADIUS + 0.05),
                        MonsterKind::Zombie => (0.05, 2.5),
                    };
                    let target = if sees {
                        (d > keep).then_some((p.x, p.y))
                    } else {
                        self.ensure_flow();
                        self.flow_target(m.x, m.y)
                    };
                    if let Some((tx, ty)) = target {
                        let (ux, uy) = (tx - m.x, ty - m.y);
                        let n = ux.hypot(uy).max(1e-6);
                        let (mx, my) = (ux / n * speed, uy / n * speed);
                        if !sees {
                            m.angle = uy.atan2(ux);
                        }
                        let (nx, ny) = self.push_out(m.x + mx, m.y + my, MONSTER_RADIUS);
                        if (nx - m.x).hypot(ny - m.y) <= 2.0 * speed + 0.02
                            && !self.monster_blocked(i, nx, ny)
                        {
                            m.x = nx;
                            m.y = ny;
                        } else {
                            if !self.monster_blocked(i, m.x + mx, m.y) {
                                m.x += mx;
                            }
                            if !self.monster_blocked(i, m.x, m.y + my) {
                                m.y += my;
                            }
                        }
                    }
                }
            }
            self.monsters[i] = m;
        }
    }

    fn monster_attack(&mut self, m: &Monster, d: f32) {
        let params = self.skill.params();
        match m.kind {
            MonsterKind::Imp if d <= PLAYER_RADIUS + MONSTER_RADIUS + 0.35 => {
                let dmg = self.rng.gen_range(params.claw_dmg.0..=params.claw_dmg.1);
                self.hurt_player(dmg);
            }
            MonsterKind::Imp => {
                let (dx, dy) = ((self.player.x - m.x) / d, (self.player.y - m.y) / d);
                let dmg = self
                    .rng
                    .gen_range(params.fireball_dmg.0..=params.fireball_dmg.1);
                self.next_id += 1;
                self.projectiles.push(Projectile {
                    id: self.next_id,
                    x: m.x + dx * (MONSTER_RADIUS + 0.1),
                    y: m.y + dy * (MONSTER_RADIUS + 0.1),
                    vx: dx * PROJ_SPEED,
                    vy: dy * PROJ_SPEED,
                    damage: dmg,
                });
            }
            MonsterKind::Zombie => {
                let p_hit = (0.9 - 0.04 * d).clamp(0.25, 0.9);
                if self.rng.gen_bool(p_hit as f64) {
                    let dmg = self.rng.gen_range(params.rifle_dmg.0..=params.rifle_dmg.1);
                    self.hurt_player(dmg);
                }
            }
        }
    }

    fn move_projectiles(&mut self) {
        let mut k = 0;
        while k < self.projectiles.len() {
            let mut pr = self.projectiles[k];
            let mut gone = false;
            for _ in 0..2 {
                pr.x += pr.vx * 0.5;
                pr.y += pr.vy * 0.5;
                if self.solid_sight(pr.x.floor() as i32, pr.y.floor() as i32) {
                    gone = true;
                    break;
                }
                let p = self.player;
                if (pr.x - p.x).hypot(pr.y - p.y) < PLAYER_RADIUS + PROJ_RADIUS {
                    self.hurt_player(pr.damage);
                    self.proj_hits += 1;
                    gone = true;
                    break;
                }
            }
            if gone {
                self.projectiles.swap_remove(k);
            } else {
                self.projectiles[k] = pr;
                k += 1;
            }
        }
    }

    // ---------------------------------------------------------------- reflex and oracle

    fn projectile_near(&self) -> bool {
        let p = self.player;
        let reach = PROJ_SPEED * DODGE_HORIZON as f32 + 1.0;
        self.projectiles
            .iter()
            .any(|pr| (pr.x - p.x).hypot(pr.y - p.y) <= reach)
    }

    /// Holds `action` for `horizon` ticks with the monsters frozen: was the player hit by a
    /// projectile already in flight? Also returns the smallest gap to a projectile.
    fn sim_hit(&self, action: DoomAction, horizon: u32) -> (bool, f32) {
        let mut sim = self.clone();
        sim.frozen = true;
        sim.veto_enabled = false;
        sim.action = Some(action);
        sim.use_pending = false;
        let hits0 = sim.proj_hits;
        let mut gap = f32::MAX;
        for _ in 0..horizon {
            sim.advance();
            if sim.proj_hits > hits0 {
                return (true, 0.0);
            }
            let p = sim.player;
            for pr in &sim.projectiles {
                gap = gap.min((pr.x - p.x).hypot(pr.y - p.y) - PLAYER_RADIUS - PROJ_RADIUS);
            }
            if sim.done() {
                break;
            }
        }
        (false, gap)
    }

    /// The navigation part of the oracle as a coded fallback (used by the reflex when the
    /// weapon is empty).
    fn nav_action(&self) -> DoomAction {
        DoomAction::from_index(argmax(&self.base_policy(&self.features(), false)))
    }

    /// The safety reflex: an empty weapon never fires (the player navigates instead), and the
    /// player never holds an action that walks into a known projectile when a dodge is free.
    /// Returns the replacement, or `None` when `action` is kept.
    pub fn veto(&self, action: DoomAction) -> Option<DoomAction> {
        let mut req = action;
        if action == DoomAction::Shoot && self.player.ammo <= 0 {
            req = self.nav_action();
        }
        if self.projectile_near() && self.sim_hit(req, VETO_HORIZON).0 {
            if let Some(alt) = DODGES
                .into_iter()
                .filter(|a| *a != req)
                .find(|a| !self.sim_hit(*a, VETO_HORIZON).0)
            {
                return Some(alt);
            }
        }
        (req != action).then_some(req)
    }

    /// The oracle's policy from the measured features alone (no dodge): use the exit, fight
    /// the nearest visible enemy (turn toward it, shoot when aligned), open doors on the
    /// route, otherwise follow the waypoint. Soft near the decision boundaries.
    fn base_policy(&self, f: &Features, combat: bool) -> Vec<f32> {
        use DoomAction as A;
        let mut p = vec![0.0f32; 8];
        if f.ahead == Ahead::Exit {
            p[A::Use.index()] = 1.0;
            return p;
        }
        if combat && self.player.ammo > 0 {
            if let Some((_, b, d)) = f.enemy {
                let ps = sigmoid((deg(cone(d)) - deg(b.abs())) / 1.5);
                let turn = if b > 0.0 { A::TurnRight } else { A::TurnLeft };
                p[A::Shoot.index()] = ps;
                p[turn.index()] = 1.0 - ps;
                return p;
            }
        }
        let gb = deg(f.nav.bearing);
        if f.ahead == Ahead::Door && gb.abs() <= DOOR_USE_DEG {
            p[A::Use.index()] = 1.0;
            return p;
        }
        let pt = sigmoid((gb.abs() - NAV_TURN_DEG) / 3.0);
        let turn = if gb > 0.0 { A::TurnRight } else { A::TurnLeft };
        p[turn.index()] = pt;
        p[A::Forward.index()] = 1.0 - pt;
        p
    }

    /// Soft targets of the `action` question.
    fn action_target(&self, f: &Features) -> Vec<f32> {
        let base = self.base_policy(f, true);
        if self.projectile_near() {
            let best = DoomAction::from_index(argmax(&base));
            if self.sim_hit(best, DODGE_HORIZON).0 {
                let safe: Vec<(DoomAction, f32)> = DODGES
                    .into_iter()
                    .filter_map(|a| {
                        let (hit, gap) = self.sim_hit(a, DODGE_HORIZON);
                        (!hit).then_some((a, gap))
                    })
                    .collect();
                if let Some(top) = safe.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)) {
                    let mut p = vec![0.0f32; 8];
                    let rest = (safe.len() - 1) as f32;
                    for (a, _) in &safe {
                        p[a.index()] = if *a == top.0 {
                            if rest > 0.0 {
                                0.85
                            } else {
                                1.0
                            }
                        } else {
                            0.15 / rest
                        };
                    }
                    return with_floor(p, 0.01);
                }
            }
        }
        with_floor(base, 0.01)
    }

    /// Soft target of `danger`: does the player, standing still, take damage within a second?
    fn danger_target(&self) -> Vec<f32> {
        let mut sim = self.clone();
        sim.veto_enabled = false;
        sim.action = None;
        sim.use_pending = false;
        let hp0 = sim.player.health;
        let mut first = None;
        for k in 1..=DANGER_TICKS + 10 {
            sim.advance();
            if sim.player.health < hp0 {
                first = Some(k);
                break;
            }
            if sim.done() {
                break;
            }
        }
        let p = first.map_or(0.0, |k| {
            sigmoid(((DANGER_TICKS as f32 + 0.5 - k as f32) / 3.0).clamp(-4.0, 4.0))
        });
        with_floor(vec![1.0 - p, p], 0.02)
    }

    /// Threat value: awake visible monsters weighted by kind and distance, an incoming
    /// projectile, amplified when health is low.
    pub fn threat_value(&self, f: &Features) -> f32 {
        let p = self.player;
        let mut t = 0.0;
        for m in self
            .monsters
            .iter()
            .filter(|m| m.alive() && m.state != MonsterState::Idle)
        {
            let d = (m.x - p.x).hypot(m.y - p.y);
            if d > SIGHT_RANGE || !self.los(p.x, p.y, m.x, m.y) {
                continue;
            }
            let w = match m.kind {
                MonsterKind::Imp => 1.0,
                MonsterKind::Zombie => 0.8,
            };
            let fd = if d < 3.0 {
                1.0
            } else if d < 7.0 {
                0.7
            } else {
                0.45
            };
            t += w * fd;
        }
        if f.proj.is_some_and(|(_, _, tti)| tti <= 40.0) {
            t += 0.8;
        }
        t * (1.0 + (50 - p.health).max(0) as f32 / 50.0)
    }

    /// Soft target of `threat` (5 levels).
    fn threat_target(&self, f: &Features) -> Vec<f32> {
        const CENTERS: [f32; 5] = [0.0, 0.45, 1.1, 2.0, 3.2];
        let t = self.threat_value(f).min(5.0);
        let p: Vec<f32> = CENTERS
            .iter()
            .map(|c| (-(t - c).powi(2) / (2.0 * 0.28 * 0.28)).exp() + 1e-6)
            .collect();
        with_floor(p, 0.01)
    }

    /// Soft targets: `action` (8), `danger` (`[false, true]`), `threat` (5 levels).
    pub fn oracle(&self) -> Vec<Vec<f32>> {
        let f = self.features();
        vec![
            self.action_target(&f),
            self.danger_target(),
            self.threat_target(&f),
        ]
    }

    pub fn oracle_action(&self) -> DoomAction {
        DoomAction::from_index(argmax(&self.action_target(&self.features())))
    }

    /// Points: 100 per kill, 25 per item, 1000 for the exit plus a time bonus.
    pub fn points(&self) -> u32 {
        let mut s = 100 * self.kills + 25 * self.items_taken;
        if self.completed {
            let secs = self.tick / TICRATE;
            s += 1000 + 5 * 120u32.saturating_sub(secs);
        }
        s
    }

    pub fn snapshot(&self) -> DoomSnapshot {
        let p = self.player;
        let w = self.map.w;
        let vis = |x: f32, y: f32| self.los(p.x, p.y, x, y);
        let mut sprites: Vec<SpriteView> = Vec::new();
        for m in &self.monsters {
            sprites.push(SpriteView {
                id: m.id,
                kind: match m.kind {
                    MonsterKind::Imp => "imp",
                    MonsterKind::Zombie => "zombie",
                },
                x: m.x,
                y: m.y,
                state: match m.state {
                    MonsterState::Idle => "idle",
                    MonsterState::Chase => "chase",
                    MonsterState::Attack => "attack",
                    MonsterState::Pain => "pain",
                    MonsterState::Dead => "dead",
                },
                angle: m.angle,
                anim: m.state_ticks,
                visible: vis(m.x, m.y),
            });
        }
        for pr in &self.projectiles {
            sprites.push(SpriteView {
                id: pr.id,
                kind: "fireball",
                x: pr.x,
                y: pr.y,
                state: "none",
                angle: pr.vy.atan2(pr.vx),
                anim: self.tick,
                visible: vis(pr.x, pr.y),
            });
        }
        for it in self.items.iter().filter(|it| !it.taken) {
            sprites.push(SpriteView {
                id: it.id,
                kind: match it.kind {
                    ItemKind::Health => "health",
                    ItemKind::Medikit => "medikit",
                    ItemKind::Ammo => "ammo",
                },
                x: it.x,
                y: it.y,
                state: "none",
                angle: 0.0,
                anim: self.tick,
                visible: vis(it.x, it.y),
            });
        }
        let nav = (!self.done()).then(|| self.nav());
        DoomSnapshot {
            map: self.map.name.clone(),
            skill: self.skill,
            width: w,
            height: self.map.h,
            cells: self.map.cells.iter().map(|c| c.code()).collect(),
            doors: self
                .doors
                .iter()
                .map(|d| DoorView {
                    x: d.cell % w,
                    y: d.cell / w,
                    open: d.open,
                })
                .collect(),
            exit: ExitView {
                x: self.map.exit % w,
                y: self.map.exit / w,
                pressed: self.completed,
            },
            player: PlayerView {
                x: p.x,
                y: p.y,
                angle: p.angle,
                health: p.health.max(0),
                ammo: p.ammo,
                radius: PLAYER_RADIUS,
            },
            sprites,
            waypoint: nav.map(|n| PointView { x: n.x, y: n.y }),
            tick: self.tick,
            kills: self.kills,
            monsters: self.monsters.len() as u32,
            items: self.items_taken,
            items_total: self.items.len() as u32,
            score: self.points(),
            fire_tick: self.last_fire.map_or(9999, |t| self.tick - t),
            hurt_tick: self.last_hurt.map_or(9999, |t| self.tick - t),
            action: self.action,
            vetoed: self.last_vetoed.is_some(),
            vetoed_request: self.last_vetoed,
            dead: self.dead,
            completed: self.completed,
        }
    }

    pub fn questions() -> &'static [QuestionSpec] {
        QUESTIONS
    }
}

impl DecisionGame for Doom {
    fn name(&self) -> &'static str {
        NAME
    }
    fn reset(&mut self) {
        Doom::reset(self)
    }
    fn done(&self) -> bool {
        Doom::done(self)
    }
    fn decision_due(&self) -> bool {
        Doom::decision_due(self)
    }
    fn questions(&self) -> &'static [QuestionSpec] {
        QUESTIONS
    }
    fn state_text(&self) -> String {
        Doom::state_text(self)
    }
    fn oracle(&self) -> Vec<Vec<f32>> {
        Doom::oracle(self)
    }
    fn apply(&mut self, choice: usize, _abort: bool) -> StepOutcome {
        self.step(DoomAction::from_index(choice))
    }
    /// `choice` outside the eight actions stands still.
    fn apply_human(&mut self, choice: usize) -> StepOutcome {
        self.step_human(DoomAction::ALL.get(choice).copied())
    }
    fn tick(&mut self) -> StepOutcome {
        self.hold()
    }
    fn stats(&self) -> EpisodeStats {
        EpisodeStats {
            score: self.kills as f32,
            max_score: self.monsters.len() as f32,
            success: self.completed,
            crashed: self.dead,
            vetoes: self.vetoes,
            steps: self.tick,
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

// -------------------------------------------------------------------- levels

/// Hand-designed levels. Legend: `#` `%` `&` `$` `@` (and space) walls with textures 1..5,
/// `.` floor, `D` door, `X` exit switch, `>` `<` `^` `v` player start and facing, `i` imp,
/// `z` zombieman, `+` stimpack, `M` medikit, `a` ammo.
pub static HAND_MAPS: &[(&str, &[&str])] = &[
    (
        "E1 Entryway",
        &[
            "#########################",
            "#.......%.......&.......#",
            "#.>.....%...z...&...i...#",
            "#.......D.......D.......#",
            "#...a...%.......&.......#",
            "#.......%...+...&...a...#",
            "####.####.......&&&&D&&&#",
            "#.......%%%%%%%%$.......#",
            "#..z....................X",
            "#.......$$$$$$$$$...i...#",
            "#...+...#       $.......#",
            "#########       #########",
        ],
    ),
    (
        "E2 Warehouse",
        &[
            "##########################",
            "#>....%.............@....#",
            "#.....%.....z.......@..M.#",
            "#.....%.............D....#",
            "#..a..D.....i.......@....#",
            "#.....%.............@@@@@#",
            "###.###.............@....#",
            "#.....%%%%%%D%%%%%%%@.z..#",
            "#..z..$.........$........#",
            "#.....$...+.....D....a...#",
            "#.....$.........$........#",
            "#.....D.........$$$$$X$$$#",
            "#..i..$.........$        #",
            "#######.........$        #",
            "      $$$$$$$$$$$        #",
        ],
    ),
    (
        "E3 Courtyard",
        &[
            "############################",
            "#..........&&&&&&&.........#",
            "#.>........D.....D....z....#",
            "#..........&.....&.........#",
            "#....%%....&..i..&...%%....#",
            "#....%%....&.....&...%%....#",
            "#..........&&&D&&&.........#",
            "#...a......&.....&.....a...#",
            "######D#####.....#####D#####",
            "#..........$..+..$.........#",
            "#..z....i..$.....$..z......#",
            "#..........D.....D.........#",
            "#.....M....$.....$.......i.#",
            "#..........$.....$.........#",
            "##############X#############",
        ],
    ),
    (
        "E4 Tech Base",
        &[
            "##############################",
            "#>..#.........#..............#",
            "#...D....z....D.......i......#",
            "#...#.........#..............#",
            "#.a.###########.....@@@@.....#",
            "#...#         #.....@..@.....#",
            "#...#         #..+..@..@..z..#",
            "#...#         #.....@@@@.....#",
            "#...###########..............#",
            "#.................z..........#",
            "#...#############D############",
            "#...%.........%.......%......#",
            "#.i.D....+....D...i...D......X",
            "#...%.........%.......%..M...#",
            "##############################",
        ],
    ),
];

fn wall_tex(c: char) -> Option<u8> {
    match c {
        '#' | ' ' => Some(1),
        '%' => Some(2),
        '&' => Some(3),
        '$' => Some(4),
        '@' => Some(5),
        _ => None,
    }
}

fn finish_map(name: String, w: usize, h: usize, cells: Vec<Cell>) -> MapData {
    let mut door_of = vec![u16::MAX; w * h];
    let mut n = 0u16;
    let mut exit = 0;
    for (i, c) in cells.iter().enumerate() {
        match c {
            Cell::Door => {
                door_of[i] = n;
                n += 1;
            }
            Cell::Exit => exit = i,
            _ => {}
        }
    }
    MapData {
        name,
        w,
        h,
        cells,
        door_of,
        exit,
    }
}

/// Parses a hand-designed level, optionally mirrored.
fn parse_map(name: &str, rows: &[&str], flip_x: bool, flip_y: bool) -> Level {
    let h = rows.len();
    let w = rows.iter().map(|r| r.chars().count()).max().unwrap_or(1);
    let at = |x: usize, y: usize| -> char {
        let (sx, sy) = (
            if flip_x { w - 1 - x } else { x },
            if flip_y { h - 1 - y } else { y },
        );
        rows[sy].chars().nth(sx).unwrap_or(' ')
    };
    let mut cells = Vec::with_capacity(w * h);
    let mut start = (1.5, 1.5, 0.0);
    let mut monsters = Vec::new();
    let mut items = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let c = at(x, y);
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            let cell = if let Some(t) = wall_tex(c) {
                Cell::Wall(t)
            } else {
                match c {
                    'D' => Cell::Door,
                    'X' => Cell::Exit,
                    _ => Cell::Floor,
                }
            };
            match c {
                '>' | '<' | '^' | 'v' => {
                    let a: f32 = match c {
                        '>' => 0.0,
                        'v' => PI / 2.0,
                        '<' => PI,
                        _ => -PI / 2.0,
                    };
                    let (mut dx, mut dy) = (a.cos(), a.sin());
                    if flip_x {
                        dx = -dx;
                    }
                    if flip_y {
                        dy = -dy;
                    }
                    start = (cx, cy, dy.atan2(dx));
                }
                'i' | 'z' => {
                    let kind = if c == 'i' {
                        MonsterKind::Imp
                    } else {
                        MonsterKind::Zombie
                    };
                    monsters.push((kind, cx, cy, ((x * 7 + y * 3) % 4) as f32 * PI / 2.0));
                }
                '+' => items.push((ItemKind::Health, cx, cy)),
                'M' => items.push((ItemKind::Medikit, cx, cy)),
                'a' => items.push((ItemKind::Ammo, cx, cy)),
                _ => {}
            }
            cells.push(cell);
        }
    }
    Level {
        map: finish_map(name.to_string(), w, h, cells),
        start,
        monsters,
        items,
    }
}

#[derive(Debug, Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Rect {
    fn center(&self) -> (usize, usize) {
        (self.x + self.w / 2, self.y + self.h / 2)
    }
    fn overlaps(&self, o: &Rect, margin: usize) -> bool {
        self.x < o.x + o.w + margin
            && o.x < self.x + self.w + margin
            && self.y < o.y + o.h + margin
            && o.y < self.y + self.h + margin
    }
    fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.x + self.w && y >= self.y && y < self.y + self.h
    }
}

/// A procedural level: rooms joined by corridors (some with doors), the exit switch in the
/// room farthest from the start, monsters away from the start.
fn generate_level(rng: &mut SmallRng, skill: Skill) -> Level {
    loop {
        if let Some(level) = try_generate(rng, skill) {
            return level;
        }
    }
}

fn try_generate(rng: &mut SmallRng, skill: Skill) -> Option<Level> {
    let w = rng.gen_range(24..=32);
    let h = rng.gen_range(18..=24);
    let n_rooms = rng.gen_range(5..=7);
    let mut rooms: Vec<Rect> = Vec::new();
    for _ in 0..300 {
        if rooms.len() == n_rooms {
            break;
        }
        let (rw, rh) = (rng.gen_range(4..=8), rng.gen_range(4..=7));
        let r = Rect {
            x: rng.gen_range(1..w - rw - 1),
            y: rng.gen_range(1..h - rh - 1),
            w: rw,
            h: rh,
        };
        if rooms.iter().all(|o| !r.overlaps(o, 2)) {
            rooms.push(r);
        }
    }
    if rooms.len() < 4 {
        return None;
    }
    // Chain the rooms from west to east so corridors stay short.
    rooms.sort_by_key(|r| r.center().0 * 4 + r.center().1);
    // Room id per cell (255 = none), corridor flag.
    let mut room_of = vec![255u8; w * h];
    let mut floor = vec![false; w * h];
    for (k, r) in rooms.iter().enumerate() {
        for y in r.y..r.y + r.h {
            for x in r.x..r.x + r.w {
                room_of[y * w + x] = k as u8;
                floor[y * w + x] = true;
            }
        }
    }
    let mut corridor = vec![false; w * h];
    let mut carve = |x: usize, y: usize, floor: &mut Vec<bool>| {
        if !floor[y * w + x] {
            floor[y * w + x] = true;
            corridor[y * w + x] = true;
        }
    };
    let mut links: Vec<(usize, usize)> = (1..rooms.len()).map(|k| (k - 1, k)).collect();
    if rng.gen_bool(0.5) && rooms.len() >= 4 {
        let a = rng.gen_range(0..rooms.len() - 2);
        links.push((a, a + 2));
    }
    for (a, b) in links {
        let (ax, ay) = rooms[a].center();
        let (bx, by) = rooms[b].center();
        if rng.gen_bool(0.5) {
            for x in ax.min(bx)..=ax.max(bx) {
                carve(x, ay, &mut floor);
            }
            for y in ay.min(by)..=ay.max(by) {
                carve(bx, y, &mut floor);
            }
        } else {
            for y in ay.min(by)..=ay.max(by) {
                carve(ax, y, &mut floor);
            }
            for x in ax.min(bx)..=ax.max(bx) {
                carve(x, by, &mut floor);
            }
        }
    }
    let at = |x: i32, y: i32| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < w
            && (y as usize) < h
            && floor[y as usize * w + x as usize]
    };
    // Doors: corridor cells in a one-wide passage that touch a room.
    let mut cells = vec![Cell::Floor; w * h];
    let mut is_door = vec![false; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if !corridor[i] {
                continue;
            }
            let (xi, yi) = (x as i32, y as i32);
            let horiz = at(xi - 1, yi) && at(xi + 1, yi) && !at(xi, yi - 1) && !at(xi, yi + 1);
            let vert = at(xi, yi - 1) && at(xi, yi + 1) && !at(xi - 1, yi) && !at(xi + 1, yi);
            let touches_room = [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|(dx, dy)| {
                let (nx, ny) = (xi + dx, yi + dy);
                at(nx, ny) && room_of[ny as usize * w + nx as usize] != 255
            });
            let near_door = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .iter()
                .any(|(dx, dy)| is_door[(yi + dy) as usize * w + (xi + dx) as usize]);
            if (horiz || vert) && touches_room && !near_door && rng.gen_bool(0.45) {
                is_door[i] = true;
            }
        }
    }
    // Wall textures follow the nearest room; corridors use texture 1.
    let tex: Vec<u8> = (0..rooms.len()).map(|_| rng.gen_range(1..=5)).collect();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if floor[i] {
                cells[i] = if is_door[i] { Cell::Door } else { Cell::Floor };
                continue;
            }
            let mut t = 1;
            'n: for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let r = room_of[ny as usize * w + nx as usize];
                        if r != 255 {
                            t = tex[r as usize];
                            break 'n;
                        }
                    }
                }
            }
            cells[i] = Cell::Wall(t);
        }
    }
    // Start in room 0, facing room 1.
    let (sx, sy) = rooms[0].center();
    let (nx, ny) = rooms[1].center();
    let (dx, dy) = (nx as f32 - sx as f32, ny as f32 - sy as f32);
    let start_angle = if dx.abs() >= dy.abs() {
        if dx >= 0.0 {
            0.0
        } else {
            PI
        }
    } else if dy >= 0.0 {
        PI / 2.0
    } else {
        -PI / 2.0
    };
    // Breadth-first distances from the start (doors passable).
    let mut dist = vec![u32::MAX; w * h];
    let s = sy * w + sx;
    dist[s] = 0;
    let mut queue = std::collections::VecDeque::from([s]);
    while let Some(c) = queue.pop_front() {
        let (x, y) = ((c % w) as i32, (c / w) as i32);
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            if at(x + dx, y + dy) {
                let ni = (y + dy) as usize * w + (x + dx) as usize;
                if dist[ni] == u32::MAX {
                    dist[ni] = dist[c] + 1;
                    queue.push_back(ni);
                }
            }
        }
    }
    let far = (1..rooms.len()).max_by_key(|&k| {
        let (cx, cy) = rooms[k].center();
        dist[cy * w + cx]
    })?;
    if rooms
        .iter()
        .any(|r| dist[r.center().1 * w + r.center().0] == u32::MAX)
    {
        return None;
    }
    // Exit switch: a wall cell on the far room's boundary with exactly one floor neighbor.
    let r = rooms[far];
    let mut candidates = Vec::new();
    for y in r.y.saturating_sub(1)..=(r.y + r.h).min(h - 1) {
        for x in r.x.saturating_sub(1)..=(r.x + r.w).min(w - 1) {
            let i = y * w + x;
            if floor[i] {
                continue;
            }
            let (xi, yi) = (x as i32, y as i32);
            let n: Vec<(i32, i32)> = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                .into_iter()
                .filter(|(dx, dy)| at(xi + dx, yi + dy))
                .collect();
            if n.len() == 1 {
                let (fx, fy) = ((xi + n[0].0) as usize, (yi + n[0].1) as usize);
                if r.contains(fx, fy) && !corridor[fy * w + fx] {
                    candidates.push(i);
                }
            }
        }
    }
    if candidates.is_empty() {
        return None;
    }
    let exit = candidates[rng.gen_range(0..candidates.len())];
    cells[exit] = Cell::Exit;

    // Monsters and items on free floor cells of the rooms.
    let mut used = vec![false; w * h];
    used[s] = true;
    let free_cell = |rng: &mut SmallRng, k: usize, used: &mut Vec<bool>| -> Option<(f32, f32)> {
        let r = rooms[k];
        for _ in 0..30 {
            let x = rng.gen_range(r.x..r.x + r.w);
            let y = rng.gen_range(r.y..r.y + r.h);
            let i = y * w + x;
            let d0 = (x as f32 - sx as f32).hypot(y as f32 - sy as f32);
            if !used[i] && d0 >= 6.0 {
                used[i] = true;
                return Some((x as f32 + 0.5, y as f32 + 0.5));
            }
        }
        None
    };
    let p = skill.params();
    let n_monsters = rng.gen_range(p.monsters.0..=p.monsters.1);
    let mut monsters = Vec::new();
    for _ in 0..n_monsters {
        let k = rng.gen_range(1..rooms.len());
        if let Some((x, y)) = free_cell(rng, k, &mut used) {
            let kind = if rng.gen_bool(0.5) {
                MonsterKind::Imp
            } else {
                MonsterKind::Zombie
            };
            monsters.push((kind, x, y, rng.gen_range(0..4) as f32 * PI / 2.0));
        }
    }
    let mut items = Vec::new();
    let kinds = [
        (ItemKind::Health, rng.gen_range(2..=3)),
        (ItemKind::Medikit, rng.gen_range(0..=1)),
        (ItemKind::Ammo, rng.gen_range(2..=3)),
    ];
    for (kind, n) in kinds {
        for _ in 0..n {
            let k = rng.gen_range(1..rooms.len());
            if let Some((x, y)) = free_cell(rng, k, &mut used) {
                items.push((kind, x, y));
            }
        }
    }
    let name = format!("Generated {w}x{h}");
    Some(Level {
        map: finish_map(name, w, h, cells),
        start: (sx as f32 + 0.5, sy as f32 + 0.5, start_angle),
        monsters,
        items,
    })
}

// -------------------------------------------------------------------- questions

pub static QUESTIONS: &[QuestionSpec] = &[
    QuestionSpec {
        id: "action",
        qtype: QType::Choice,
        ins_en: &[
            "Which action should the player take now to survive, clear the level and reach the exit?",
            "Pick the player's next move in this first-person shooter.",
            "What should the player do right now?",
            "Choose the next action for the player in the maze full of monsters.",
        ],
        ins_fr: &[
            "Quelle action le joueur doit-il faire maintenant pour survivre et atteindre la sortie ?",
            "Choisis le prochain mouvement du joueur dans ce jeu de tir.",
            "Que doit faire le joueur tout de suite ?",
        ],
        options: &[
            OptionSpec {
                key: "forward",
                synonyms: &["advance", "walk", "avancer"],
                desc_en: &["move forward", "walk ahead", "step forward"],
                desc_fr: &["avancer", "marcher tout droit"],
            },
            OptionSpec {
                key: "turn_left",
                synonyms: &["rotate_left", "look_left", "gauche"],
                desc_en: &["turn to the left", "rotate left in place", "look left"],
                desc_fr: &["tourner à gauche", "pivoter vers la gauche"],
            },
            OptionSpec {
                key: "turn_right",
                synonyms: &["rotate_right", "look_right", "droite"],
                desc_en: &["turn to the right", "rotate right in place", "look right"],
                desc_fr: &["tourner à droite", "pivoter vers la droite"],
            },
            OptionSpec {
                key: "strafe_left",
                synonyms: &["sidestep_left", "slide_left", "pas_gauche"],
                desc_en: &[
                    "sidestep to the left",
                    "strafe left without turning",
                    "slide left",
                ],
                desc_fr: &["pas de côté à gauche", "glisser vers la gauche"],
            },
            OptionSpec {
                key: "strafe_right",
                synonyms: &["sidestep_right", "slide_right", "pas_droite"],
                desc_en: &[
                    "sidestep to the right",
                    "strafe right without turning",
                    "slide right",
                ],
                desc_fr: &["pas de côté à droite", "glisser vers la droite"],
            },
            OptionSpec {
                key: "shoot",
                synonyms: &["fire", "attack", "tirer"],
                desc_en: &[
                    "fire the weapon at the crosshair",
                    "shoot straight ahead",
                    "pull the trigger",
                ],
                desc_fr: &["tirer droit devant", "faire feu"],
            },
            OptionSpec {
                key: "use",
                synonyms: &["open", "press", "activer"],
                desc_en: &[
                    "open the door or press the switch ahead",
                    "use what is in front",
                    "activate the thing ahead",
                ],
                desc_fr: &[
                    "ouvrir la porte ou appuyer sur l'interrupteur",
                    "utiliser ce qui est devant",
                ],
            },
            OptionSpec {
                key: "back",
                synonyms: &["retreat", "backpedal", "reculer"],
                desc_en: &["step backward", "back away", "move backward"],
                desc_fr: &["reculer", "faire marche arrière"],
            },
        ],
    },
    QuestionSpec {
        id: "danger",
        qtype: QType::Noul,
        ins_en: &[
            "Will the player take damage in the next second if nothing changes?",
            "If the player stands still for one second, does it get hurt?",
            "The player is about to be hit within a second unless it reacts.",
        ],
        ins_fr: &[
            "Le joueur va-t-il subir des dégâts dans la seconde si rien ne change ?",
            "Si le joueur reste immobile une seconde, est-il touché ?",
        ],
        options: &[
            OptionSpec {
                key: "false",
                synonyms: &[],
                desc_en: &["", "the player stays unharmed", "no damage coming"],
                desc_fr: &["le joueur reste indemne", "aucun dégât"],
            },
            OptionSpec {
                key: "true",
                synonyms: &[],
                desc_en: &["", "the player gets hurt", "damage is coming"],
                desc_fr: &["le joueur est touché", "des dégâts arrivent"],
            },
        ],
    },
    QuestionSpec {
        id: "threat",
        qtype: QType::Score,
        ins_en: &[
            "How threatening is the current situation for the player?",
            "Rate the danger the player faces right now.",
            "Score the threat level around the player.",
        ],
        ins_fr: &[
            "À quel point la situation est-elle menaçante pour le joueur ?",
            "Évalue le danger que court le joueur en ce moment.",
        ],
        options: &[
            OptionSpec {
                key: "0",
                synonyms: &[],
                desc_en: &["no threat in sight", "all clear", "nothing hostile around"],
                desc_fr: &["aucune menace en vue", "rien d'hostile"],
            },
            OptionSpec {
                key: "1",
                synonyms: &[],
                desc_en: &["low: a distant enemy", "minor threat", "a far enemy"],
                desc_fr: &["faible : un ennemi lointain", "menace mineure"],
            },
            OptionSpec {
                key: "2",
                synonyms: &[],
                desc_en: &[
                    "moderate: an enemy is engaging",
                    "medium threat",
                    "one active enemy",
                ],
                desc_fr: &["modérée : un ennemi attaque", "menace moyenne"],
            },
            OptionSpec {
                key: "3",
                synonyms: &[],
                desc_en: &[
                    "high: several enemies or one up close",
                    "serious threat",
                    "close combat",
                ],
                desc_fr: &[
                    "élevée : plusieurs ennemis ou un ennemi proche",
                    "menace sérieuse",
                ],
            },
            OptionSpec {
                key: "4",
                synonyms: &[],
                desc_en: &[
                    "critical: overwhelmed or nearly dead",
                    "deadly threat",
                    "the player may die",
                ],
                desc_fr: &["critique : submergé ou presque mort", "menace mortelle"],
            },
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_maps_are_well_formed() {
        for (name, rows) in HAND_MAPS {
            for (fx, fy) in [(false, false), (true, false), (false, true), (true, true)] {
                let level = parse_map(name, rows, fx, fy);
                let m = &level.map;
                let exits = m.cells.iter().filter(|c| **c == Cell::Exit).count();
                assert_eq!(exits, 1, "{name}");
                let mut g = Doom::new(1);
                g.map = Arc::new(finish_map(m.name.clone(), m.w, m.h, m.cells.clone()));
                g.doors = g
                    .map
                    .cells
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c == Cell::Door)
                    .map(|(i, _)| DoorState {
                        cell: i,
                        open: 0.0,
                        opening: false,
                    })
                    .collect();
                let start = g.cell_of(level.start.0, level.start.1);
                assert_eq!(g.map.cells[start], Cell::Floor, "{name}");
                assert!(
                    !g.exit_goal_cells().is_empty(),
                    "{name}: exit has no floor in front"
                );
                assert!(
                    g.astar(start, &g.exit_goal_cells()).is_some(),
                    "{name}: exit unreachable"
                );
                // Doors sit in one-wide openings.
                for d in &g.doors {
                    let (x, y) = ((d.cell % m.w) as i32, (d.cell / m.w) as i32);
                    let solid = |dx: i32, dy: i32| {
                        !matches!(
                            g.cell_at(x + dx, y + dy),
                            Some(Cell::Floor) | Some(Cell::Door)
                        )
                    };
                    assert!(
                        (solid(1, 0) && solid(-1, 0)) || (solid(0, 1) && solid(0, -1)),
                        "{name}: door at {x},{y} is not in a one-wide opening"
                    );
                }
            }
        }
    }

    #[test]
    fn generated_levels_are_valid() {
        let mut rng = SmallRng::seed_from_u64(3);
        for _ in 0..40 {
            let level = generate_level(&mut rng, Skill::Easy);
            let m = &level.map;
            assert_eq!(m.cells.iter().filter(|c| **c == Cell::Exit).count(), 1);
            assert!(!level.monsters.is_empty());
        }
    }

    #[test]
    fn bearings_follow_the_conventions() {
        let mut g = Doom::new(2);
        g.player.angle = 0.0;
        let (x, y) = (g.player.x, g.player.y);
        assert!(
            g.bearing_to(x, y + 1.0) > 0.0,
            "south is on the right when facing east"
        );
        assert!(g.bearing_to(x, y - 1.0) < 0.0);
        g.player.angle = PI / 2.0;
        assert!(
            g.bearing_to(x - 1.0, y) > 0.0,
            "west is on the right when facing south"
        );
    }

    #[test]
    fn empty_weapon_is_vetoed() {
        let mut g = Doom::new(4);
        g.player.ammo = 0;
        let r = g.veto(DoomAction::Shoot);
        assert!(r.is_some_and(|a| a != DoomAction::Shoot), "{r:?}");
        g.player.ammo = 5;
        assert_eq!(g.veto(DoomAction::Shoot), None);
    }

    #[test]
    fn reflex_dodges_a_fireball() {
        let mut g = Doom::new(5);
        // A fireball 1.2 cells ahead, flying straight at the player.
        let (x, y, a) = (g.player.x, g.player.y, g.player.angle);
        g.projectiles.push(Projectile {
            id: 999,
            x: x + 1.2 * a.cos(),
            y: y + 1.2 * a.sin(),
            vx: -PROJ_SPEED * a.cos(),
            vy: -PROJ_SPEED * a.sin(),
            damage: 10,
        });
        let r = g.veto(DoomAction::TurnLeft);
        assert!(
            matches!(
                r,
                Some(DoomAction::StrafeLeft)
                    | Some(DoomAction::StrafeRight)
                    | Some(DoomAction::Back)
            ),
            "{r:?}"
        );
    }

    #[test]
    fn state_text_has_fourteen_pairs() {
        let g = Doom::new(6);
        let t = g.state_text();
        assert_eq!(t.split_whitespace().count(), 28, "{t}");
    }
}
