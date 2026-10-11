//! Drone command plans and the safety validator that sits between a plan decoder and the
//! flight controller.
//!
//! Frame convention for relative moves: `dx_m` east, `dy_m` north, `dz_m` up. Altitudes are
//! above the home point. The geofence is a circle around home.

use serde::{Deserialize, Serialize};

/// Maximum number of actions a plan may contain.
pub const MAX_ACTIONS: usize = 6;

/// A command plan: `{"plan": [{"action": "takeoff", "alt_m": 10}, ...]}`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub plan: Vec<Action>,
}

/// One plan step, tagged by `action`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    Takeoff {
        alt_m: f32,
    },
    Land,
    Hover {
        s: f32,
    },
    Goto {
        lat: f64,
        lon: f64,
        alt_m: f32,
    },
    Move {
        dx_m: f32,
        dy_m: f32,
        dz_m: f32,
    },
    SetSpeed {
        mps: f32,
    },
    Orbit {
        radius_m: f32,
        turns: f32,
    },
    SurveyGrid {
        w_m: f32,
        h_m: f32,
        spacing_m: f32,
        alt_m: f32,
    },
    Inspect {
        target_desc: String,
        distance_m: f32,
    },
    TakePhoto,
    StartVideo,
    StopVideo,
    ReturnHome,
    End,
    Abort,
    Clarify {
        question: String,
    },
    Refuse {
        reason: String,
    },
}

impl Action {
    /// The `action` tag.
    pub fn name(&self) -> &'static str {
        match self {
            Action::Takeoff { .. } => "takeoff",
            Action::Land => "land",
            Action::Hover { .. } => "hover",
            Action::Goto { .. } => "goto",
            Action::Move { .. } => "move",
            Action::SetSpeed { .. } => "set_speed",
            Action::Orbit { .. } => "orbit",
            Action::SurveyGrid { .. } => "survey_grid",
            Action::Inspect { .. } => "inspect",
            Action::TakePhoto => "take_photo",
            Action::StartVideo => "start_video",
            Action::StopVideo => "stop_video",
            Action::ReturnHome => "return_home",
            Action::End => "end",
            Action::Abort => "abort",
            Action::Clarify { .. } => "clarify",
            Action::Refuse { .. } => "refuse",
        }
    }

    fn numbers(&self) -> Vec<f64> {
        match self {
            Action::Takeoff { alt_m } => vec![*alt_m as f64],
            Action::Hover { s } => vec![*s as f64],
            Action::Goto { lat, lon, alt_m } => vec![*lat, *lon, *alt_m as f64],
            Action::Move { dx_m, dy_m, dz_m } => vec![*dx_m as f64, *dy_m as f64, *dz_m as f64],
            Action::SetSpeed { mps } => vec![*mps as f64],
            Action::Orbit { radius_m, turns } => vec![*radius_m as f64, *turns as f64],
            Action::SurveyGrid {
                w_m,
                h_m,
                spacing_m,
                alt_m,
            } => {
                vec![*w_m as f64, *h_m as f64, *spacing_m as f64, *alt_m as f64]
            }
            Action::Inspect { distance_m, .. } => vec![*distance_m as f64],
            _ => Vec::new(),
        }
    }
}

/// Hard limits enforced on every plan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SafetyLimits {
    pub max_alt_m: f32,
    pub max_speed_mps: f32,
    pub geofence_radius_m: f32,
    pub min_battery_pct: f32,
}

impl Default for SafetyLimits {
    fn default() -> Self {
        SafetyLimits {
            max_alt_m: 120.0,
            max_speed_mps: 15.0,
            geofence_radius_m: 500.0,
            min_battery_pct: 25.0,
        }
    }
}

/// Current vehicle state used to check a plan.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Telemetry {
    pub lat: f64,
    pub lon: f64,
    /// Altitude above home, meters.
    pub alt_m: f32,
    pub home_lat: f64,
    pub home_lon: f64,
    pub battery_pct: f32,
    pub airborne: bool,
}

impl Telemetry {
    /// A vehicle on the ground at its home point.
    pub fn at_home(lat: f64, lon: f64, battery_pct: f32) -> Self {
        Telemetry {
            lat,
            lon,
            alt_m: 0.0,
            home_lat: lat,
            home_lon: lon,
            battery_pct,
            airborne: false,
        }
    }
}

/// Result of [`validate`]: `corrected` is always safe to execute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// True when the plan needed no correction.
    pub ok: bool,
    pub violations: Vec<String>,
    pub corrected: Plan,
}

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// (east, north) offset in meters of `(lat, lon)` from `(lat0, lon0)` (equirectangular, fine at
/// geofence scale).
fn offset_m(lat0: f64, lon0: f64, lat: f64, lon: f64) -> (f64, f64) {
    let north = (lat - lat0).to_radians() * EARTH_RADIUS_M;
    let east = (lon - lon0).to_radians() * EARTH_RADIUS_M * lat0.to_radians().cos();
    (east, north)
}

/// Checks a plan against the limits and returns a corrected copy.
///
/// Rules: at most [`MAX_ACTIONS`] actions; non-finite numbers are dropped; altitudes are clamped
/// to `[0, max_alt_m]` and speeds to `[0, max_speed_mps]`; any step that would leave the
/// geofence is dropped; nothing runs after `end`/`abort`; below `min_battery_pct` the plan is
/// replaced by `return_home` (an explicit `land`/`abort`/`return_home` plan is kept).
pub fn validate(plan: &Plan, telemetry: &Telemetry, limits: &SafetyLimits) -> Verdict {
    let mut violations = Vec::new();

    if telemetry.battery_pct < limits.min_battery_pct {
        let already_safe = !plan.plan.is_empty()
            && plan
                .plan
                .iter()
                .all(|a| matches!(a, Action::Land | Action::Abort | Action::ReturnHome));
        if !already_safe {
            violations.push(format!(
                "battery {:.0}% below minimum {:.0}%: plan replaced by return_home",
                telemetry.battery_pct, limits.min_battery_pct
            ));
            return Verdict {
                ok: false,
                violations,
                corrected: Plan {
                    plan: vec![Action::ReturnHome],
                },
            };
        }
    }

    let fence = limits.geofence_radius_m as f64;
    let (mut east, mut north) = offset_m(
        telemetry.home_lat,
        telemetry.home_lon,
        telemetry.lat,
        telemetry.lon,
    );
    let mut alt = telemetry.alt_m;
    let clamp_alt = |a: f32, i: usize, violations: &mut Vec<String>| -> f32 {
        let c = a.clamp(0.0, limits.max_alt_m);
        if c != a {
            violations.push(format!(
                "action {i}: altitude {a} m clamped to {c} m (limit {} m)",
                limits.max_alt_m
            ));
        }
        c
    };
    let outside = |e: f64, n: f64| (e * e + n * n).sqrt() > fence;

    let mut corrected = Vec::new();
    for (i, action) in plan.plan.iter().enumerate() {
        if i >= MAX_ACTIONS {
            violations.push(format!(
                "plan has {} actions, truncated to {MAX_ACTIONS}",
                plan.plan.len()
            ));
            break;
        }
        if action.numbers().iter().any(|v| !v.is_finite()) {
            violations.push(format!(
                "action {i} ({}): non-finite value, dropped",
                action.name()
            ));
            continue;
        }
        let fixed = match action {
            Action::Takeoff { alt_m } => {
                let a = clamp_alt(*alt_m, i, &mut violations);
                alt = a;
                Action::Takeoff { alt_m: a }
            }
            Action::Hover { s } if *s < 0.0 => {
                violations.push(format!("action {i}: negative hover time clamped to 0 s"));
                Action::Hover { s: 0.0 }
            }
            Action::Goto { lat, lon, alt_m } => {
                let (e, n) = offset_m(telemetry.home_lat, telemetry.home_lon, *lat, *lon);
                if outside(e, n) {
                    violations.push(format!(
                        "action {i} (goto): target {:.0} m from home breaches the {fence:.0} m geofence, dropped",
                        (e * e + n * n).sqrt()
                    ));
                    continue;
                }
                east = e;
                north = n;
                let a = clamp_alt(*alt_m, i, &mut violations);
                alt = a;
                Action::Goto {
                    lat: *lat,
                    lon: *lon,
                    alt_m: a,
                }
            }
            Action::Move { dx_m, dy_m, dz_m } => {
                let (e, n) = (east + *dx_m as f64, north + *dy_m as f64);
                if outside(e, n) {
                    violations.push(format!(
                        "action {i} (move): destination breaches the {fence:.0} m geofence, dropped"
                    ));
                    continue;
                }
                east = e;
                north = n;
                let target = clamp_alt(alt + dz_m, i, &mut violations);
                let dz = target - alt;
                alt = target;
                Action::Move {
                    dx_m: *dx_m,
                    dy_m: *dy_m,
                    dz_m: dz,
                }
            }
            Action::SetSpeed { mps } => {
                let c = mps.clamp(0.0, limits.max_speed_mps);
                if c != *mps {
                    violations.push(format!(
                        "action {i}: speed {mps} m/s clamped to {c} m/s (limit {} m/s)",
                        limits.max_speed_mps
                    ));
                }
                Action::SetSpeed { mps: c }
            }
            Action::Orbit { radius_m, turns } => {
                let r = radius_m.abs() as f64;
                if (east * east + north * north).sqrt() + r > fence {
                    violations.push(format!(
                        "action {i} (orbit): radius {r:.0} m breaches the {fence:.0} m geofence, dropped"
                    ));
                    continue;
                }
                Action::Orbit {
                    radius_m: radius_m.abs(),
                    turns: *turns,
                }
            }
            Action::SurveyGrid {
                w_m,
                h_m,
                spacing_m,
                alt_m,
            } => {
                let half_diag = ((*w_m as f64).powi(2) + (*h_m as f64).powi(2)).sqrt() / 2.0;
                if (east * east + north * north).sqrt() + half_diag > fence {
                    violations.push(format!(
                        "action {i} (survey_grid): area breaches the {fence:.0} m geofence, dropped"
                    ));
                    continue;
                }
                let a = clamp_alt(*alt_m, i, &mut violations);
                alt = a;
                Action::SurveyGrid {
                    w_m: w_m.abs(),
                    h_m: h_m.abs(),
                    spacing_m: spacing_m.abs(),
                    alt_m: a,
                }
            }
            Action::ReturnHome => {
                east = 0.0;
                north = 0.0;
                action.clone()
            }
            Action::Land => {
                alt = 0.0;
                action.clone()
            }
            other => other.clone(),
        };
        let terminal = matches!(fixed, Action::End | Action::Abort);
        corrected.push(fixed);
        if terminal && i + 1 < plan.plan.len() {
            violations.push(format!(
                "actions after {} (index {i}) dropped",
                plan.plan[i].name()
            ));
            break;
        }
    }

    Verdict {
        ok: violations.is_empty(),
        violations,
        corrected: Plan { plan: corrected },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOME: (f64, f64) = (48.8566, 2.3522);

    fn tele(battery: f32) -> Telemetry {
        Telemetry::at_home(HOME.0, HOME.1, battery)
    }

    #[test]
    fn json_shape() {
        let json = r#"{"plan":[{"action":"takeoff","alt_m":10.0},{"action":"survey_grid","w_m":40.0,"h_m":20.0,"spacing_m":5.0,"alt_m":30.0},{"action":"inspect","target_desc":"north wall of the barn","distance_m":8.0},{"action":"take_photo"},{"action":"return_home"},{"action":"end"}]}"#;
        let plan: Plan = serde_json::from_str(json).unwrap();
        assert_eq!(plan.plan.len(), 6);
        assert_eq!(plan.plan[3], Action::TakePhoto);
        assert_eq!(serde_json::to_string(&plan).unwrap(), json);
        let v = validate(&plan, &tele(80.0), &SafetyLimits::default());
        assert!(v.ok, "{:?}", v.violations);
        assert_eq!(v.corrected, plan);
    }

    #[test]
    fn clamps_altitude_and_speed() {
        let plan = Plan {
            plan: vec![
                Action::Takeoff { alt_m: 300.0 },
                Action::SetSpeed { mps: 40.0 },
                Action::Move {
                    dx_m: 10.0,
                    dy_m: 0.0,
                    dz_m: 50.0,
                },
            ],
        };
        let v = validate(&plan, &tele(90.0), &SafetyLimits::default());
        assert!(!v.ok);
        // takeoff altitude, speed, and the climb of the move are all clamped.
        assert_eq!(v.violations.len(), 3);
        assert_eq!(
            v.corrected.plan,
            vec![
                Action::Takeoff { alt_m: 120.0 },
                Action::SetSpeed { mps: 15.0 },
                Action::Move {
                    dx_m: 10.0,
                    dy_m: 0.0,
                    dz_m: 0.0
                },
            ]
        );
    }

    #[test]
    fn rejects_geofence_breach() {
        // ~1.1 km north of home.
        let plan = Plan {
            plan: vec![
                Action::Takeoff { alt_m: 20.0 },
                Action::Goto {
                    lat: HOME.0 + 0.01,
                    lon: HOME.1,
                    alt_m: 20.0,
                },
                Action::Move {
                    dx_m: 0.0,
                    dy_m: 450.0,
                    dz_m: 0.0,
                },
                Action::Move {
                    dx_m: 0.0,
                    dy_m: 100.0,
                    dz_m: 0.0,
                },
                Action::Orbit {
                    radius_m: 30.0,
                    turns: 1.0,
                },
                Action::Land,
            ],
        };
        let v = validate(&plan, &tele(90.0), &SafetyLimits::default());
        assert!(!v.ok);
        let names: Vec<&str> = v.corrected.plan.iter().map(Action::name).collect();
        // goto dropped; first move ok (450 m); second move (550 m) dropped; orbit (480 m) ok.
        assert_eq!(names, ["takeoff", "move", "orbit", "land"]);
        assert_eq!(v.violations.len(), 2);
    }

    #[test]
    fn low_battery_forces_return_home() {
        let plan = Plan {
            plan: vec![Action::Takeoff { alt_m: 20.0 }, Action::TakePhoto],
        };
        let v = validate(&plan, &tele(12.0), &SafetyLimits::default());
        assert!(!v.ok);
        assert_eq!(v.corrected.plan, vec![Action::ReturnHome]);
        let land = Plan {
            plan: vec![Action::Land],
        };
        assert!(validate(&land, &tele(12.0), &SafetyLimits::default()).ok);
    }

    #[test]
    fn truncates_long_plans_and_stops_after_end() {
        let plan = Plan {
            plan: vec![Action::TakePhoto; 8],
        };
        let v = validate(&plan, &tele(90.0), &SafetyLimits::default());
        assert_eq!(v.corrected.plan.len(), MAX_ACTIONS);
        assert!(!v.ok);
        let plan = Plan {
            plan: vec![Action::End, Action::TakePhoto],
        };
        let v = validate(&plan, &tele(90.0), &SafetyLimits::default());
        assert_eq!(v.corrected.plan, vec![Action::End]);
    }

    #[test]
    fn drops_non_finite_values() {
        let plan = Plan {
            plan: vec![Action::Hover { s: f32::NAN }, Action::Land],
        };
        let v = validate(&plan, &tele(90.0), &SafetyLimits::default());
        assert_eq!(v.corrected.plan, vec![Action::Land]);
        assert!(!v.ok);
    }
}
