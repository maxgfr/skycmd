//! Structural validation of teacher-generated drone plans.
//!
//! Mirrors the `sky-schema` plan format: `{"plan": [{"action": "takeoff", "alt_m": 10}, ...]}`,
//! at most 6 actions, altitudes above home ≤ 120 m, speed ≤ 15 m/s. Validation is strict:
//! unknown actions, missing or extra fields, wrong types or out-of-range values reject the plan.

use anyhow::{anyhow, bail, Result};
use serde_json::{Map, Value};

pub const MAX_ACTIONS: usize = 6;
pub const MAX_ALT_M: f64 = 120.0;
pub const MAX_SPEED_MPS: f64 = 15.0;
pub const MAX_RANGE_M: f64 = 500.0;

/// Kind of a field value.
#[derive(Clone, Copy)]
enum F {
    /// Number within `[lo, hi]`; `open_lo` excludes `lo`.
    Num { lo: f64, hi: f64, open_lo: bool },
    Text,
}

const fn pos(hi: f64) -> F {
    F::Num { lo: 0.0, hi, open_lo: true }
}

const fn range(lo: f64, hi: f64) -> F {
    F::Num { lo, hi, open_lo: false }
}

/// Action names (the `sky-schema` tags) in a fixed order.
pub const ACTIONS: [&str; 17] = [
    "takeoff",
    "land",
    "hover",
    "goto",
    "move",
    "set_speed",
    "orbit",
    "survey_grid",
    "inspect",
    "take_photo",
    "start_video",
    "stop_video",
    "return_home",
    "end",
    "abort",
    "clarify",
    "refuse",
];

fn fields(action: &str) -> Option<Vec<(&'static str, F)>> {
    Some(match action {
        "takeoff" => vec![("alt_m", pos(MAX_ALT_M))],
        "hover" => vec![("s", pos(600.0))],
        "goto" => vec![("lat", range(-90.0, 90.0)), ("lon", range(-180.0, 180.0)), ("alt_m", pos(MAX_ALT_M))],
        "move" => vec![
            ("dx_m", range(-MAX_RANGE_M, MAX_RANGE_M)),
            ("dy_m", range(-MAX_RANGE_M, MAX_RANGE_M)),
            ("dz_m", range(-MAX_ALT_M, MAX_ALT_M)),
        ],
        "set_speed" => vec![("mps", pos(MAX_SPEED_MPS))],
        "orbit" => vec![("radius_m", pos(MAX_RANGE_M)), ("turns", pos(20.0))],
        "survey_grid" => vec![
            ("w_m", pos(2.0 * MAX_RANGE_M)),
            ("h_m", pos(2.0 * MAX_RANGE_M)),
            ("spacing_m", pos(200.0)),
            ("alt_m", pos(MAX_ALT_M)),
        ],
        "inspect" => vec![("target_desc", F::Text), ("distance_m", pos(100.0))],
        "clarify" => vec![("question", F::Text)],
        "refuse" => vec![("reason", F::Text)],
        "land" | "take_photo" | "start_video" | "stop_video" | "return_home" | "end" | "abort" => vec![],
        _ => return None,
    })
}

fn check_action(i: usize, a: &Map<String, Value>) -> Result<String> {
    let name = a
        .get("action")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("action {i}: missing `action` tag"))?;
    let spec = fields(name).ok_or_else(|| anyhow!("action {i}: unknown action {name}"))?;
    for k in a.keys() {
        if k != "action" && !spec.iter().any(|(f, _)| f == k) {
            bail!("action {i} ({name}): unexpected field {k}");
        }
    }
    for (f, kind) in spec {
        let v = a.get(f).ok_or_else(|| anyhow!("action {i} ({name}): missing {f}"))?;
        match kind {
            F::Text => {
                let s = v.as_str().ok_or_else(|| anyhow!("action {i} ({name}): {f} must be a string"))?;
                if s.trim().is_empty() {
                    bail!("action {i} ({name}): empty {f}");
                }
            }
            F::Num { lo, hi, open_lo } => {
                let x = v.as_f64().ok_or_else(|| anyhow!("action {i} ({name}): {f} must be a number"))?;
                let low_ok = if open_lo { x > lo } else { x >= lo };
                if !x.is_finite() || !low_ok || x > hi {
                    bail!("action {i} ({name}): {f}={x} out of range");
                }
            }
        }
    }
    Ok(name.to_string())
}

/// Validates a plan given either as `{"plan": [...]}` or as the bare action list.
/// Returns the action names.
pub fn validate_plan(v: &Value) -> Result<Vec<String>> {
    let list = match v {
        Value::Object(m) if m.len() == 1 => m.get("plan").and_then(Value::as_array),
        Value::Array(a) => Some(a),
        _ => None,
    }
    .ok_or_else(|| anyhow!("plan must be {{\"plan\": [...]}} or a list"))?;
    if list.is_empty() || list.len() > MAX_ACTIONS {
        bail!("plan must have 1..={MAX_ACTIONS} actions, got {}", list.len());
    }
    let names = list
        .iter()
        .enumerate()
        .map(|(i, a)| a.as_object().ok_or_else(|| anyhow!("action {i} is not an object")).and_then(|a| check_action(i, a)))
        .collect::<Result<Vec<_>>>()?;
    let terminal = names.iter().filter(|n| matches!(n.as_str(), "clarify" | "refuse")).count();
    if terminal > 0 && names.len() != 1 {
        bail!("clarify/refuse must be the only action of a plan");
    }
    Ok(names)
}

/// Checks that a plan is consistent with the order category produced by the teacher.
pub fn check_category(category: &str, names: &[String]) -> Result<()> {
    let first = names.first().map(String::as_str).unwrap_or("");
    match category {
        "unsafe" if first != "refuse" => bail!("unsafe order must be refused"),
        "ambiguous" if first != "clarify" => bail!("ambiguous order must ask for clarification"),
        "unsafe" | "ambiguous" => Ok(()),
        _ if matches!(first, "refuse" | "clarify") => bail!("{category} order must not be refused or clarified"),
        _ => Ok(()),
    }
}

/// The decision label of a plan: its first mission action, skipping a leading `takeoff`
/// (most plans start with one, so it carries no information) unless it is the only action.
pub fn first_mission_action(names: &[String]) -> Option<&str> {
    match names {
        [] => None,
        [only] => Some(only),
        [first, second, ..] if first == "takeoff" => Some(second),
        [first, ..] => Some(first),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn accepts_valid_plans() {
        let p = json!({"plan":[{"action":"takeoff","alt_m":30},{"action":"survey_grid","w_m":200,"h_m":100,"spacing_m":20,"alt_m":40},{"action":"return_home"}]});
        assert_eq!(validate_plan(&p).unwrap(), vec!["takeoff", "survey_grid", "return_home"]);
        assert!(validate_plan(&json!([{"action":"refuse","reason":"weapons are out of scope"}])).is_ok());
        assert!(validate_plan(&json!([{"action":"goto","lat":48.85,"lon":2.35,"alt_m":50}])).is_ok());
    }

    #[test]
    fn rejects_invalid_plans() {
        for bad in [
            json!([]),
            json!([{"action":"fly"}]),
            json!([{"action":"takeoff"}]),
            json!([{"action":"takeoff","alt_m":500}]),
            json!([{"action":"takeoff","alt_m":"30"}]),
            json!([{"action":"set_speed","mps":40}]),
            json!([{"action":"land","extra":1}]),
            json!([{"action":"inspect","target_desc":" ","distance_m":5}]),
            json!([{"action":"takeoff","alt_m":10},{"action":"refuse","reason":"x"}]),
            json!([{"action":"hover","s":1},{"action":"hover","s":1},{"action":"hover","s":1},{"action":"hover","s":1},{"action":"hover","s":1},{"action":"hover","s":1},{"action":"land"}]),
            json!({"steps":[]}),
        ] {
            assert!(validate_plan(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn categories_and_labels() {
        let refuse = vec!["refuse".to_string()];
        let normal = vec!["takeoff".to_string(), "inspect".to_string(), "land".to_string()];
        assert!(check_category("unsafe", &refuse).is_ok());
        assert!(check_category("unsafe", &normal).is_err());
        assert!(check_category("inspection", &refuse).is_err());
        assert!(check_category("mapping", &normal).is_ok());
        assert_eq!(first_mission_action(&normal), Some("inspect"));
        assert_eq!(first_mission_action(&refuse), Some("refuse"));
        assert_eq!(first_mission_action(&["takeoff".to_string()]), Some("takeoff"));
        assert_eq!(ACTIONS.len(), 17);
        assert!(ACTIONS.iter().all(|a| fields(a).is_some()));
    }
}
