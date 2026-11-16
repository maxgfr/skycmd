//! The fused tool router (s6-nano v3) natively: the two files are not in the repository
//! (`models/s6-nano2/skycmd-s6-route-nano-ternary.gguf` and
//! `models/s6-nano3/skycmd-s6-route-embed.gguf`, or `SKY_ROUTE_MODEL` / `SKY_ROUTE_EMBED`);
//! without them the test says so and passes. The parity with the Python reference on the
//! validation slates is `sky-route fuse` and `bench/route3-node.mjs`.

use std::path::PathBuf;

use sky_web::router::Router;

fn file(var: &str, rel: &str) -> Option<Vec<u8>> {
    let p = std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel)
    });
    let b = std::fs::read(&p).ok();
    if b.is_none() {
        eprintln!("router: {} not found: skipped", p.display());
    }
    b
}

#[test]
fn routes_with_the_tool_list_cached() {
    let (Some(m), Some(e)) = (
        file(
            "SKY_ROUTE_MODEL",
            "models/s6-nano2/skycmd-s6-route-nano-ternary.gguf",
        ),
        file(
            "SKY_ROUTE_EMBED",
            "models/s6-nano3/skycmd-s6-route-embed.gguf",
        ),
    ) else {
        return;
    };
    let mut r = Router::load(&m, &e).unwrap();
    let tools: Vec<String> = [
        "lights_on: Turns the lights on in a room.",
        "get_weather: Gives the weather forecast for a city.",
        "send_email: Sends an email to a contact.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let a = r
        .route("turn on the lights in the kitchen", &tools, false)
        .unwrap();
    assert_eq!(a.probs.len(), 4);
    assert!((a.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(a.answer.index, 0);
    assert!(!a.tools_hit);
    let b = r
        .route(
            "what is the weather forecast for Paris tomorrow?",
            &tools,
            false,
        )
        .unwrap();
    assert!(b.tools_hit && b.static_hit);
    assert_eq!(b.answer.index, 1);
    let f = r
        .route(
            "what is the weather forecast for Paris tomorrow?",
            &tools,
            true,
        )
        .unwrap();
    assert!(f.layers_used < b.layers_used);
    // the ask decision is the top probability against tau_ask3
    assert_eq!(b.answer.ask, b.answer.top < r.embedding().tau_ask);
    let v = r
        .route_value(
            "send a mail to Zoé",
            &serde_json::to_string(&tools).unwrap(),
            false,
        )
        .unwrap();
    assert_eq!(v["answer"], "send_email");
    assert!(v["decision"] == "tool" || v["decision"] == "ask");
}
