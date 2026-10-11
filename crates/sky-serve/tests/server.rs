//! Spin the server on a random port with a random-init s1 GGUF and check the API shape; run the
//! `skycmd` binary on the same file.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;

use serde_json::Value;
use sky_infer::convert::{ladder_config, random_gguf};
use sky_infer::Tokenizer;
use sky_serve::{LoadedModel, ServeConfig, Server, SAMPLE_REQUEST};

fn random_model() -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sky-infer/tests/fixtures/small_tokenizer.json");
    let tok = Tokenizer::from_tokenizer_json(&std::fs::read_to_string(p).unwrap()).unwrap();
    let cfg = ladder_config("s1-pico").unwrap();
    random_gguf(&cfg, &tok, 11, sky_gguf::GgmlType::F32).unwrap()
}

/// One HTTP/1.1 exchange with `Connection: close`; returns (status, body).
fn http(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let body = body.unwrap_or("");
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    let (head, rest) = out.split_once("\r\n\r\n").unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    // tiny_http may answer with chunked encoding
    let body = if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        let mut b = String::new();
        let mut r = rest;
        loop {
            let (len, tail) = r.split_once("\r\n").unwrap();
            let n = usize::from_str_radix(len.trim(), 16).unwrap();
            if n == 0 {
                break;
            }
            b.push_str(&tail[..n]);
            r = &tail[n + 2..];
        }
        b
    } else {
        rest.to_string()
    };
    (status, body)
}

fn assert_prob(v: &Value) {
    let p = v.as_f64().unwrap();
    assert!((0.0..=1.0).contains(&p), "{p} is not a probability");
}

fn check_answers(resp: &Value) {
    assert_eq!(resp["model"], "s1-pico");
    let answers = resp["answers"].as_object().unwrap();
    let ids: Vec<&str> = answers.keys().map(String::as_str).collect();
    assert_eq!(
        ids,
        ["action", "danger", "margin"],
        "answers keep the question order"
    );

    let a = &answers["action"];
    assert_eq!(a["type"], "choice");
    let probs = a["probabilities"].as_object().unwrap();
    assert_eq!(probs.keys().collect::<Vec<_>>(), ["thrust", "glide"]);
    let sum: f64 = probs.values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert!(probs.contains_key(a["choice"].as_str().unwrap()));
    assert_prob(&a["confidence"]);

    let d = &answers["danger"];
    assert_eq!(d["type"], "noul");
    assert_prob(&d["noul"]);

    let m = &answers["margin"];
    assert_eq!(m["type"], "score");
    let score = m["score"].as_f64().unwrap();
    assert!((0.0..=2.0).contains(&score));
    assert_eq!(m["legend"]["0"], "none");
    assert_eq!(m["legend"]["2"], "comfortable");
    assert_eq!(m["probabilities"].as_object().unwrap().len(), 3);
    assert_prob(&m["confidence"]);

    assert!(resp["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(resp["usage"]["output_tokens"], 0);
}

#[test]
fn server_answers_systemone_health_and_models() {
    let (model, engine) = LoadedModel::from_bytes("random.gguf", random_model()).unwrap();
    let cfg = ServeConfig {
        addr: "127.0.0.1:0".into(),
        threads: 2,
        cors: true,
    };
    let server = Server::start(model, engine, &cfg).unwrap();
    let addr = server.addr();
    assert_ne!(addr.port(), 0);

    let (st, body) = http(addr, "GET", "/health", None);
    assert_eq!(st, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["status"],
        "ok"
    );

    let (st, body) = http(addr, "GET", "/v1/models", None);
    assert_eq!(st, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "s1-pico");
    assert_eq!(v["data"][0]["object"], "model");
    assert_eq!(v["data"][0]["meta"]["architecture"], "modern-bert");
    assert_eq!(v["data"][0]["meta"]["hidden_size"], 64);
    // llama.cpp fields
    assert_eq!(
        v["data"][0]["architecture"]["output_modalities"][0],
        "decisions"
    );
    assert_eq!(v["data"][0]["meta"]["n_embd"], 64);
    assert_eq!(v["data"][0]["meta"]["n_ctx_train"], 256);
    assert_eq!(v["data"][0]["aliases"][0], "random.gguf");

    let (st, body) = http(addr, "POST", "/v1/systemone", Some(SAMPLE_REQUEST));
    assert_eq!(st, 200, "{body}");
    let first: Value = serde_json::from_str(&body).unwrap();
    check_answers(&first);

    // same request on another worker: same answer
    for _ in 0..4 {
        let (_, body) = http(addr, "POST", "/v1/systemone", Some(SAMPLE_REQUEST));
        assert_eq!(serde_json::from_str::<Value>(&body).unwrap(), first);
    }

    // matches the library path exactly
    let (_, mut engine) = LoadedModel::from_bytes("random.gguf", random_model()).unwrap();
    let direct: Value =
        serde_json::from_str(&sky_serve::decide_json(&mut engine, SAMPLE_REQUEST).unwrap())
            .unwrap();
    assert_eq!(direct, first);

    // errors
    let (st, body) = http(addr, "POST", "/v1/systemone", Some(r#"{"questions": {}}"#));
    assert_eq!(st, 400);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], 400);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("state"));

    let (st, _) = http(addr, "POST", "/v1/systemone", Some("not json"));
    assert_eq!(st, 400);
    let (st, _) = http(addr, "GET", "/v1/systemone", None);
    assert_eq!(st, 405);
    let (st, _) = http(addr, "GET", "/nope", None);
    assert_eq!(st, 404);

    server.shutdown();
}

#[test]
fn bench_reports_latency() {
    let (_, mut engine) = LoadedModel::from_bytes("random.gguf", random_model()).unwrap();
    let r = sky_serve::bench(&mut engine, SAMPLE_REQUEST, 2, 10).unwrap();
    assert_eq!(r.iterations, 10);
    assert_eq!(r.questions, 3);
    assert!(r.min_us > 0.0 && r.min_us <= r.p50_us && r.p50_us <= r.p99_us);
}

#[test]
fn cli_decide_info_bench() {
    let dir = std::env::temp_dir().join(format!("skycmd-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("pico.gguf");
    std::fs::write(&model, random_model()).unwrap();
    let req = dir.join("req.json");
    std::fs::write(&req, SAMPLE_REQUEST).unwrap();
    let bin = env!("CARGO_BIN_EXE_skycmd");

    let out = std::process::Command::new(bin)
        .args(["decide", "--model"])
        .arg(&model)
        .arg("--request")
        .arg(&req)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    check_answers(&serde_json::from_slice(&out.stdout).unwrap());

    // request on stdin
    let mut child = std::process::Command::new(bin)
        .arg("decide")
        .arg(&model)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(SAMPLE_REQUEST.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    check_answers(&serde_json::from_slice(&out.stdout).unwrap());

    let out = std::process::Command::new(bin)
        .arg("info")
        .arg(&model)
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["architecture"], "modern-bert");
    assert!(v["n_params"].as_u64().unwrap() > 100_000);

    let out = std::process::Command::new(bin)
        .arg("bench")
        .arg(&model)
        .args(["--iterations", "5", "--warmup", "1", "--json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["iterations"], 5);

    let out = std::process::Command::new(bin)
        .args(["decide", "--bogus", "x"])
        .output()
        .unwrap();
    assert!(!out.status.success());

    std::fs::remove_dir_all(&dir).ok();
}

/// s2 (`skycmd-s2`) files go through the same CLI and server; the answers keep the s1 shape.
#[test]
fn server_and_cli_load_s2() {
    use sky_infer::s2::random::{ladder_config, random_gguf, test_tokenizer, WeightFormat};
    let cfg = ladder_config("s2-pico").unwrap();
    let bytes = random_gguf(&cfg, &test_tokenizer(), 5, WeightFormat::Ternary).unwrap();

    let (model, engine) = LoadedModel::from_bytes("s2.gguf", bytes.clone()).unwrap();
    assert_eq!(model.architecture, "skycmd-s2");
    let cfg_srv = ServeConfig {
        addr: "127.0.0.1:0".into(),
        threads: 1,
        cors: false,
    };
    let server = Server::start(model, engine, &cfg_srv).unwrap();
    let (st, body) = http(server.addr(), "POST", "/v1/systemone", Some(SAMPLE_REQUEST));
    assert_eq!(st, 200, "{body}");
    let resp: Value = serde_json::from_str(&body).unwrap();
    let answers = resp["answers"].as_object().unwrap();
    assert_eq!(
        answers.keys().map(String::as_str).collect::<Vec<_>>(),
        ["action", "danger", "margin"]
    );
    let probs = answers["action"]["probabilities"].as_object().unwrap();
    let sum: f64 = probs.values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert_prob(&answers["danger"]["noul"]);
    let (_, body) = http(server.addr(), "GET", "/v1/models", None);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["meta"]["architecture"], "skycmd-s2");
    server.shutdown();

    let dir = std::env::temp_dir().join(format!("skycmd-s2-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("s2.gguf");
    std::fs::write(&path, &bytes).unwrap();
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_skycmd"))
        .args(["decide", "--model"])
        .arg(&path)
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(SAMPLE_REQUEST.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let cli: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(cli["answers"], resp["answers"]);
    std::fs::remove_dir_all(&dir).ok();
}
