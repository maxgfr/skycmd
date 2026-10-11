//! `sky-serve`: the `skycmd` command-line tool and its HTTP server.
//!
//! The server speaks the same API as llama.cpp's decision server:
//!
//! - `POST /v1/systemone`: a System One request (`{"state": ..., "questions": {...}}`), answered
//!   with `{"model", "answers", "usage"}`;
//! - `GET /health`: `{"status": "ok"}`;
//! - `GET /v1/models`: an OpenAI-style model list with the model description in `meta`.
//!
//! It runs one [`sky_infer::Engine`] per worker thread (the engine reuses its buffers and takes
//! `&mut self`), so requests never wait on a lock.

use std::io::Read;
use std::net::SocketAddr;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use sky_infer::Engine;

/// Largest accepted request body.
pub const MAX_BODY_BYTES: u64 = 8 << 20;

/// A request that exercises the three question types (a Flappy Drone frame), used by `bench`.
pub const SAMPLE_REQUEST: &str = r#"{
  "state": "vy 0 dx 36 up 9 dn 7 nxt 10 alt 32",
  "questions": {
    "action": {
      "type": "choice",
      "instructions": "Which input keeps the drone flying through the next gap?",
      "criteria": {
        "thrust": "fire the rotors for an upward impulse",
        "glide": "let gravity pull the drone down"
      }
    },
    "danger": {
      "type": "noul",
      "instructions": "Will the drone crash within the next half second if it keeps gliding?"
    },
    "margin": {
      "type": "score",
      "instructions": "How much room is left above and below the drone?",
      "criteria": ["none", "tight", "comfortable"]
    }
  }
}"#;

/// Errors of this crate.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Model(sky_infer::Error),
    Other(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Model(e) => write!(f, "{e}"),
            Error::Other(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<sky_infer::Error> for Error {
    fn from(e: sky_infer::Error) -> Self {
        Error::Model(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// A GGUF file read into memory, with the engine it builds.
pub struct LoadedModel {
    /// Display name: `general.name`, else the file stem.
    pub name: String,
    pub path: String,
    pub bytes: Arc<Vec<u8>>,
    pub architecture: String,
}

impl LoadedModel {
    /// Read a GGUF file and check that the engine can load it.
    pub fn open(path: &str) -> Result<(Self, Engine)> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Other(format!("cannot read model {path:?}: {e}")))?;
        Self::from_bytes(path, bytes)
    }

    pub fn from_bytes(path: &str, bytes: Vec<u8>) -> Result<(Self, Engine)> {
        let architecture = sky_gguf::Gguf::parse(&bytes)
            .map_err(sky_infer::Error::from)?
            .architecture()
            .unwrap_or("unknown")
            .to_string();
        let mut engine = Engine::from_gguf(&bytes)?;
        let mut name = engine.model_info().name;
        if name.is_empty() {
            name = std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".into());
        }
        let model = LoadedModel {
            name,
            path: path.to_string(),
            bytes: Arc::new(bytes),
            architecture,
        };
        Ok((model, engine))
    }

    /// Model description: the engine's `model_info` plus file details.
    pub fn info(&self, engine: &mut Engine) -> Value {
        let mut v: Value =
            serde_json::from_str(&engine.model_info().to_json()).unwrap_or_else(|_| json!({}));
        if let Value::Object(m) = &mut v {
            m.insert("architecture".into(), json!(self.architecture));
            m.insert("file".into(), json!(self.path));
            m.insert("bytes".into(), json!(self.bytes.len()));
        }
        v
    }
}

/// Answer a request given as JSON text: the shared code path of `decide`, the server and tests.
pub fn decide_json(engine: &mut Engine, request: &str) -> Result<String> {
    Ok(engine.decide_json(request)?)
}

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServeConfig {
    /// Address to bind, e.g. `127.0.0.1:8080` (port 0 picks a free port).
    pub addr: String,
    /// Worker threads, each with its own engine.
    pub threads: usize,
    /// Add `Access-Control-Allow-Origin: *` so browsers can call the server.
    pub cors: bool,
}

impl Default for ServeConfig {
    fn default() -> Self {
        ServeConfig {
            addr: "127.0.0.1:8080".into(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get().min(4))
                .unwrap_or(1),
            cors: true,
        }
    }
}

/// A running server.
pub struct Server {
    http: Arc<tiny_http::Server>,
    addr: SocketAddr,
    workers: Vec<JoinHandle<()>>,
}

impl Server {
    /// Bind the address and start the workers. `engine` is reused by the first worker; the
    /// others build their own from the model bytes.
    pub fn start(model: LoadedModel, mut engine: Engine, cfg: &ServeConfig) -> Result<Server> {
        let info = model.info(&mut engine);
        let http = tiny_http::Server::http(&cfg.addr)
            .map_err(|e| Error::Other(format!("cannot listen on {}: {e}", cfg.addr)))?;
        let addr = http
            .server_addr()
            .to_ip()
            .ok_or_else(|| Error::Other("not an IP listener".into()))?;
        let http = Arc::new(http);
        let mut engines = vec![engine];
        for _ in 1..cfg.threads.max(1) {
            engines.push(Engine::from_gguf(&model.bytes)?);
        }
        let shared = Arc::new(Shared {
            info,
            model,
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            cors: cfg.cors,
        });
        let workers = engines
            .into_iter()
            .map(|mut engine| {
                let http = Arc::clone(&http);
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    while let Ok(req) = http.recv() {
                        handle(&shared, &mut engine, req);
                    }
                })
            })
            .collect();
        Ok(Server {
            http,
            addr,
            workers,
        })
    }

    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Block until the server stops.
    pub fn join(self) {
        for w in self.workers {
            let _ = w.join();
        }
    }

    /// Stop accepting requests and wait for the workers.
    pub fn shutdown(self) {
        for _ in 0..self.workers.len() {
            self.http.unblock();
        }
        self.join();
    }
}

struct Shared {
    model: LoadedModel,
    info: Value,
    created: u64,
    cors: bool,
}

fn error_body(code: u16, message: &str) -> Value {
    let kind = match code {
        400 => "invalid_request_error",
        404 => "not_found_error",
        405 => "invalid_request_error",
        413 => "invalid_request_error",
        _ => "server_error",
    };
    json!({"error": {"code": code, "message": message, "type": kind}})
}

fn handle(shared: &Shared, engine: &mut Engine, mut req: tiny_http::Request) {
    let method = req.method().clone();
    let path = req.url().split('?').next().unwrap_or("").to_string();
    let (status, body) = match (&method, path.as_str()) {
        (tiny_http::Method::Options, _) => (204, Value::Null),
        (tiny_http::Method::Get, "/health" | "/v1/health") => (200, json!({"status": "ok"})),
        (tiny_http::Method::Get, "/v1/models" | "/models") => (200, models(shared)),
        (tiny_http::Method::Get, "/") => (
            200,
            json!({
                "name": "skycmd",
                "version": env!("CARGO_PKG_VERSION"),
                "model": shared.model.name,
                "endpoints": ["POST /v1/systemone", "GET /health", "GET /v1/models"]
            }),
        ),
        (tiny_http::Method::Post, "/v1/systemone" | "/systemone") => systemone(engine, &mut req),
        (_, "/v1/systemone" | "/systemone" | "/health" | "/v1/models") => {
            (405, error_body(405, "Method Not Allowed"))
        }
        _ => (404, error_body(404, "File Not Found")),
    };
    let text = if body.is_null() {
        String::new()
    } else {
        body.to_string()
    };
    let mut resp = tiny_http::Response::from_string(text).with_status_code(status);
    let mut headers = vec![("Content-Type", "application/json; charset=utf-8")];
    if shared.cors {
        headers.extend([
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            (
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization",
            ),
        ]);
    }
    for (k, v) in headers {
        if let Ok(h) = tiny_http::Header::from_bytes(k.as_bytes(), v.as_bytes()) {
            resp.add_header(h);
        }
    }
    let _ = req.respond(resp);
}

/// `GET /v1/models`: OpenAI-style list, with the fields llama.cpp adds for decision models
/// (`architecture` modalities, `meta.n_vocab` / `n_ctx_train` / `n_embd` / `n_params` / `size`)
/// next to the skycmd model description.
fn models(shared: &Shared) -> Value {
    let mut meta = shared.info.clone();
    if let Value::Object(m) = &mut meta {
        let alias = |from: &str| m.get(from).cloned().unwrap_or(Value::Null);
        let extra = [
            ("n_vocab", alias("vocab_size")),
            ("n_ctx_train", alias("max_len")),
            ("n_embd", alias("hidden_size")),
            ("size", alias("bytes")),
        ];
        for (k, v) in extra {
            m.entry(k).or_insert(v);
        }
    }
    json!({
        "object": "list",
        "data": [{
            "id": shared.model.name,
            "aliases": [shared.model.path],
            "object": "model",
            "created": shared.created,
            "owned_by": "skycmd",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]},
            "meta": meta,
        }]
    })
}

fn systemone(engine: &mut Engine, req: &mut tiny_http::Request) -> (u16, Value) {
    let mut body = String::new();
    if let Err(e) = req
        .as_reader()
        .take(MAX_BODY_BYTES + 1)
        .read_to_string(&mut body)
    {
        return (400, error_body(400, &format!("cannot read the body: {e}")));
    }
    if body.len() as u64 > MAX_BODY_BYTES {
        return (413, error_body(413, "request body too large"));
    }
    match engine.decide_json(&body) {
        Ok(out) => match serde_json::from_str::<Value>(&out) {
            Ok(v) => (200, v),
            Err(e) => (500, error_body(500, &e.to_string())),
        },
        Err(e @ sky_infer::Error::InvalidRequest(_)) => (400, error_body(400, &e.to_string())),
        Err(e) => (500, error_body(500, &e.to_string())),
    }
}

/// Latency summary of `bench`.
#[derive(Clone, Debug)]
pub struct BenchReport {
    pub iterations: usize,
    pub questions: usize,
    pub input_tokens: u64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
}

impl BenchReport {
    pub fn requests_per_s(&self) -> f64 {
        1e6 / self.mean_us
    }

    pub fn to_json(&self) -> Value {
        json!({
            "iterations": self.iterations,
            "questions_per_request": self.questions,
            "input_tokens_per_request": self.input_tokens,
            "mean_us": self.mean_us,
            "p50_us": self.p50_us,
            "p90_us": self.p90_us,
            "p99_us": self.p99_us,
            "min_us": self.min_us,
            "requests_per_s": self.requests_per_s(),
            "questions_per_s": self.requests_per_s() * self.questions as f64,
        })
    }
}

/// Time `iterations` requests after `warmup` untimed ones.
pub fn bench(
    engine: &mut Engine,
    request: &str,
    warmup: usize,
    iterations: usize,
) -> Result<BenchReport> {
    let parsed = sky_infer::DecisionRequest::from_json(request)?;
    let first = engine.decide(&parsed)?;
    for _ in 0..warmup {
        engine.decide(&parsed)?;
    }
    let iterations = iterations.max(1);
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let t0 = Instant::now();
        let r = engine.decide(&parsed)?;
        times.push(t0.elapsed().as_secs_f64() * 1e6);
        std::hint::black_box(r);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| times[((times.len() - 1) as f64 * p).round() as usize];
    Ok(BenchReport {
        iterations,
        questions: parsed.questions.len(),
        input_tokens: first.input_tokens as u64,
        mean_us: times.iter().sum::<f64>() / times.len() as f64,
        p50_us: pct(0.5),
        p90_us: pct(0.9),
        p99_us: pct(0.99),
        min_us: times[0],
    })
}
