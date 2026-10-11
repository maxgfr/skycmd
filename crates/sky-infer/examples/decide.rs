//! Answer a System One request with a GGUF model, like `POST /v1/systemone`.
//!
//! ```text
//! cargo run --release -p sky-infer --example decide -- model.gguf request.json
//! ```
use sky_infer::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(model), Some(request)) = (args.next(), args.next()) else {
        eprintln!("usage: decide <model.gguf> <request.json>");
        std::process::exit(2);
    };
    let mut engine = Engine::from_gguf(&std::fs::read(model)?)?;
    let request = std::fs::read_to_string(request)?;
    println!("{}", engine.decide_json(&request)?);
    Ok(())
}
