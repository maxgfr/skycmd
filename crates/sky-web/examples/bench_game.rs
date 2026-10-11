//! Native counterpart of `benchGame()` in the browser: µs per AI decision of a game (every
//! question of the game on one state + the physics), with the same Rust code as the WASM build.
//!
//! `cargo run --release -p sky-web --example bench_game -- model.gguf [game] [n] [tau]`

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .ok_or("usage: bench_game model.gguf [game] [n] [tau]")?;
    let game = args.get(1).map_or("doom", |s| s.as_str());
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let mut model = sky_web::Model::load(&bytes)?;
    if let Some(tau) = args.get(3).and_then(|v| v.parse().ok()) {
        model.set_exit_threshold(Some(tau));
    }
    sky_web::session::bench_game(&mut model, game, 200)?;
    let r = sky_web::session::bench_game(&mut model, game, n)?;
    println!("{r}");
    Ok(())
}
