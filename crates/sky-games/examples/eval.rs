//! Oracle vs random baselines on every game: `cargo run --release -p sky-games --example eval`.

use sky_games::{evaluate_with, EvalOptions, OraclePolicy, RandomPolicy, GAMES};

fn main() {
    let opts = EvalOptions {
        flappy_max_pipes: 200,
        veto: true,
    };
    for game in GAMES {
        let o = evaluate_with(game, &mut OraclePolicy::default(), 10, 1, opts);
        let r = evaluate_with(game, &mut RandomPolicy::new(1), 10, 1, opts);
        println!("{game:>7} oracle: {}", serde_json::to_string(&o).unwrap());
        println!("{game:>7} random: {}", serde_json::to_string(&r).unwrap());
    }
}
