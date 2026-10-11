//! Oracle and random play on the Doom-like game, per episode:
//! `cargo run --release -p sky-games --example doom_eval -- [episodes] [seed] [easy|normal] [-v]`.

use sky_games::common::{argmax, DecisionGame};
use sky_games::doom::{Doom, Skill};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let episodes: usize = args.first().and_then(|v| v.parse().ok()).unwrap_or(20);
    let seed: u64 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2026);
    let skill = if args.iter().any(|a| a == "normal") {
        Skill::Normal
    } else {
        Skill::Easy
    };
    let verbose = args.iter().any(|a| a == "-v");
    // DOOM_TRACE=<episode>: print the state, position and oracle action every 10 decisions, and the map at tick 400.
    let trace: Option<usize> = std::env::var("DOOM_TRACE")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut g = Doom::with_skill(seed, skill);
    let (mut done, mut deaths, mut kills, mut monsters, mut ticks, mut vetoes) =
        (0, 0, 0, 0, 0u64, 0);
    let (mut decisions, t0) = (0u64, Instant::now());
    // Time spent in the game itself (reflex + physics), without the oracle.
    let mut physics = std::time::Duration::ZERO;
    for ep in 0..episodes {
        if ep > 0 {
            g.reset();
        }
        while !g.done() {
            if g.decision_due() {
                decisions += 1;
                let t = g.oracle();
                if trace == Some(ep) && g.tick.is_multiple_of(40) {
                    let nav = g.nav();
                    println!(
                        "t {:5} pos {:.2},{:.2} a {:4.0} nav {:.1},{:.1} {:?} | {} -> {:?}",
                        g.tick,
                        g.player.x,
                        g.player.y,
                        g.player.angle.to_degrees(),
                        nav.x,
                        nav.y,
                        nav.goal,
                        g.state_text(),
                        sky_games::DoomAction::from_index(argmax(&t[0]))
                    );
                }
                if trace == Some(ep) && g.tick == 400 {
                    let snap = g.snapshot();
                    for y in 0..snap.height {
                        let row: String = (0..snap.width)
                            .map(|x| {
                                if x == g.player.x as usize && y == g.player.y as usize {
                                    'P'
                                } else {
                                    snap.cells.as_bytes()[y * snap.width + x] as char
                                }
                            })
                            .collect();
                        println!("{row}");
                    }
                }
                let tp = Instant::now();
                g.apply(argmax(&t[0]), false);
                physics += tp.elapsed();
            } else {
                let tp = Instant::now();
                g.tick();
                physics += tp.elapsed();
            }
        }
        let s = g.stats();
        done += s.success as usize;
        deaths += s.crashed as usize;
        kills += s.score as u32;
        monsters += s.max_score as u32;
        ticks += s.steps as u64;
        vetoes += s.vetoes;
        if verbose || !s.success {
            println!(
                "ep {ep:3} {:22} done {} dead {} kills {}/{} ticks {} hp {} ammo {} vetoes {}",
                g.map_name(),
                s.success,
                s.crashed,
                s.score,
                s.max_score,
                s.steps,
                g.player.health,
                g.player.ammo,
                s.vetoes
            );
        }
    }
    let n = episodes as f64;
    println!(
        "oracle {skill:?}: completion {:.2} deaths {deaths} kill rate {:.2} mean ticks {:.0} vetoes {vetoes} | {:.0} µs per decision, of which {:.1} µs game (reflex + physics)",
        done as f64 / n,
        kills as f64 / monsters.max(1) as f64,
        ticks as f64 / n,
        t0.elapsed().as_secs_f64() * 1e6 / decisions.max(1) as f64,
        physics.as_secs_f64() * 1e6 / decisions.max(1) as f64
    );
}
