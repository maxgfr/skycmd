# 18 A Doom-like shooter

The first three games are 2D and small: one drone, a handful of numbers. This chapter adds a game with a first-person view, monsters that hunt the player, doors, pickups and an exit switch, and asks the same kind of model to play it: a text state plus typed questions in, calibrated probabilities out, one forward pass per question. It is not a reinforcement-learning policy network. The model of this chapter is s2-nano ([chapter 19](./19-s2-architecture)), fine-tuned so it plays the three earlier games too. The later s3-pico ([chapter 21](./21-s3-recursive-tiny)) plays all four games with one file, and the Play page picks it for Doom when it is published (see the end of this chapter).

[Play it](/play/?game=doom). The tab is labelled "Doom-like (Freedoom assets)": the code, rules and levels are ours, the art comes from [Freedoom](https://github.com/freedoom/freedoom) (BSD-3-Clause), and nothing comes from id Software's Doom.

## The game

`crates/sky-games/src/doom.rs` is a pure, seedable Rust module like the others: no I/O, no clock, no threads, so the browser, `sky-arena` and the data generator all run the same code.

- **Grid levels.** A level is a grid of cells: floor, walls with five textures, doors and one exit switch (a wall cell that ends the level when used). 30% of the episodes draw one of four hand-designed levels (`HAND_MAPS`), mirrored at random; the others are generated from the episode seed: 5 to 7 rooms chained by corridors, doors where a one-wide corridor enters a room, the exit switch on the wall of the room farthest from the start, monsters at least 6 cells from the start, stimpacks, medikits and boxes of shells.
- **Doom's clock.** Physics runs at Doom's 35 Hz tick. The player moves 0.11 cell per tick, strafes 0.10, backs off 0.08 and turns 4 degrees per tick. Collisions push the player's circle out of the walls, so it slides around corners like in Doom.
- **The weapon** is hitscan along the facing direction, with an 18-tick cooldown and an aim cone of 9 degrees (Doom's autoaim, horizontal here). 25 to 45 damage per hit.
- **Monsters.** The imp-like monster (60 HP) claws up close and throws fireballs that fly at 0.17 cell per tick, slow enough to sidestep. The zombieman-like monster (20 HP) fires a hitscan rifle whose accuracy drops with distance. Monsters wake when they see the player in front of them, or when they hear a shot within 12 cells of open path. They chase by sight, or along a breadth-first flow field when they lose sight, and they attack with cooldowns and a short wind-up. Hits stun them (the pain state).
- **The episode** ends at the exit switch, at death, or after 150 seconds. The score is the kills (100 points each), the items (25), the exit (1000) and a time bonus.
- **Skill.** `Skill::Easy` is the game everything trains and evaluates on (`make_game("doom")`). `Skill::Normal` hits harder, attacks more often and puts more monsters in generated levels.

`snapshot()` hands the renderer everything it needs: the cell string, door openings, the player pose, every sprite with kind, state, animation tick and visibility, the waypoint, kills, items, the last shot and the last hit.

## A decision every 4 ticks

Doom players react at roughly 10 Hz. The model decides every 4 ticks (8.75 decisions per second) and the action is held in between, exactly like the 6-frame cadence of Obstacle Dodge. Each decision asks three questions about the same state:

| id | type | options |
|---|---|---|
| `action` | choice | `forward`, `turn_left`, `turn_right`, `strafe_left`, `strafe_right`, `shoot`, `use`, `back` |
| `danger` | noul | "Will the player take damage in the next second if nothing changes?" |
| `threat` | score | 5 levels, from "no threat in sight" to "critical: overwhelmed or nearly dead" |

Each question has EN and FR paraphrases, and every choice key has synonyms (`fire`, `tirer`, `sidestep_left`, ...), like the other games. On s2, the state is encoded once per decision and the two extra questions only cost their fusion head.

## The state: the code measures, the model decides

A first-person view is far too big for a 64-token state, and pixels would turn the task into vision. The game measures what a player perceives and writes 14 small integers:

```text
ea -3 ed 6 al 0 en 1 hp 8 am 17 wf 12 wl 2 wr 5 dr 0 gb 1 gd 4 pb 0 pd 0
```

| key | meaning |
|---|---|
| `ea` `ed` | nearest visible enemy (line of sight, within 14 cells): bearing in 10-degree bins (positive = right) and distance in cells (0 = none) |
| `al` | 1 when that enemy is inside the aim cone: a shot now would hit it |
| `en` | number of visible enemies |
| `hp` `am` | health in tens, ammo |
| `wf` `wl` `wr` | free distance ahead, left and right, in quarter cells (up to 3 cells) |
| `dr` | what `use` would reach: 0 nothing, 1 a closed door, 2 the exit switch |
| `gb` `gd` | bearing and distance of the next waypoint toward the goal |
| `pb` `pd` | the incoming fireball that hits first: bearing and distance in half cells (0 = none) |

The waypoint comes from an A* path computed in code (8-connected, no corner cutting, closed doors cost extra), string-pulled to the farthest path cell the player can walk to in a straight line, and stopped at the first closed door. The goal is the exit switch, or the nearest stimpack or medikit at 40 health or less, or ammo at 2 shells or less. Only "incoming" projectiles count: those whose closest approach passes within the player's radius.

These are 28 pieces, above the 24 of the 2D games: `state_text_fits_token_budget` allows 30 for doom (and still checks the key/value pairs and the value range). With s2's `[NUM]` tokens this is about 38 tokens (`BENCH.md`), more than half of the 64-token state budget; s3's value-fused tokens bring it down to 26.

## The oracle

`Doom::oracle()` has the whole simulator. Its policy is written on the measured features, so a model that sees the state text can imitate it:

1. facing the exit switch (`dr 2`): `use`;
2. an enemy visible and ammo left: `shoot` when aligned, else turn toward it; the soft target mixes `shoot` and the turn with a sigmoid of the margin to the aim cone;
3. a closed door ahead on the route (`dr 1`, waypoint within 35 degrees): `use`;
4. otherwise follow the waypoint: turn when it is more than about 22 degrees off, else `forward`, mixed with a sigmoid around the threshold.

Then a short lookahead: when a fireball is in the air, the oracle clones the game, holds its chosen action for 16 ticks with the monsters frozen, and if a fireball already in flight would hit, it moves the target mass to the dodges (`strafe_left`, `strafe_right`, `back`, `forward`) that stay clear, 0.85 on the one with the largest clearance.

`danger` is a lookahead too: the clone stands still for 45 ticks with the monsters acting normally (same RNG state), and the target is a sigmoid of the first tick with damage around 35 ticks. `threat` sums the awake visible monsters (imp 1.0, zombieman 0.8, scaled by distance), adds 0.8 for an incoming fireball, amplifies by low health, and turns the value into a soft distribution over the 5 levels.

The safety reflex (`Doom::veto`) is coded, as in Rescue Run and Obstacle Dodge: `shoot` with an empty weapon becomes the oracle's navigation action, and any action that would walk into, or stand in, the path of a known fireball within 12 ticks becomes the first free dodge.

Gates, in `crates/sky-games/tests/games.rs` (release builds):

| policy | episodes | completion | deaths | kill rate | mean ticks |
|---|---:|---:|---:|---:|---:|
| oracle, easy (seed 2026) | 100 | 1.00 | 0 | 0.69 | 464 |
| oracle, easy (seed 7) | 100 | 1.00 | 0 | 0.76 | 490 |
| oracle, normal (seed 7) | 50 | 1.00 | 0 | 0.80 | 568 |
| random, easy | 100 | 0.00 | | | 5250 (timeout) |

The test requires at least 90% completion with no death for the oracle and 0% for random play. `cargo run --release -p sky-games --example doom_eval -- 100 2026 [normal] [-v]` prints the episodes one by one (`DOOM_TRACE=<episode>` adds the state, the waypoint and the map).

## Data

`sky-games-gen --game doom --n 100000 --seed 1` writes `data/games/doom.jsonl` with the same machinery as the other games (paraphrases, renamed and shuffled options, 30% French, class balancing). With three questions, the 30% secondary share is split between `danger` and `threat`. Label mix of the 100k records: `forward` 24.8%, turns 33.8%, `shoot` 14.6%, `use` 9.6%, strafes and `back` 17.3%; `danger` true 31%; `threat` 47 / 20 / 21 / 12 / 1% over levels 0 to 4.

Hugging Face was searched for Doom data that could help: about 70 Doom and ViZDoom datasets, from Sample-Factory rollouts to GameNGen frame sets. None was used. Most hold pixels and actions only, many declare no license, the permissive ones with game variables (pose, health, ammo) have no enemy, wall or projectile information, and their actions come from reward-trained agents or humans in different scenarios, not from labels of our states. The full list and the reasons are in `DATA_LICENSES.md` and `docs/research.md`.

## Training and DAgger

`scripts/train-doom.sh` reproduces the model:

- **Base.** It starts from the DAgger-improved s2-nano (`runs/s2-nano-g2/decide/last`), already ternary, so every step runs with ternary fake-quant (`--qat-frac 1.0`).
- **Records.** It uses one group per game file so the weights are explicit: the Hugging Face decisions, the four oracle files, and a DAgger set of the three 2D games collected from the base model's own play (`data/dagger-doom/others`), so the fine-tune keeps visiting their hard states.
- **Each round** calibrates, exports the ternary GGUF, sets τ = 0.95 for every question type (`gguf-set`), and plays 50 episodes of every game with `sky-arena`.
- **The gate** asks for doom completion of at least 90% of the oracle's and a kill rate of at least 70% of the oracle's. The other three games must be no worse than the base model at the same τ and seed (one episode in 50 of tolerance on completion, 5% on Flappy's mean score).
- **When the gate fails**, the latest model plays doom and the three 2D games. The oracle labels the states it reaches (`data/dagger-doom/rN`, `oN`), and the next round trains on everything, the newest round weighted most.

Early exit matters in closed-loop play. The calibrated per-type thresholds of the s2 files let the first exit (one layer) answer some critical states, which is fine for held-out accuracy and bad for a game where one wrong frame crashes the drone. Every arena run here uses τ = 0.95, and the published file carries it.

## Results

All the numbers below come from `sky-arena` on an Apple M5 at τ = 0.95, with 50 episodes per game (seed 7) unless noted. Round 0 is the fine-tune on oracle data. It took 12 minutes and 5,496 steps of 128 questions; the doom held-out accuracy went from 0.22 to 0.88 at the last exit, ternary. Round 1 took 2,958 steps, with the GPU shared with another training job.

| model | doom completion | doom kill rate | doom deaths | flappy pipes (crashes) | rescue completion | dodge completion |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 1.00 | 0.75 | 0 | 200 (0) | 1.00 | 0.78 |
| s2-nano (before its DAgger rounds) | | | | 30.6 (50) | 1.00 | 0.58 |
| s2-nano-g2 (base) | 0.00 | 0.00 | | 196.2 (1) | 1.00 | 0.74 |
| round 0 | 1.00 | 0.75 | 0 | 152.4 (22) | 1.00 | 0.72 |
| **round 1 = s2-nano-doom** | **1.00** | **0.75** | **0** | **197.6 (2)** | **1.00** | **0.72** |

Round 0 already passed the doom gate, but Flappy Drone regressed: 22 crashes in 50 runs against 1 for the base. Flappy decides every frame, so a small shift in its decision boundary costs a lot. Round 1 added the DAgger states of all four games played by the round-0 model, and Flappy came back to the base level. Dodge stays one episode in 50 below the base (0.72 against 0.74). Its reflex fires about three times as often as with the base model (3,073 vetoes against 975), so the fine-tune made the model lean more on the coded reflex in that game.

On 100 fresh doom levels (seed 2026) the published model matches the oracle exactly: completion 1.00, kill rate 0.69 (oracle 0.69), no death, mean 469 ticks (oracle 464), and 25 reflex vetoes in 11,800 decisions.

The model is the s2-nano shape: 1,532,033 parameters, 702 KB as a ternary GGUF (`models/s2-nano-doom/`, `site/public/models/s2-nano-doom.gguf`).

- **Latency.** A doom state is about 38 tokens against about 16 for Flappy Drone, and closed-loop play keeps almost every question at full depth (3.99 of 4 layers). One action question takes about 1.05 ms natively (`sky-arena` p50 on a busy machine). A full decision in WASM, with the 3 questions on one state encoding plus the physics, takes about 1.8 ms (`node crates/sky-web/bench/bench-node.mjs --game doom`, Node 24). Both were measured before the engine work of the s3 release; with it, one Doom question costs 740 µs natively and 1,282 µs in WASM SIMD128 (`BENCH.md`, "Inference").
- **Time budget.** A decision is due every 114 ms, so the browser has room to spare.
- **Caching.** An untrained model that stays stuck measures 0.5 ms, because its state string repeats and the s2 engine reuses the cached state encoding: benchmark a model that actually plays.

## s3-pico plays it too

s3-pico ([chapter 21](./21-s3-recursive-tiny)) was trained with `data/games/doom.jsonl` in its decision data, without a Doom-specific fine-tune or Doom DAgger round. In `sky-arena`, 30 episodes per game at its own τ = 0.6 (seed 7), it completes every level with the oracle's kill rate (0.77) and no death, with 10 reflex vetoes, using 1.73 of its 4 loops per Doom decision at a median of 71 µs (`runs/ship-s3-pico.arena.json`). It keeps the three drone games at the oracle's level too, with a 251 KB file instead of 702 KB. A Doom frame is 26 value-fused tokens for s3-pico and costs 77 µs natively and 138 µs in WASM SIMD128 per question (`BENCH.md`, "Inference").

## Limits

- **Measured state.** The model does not see the screen. It plays from 14 measured numbers, and the A* waypoint does most of the navigation. What it learns is the decision layer of a Doom player (aim, fight or move on, dodge, open, press), not perception.
- **Small world.** The levels are small, flat grids: no heights, stairs, lifts, keys or secrets, and two monster types. The weapon has a generous horizontal autoaim.
- **Easy and normal.** Gates and training use the easy skill; normal is only measured for the oracle.
- **Fireballs.** Dodging is partly carried by the coded reflex: the model has `pb`/`pd` but not the fireball's exact path.
- **Sprites.** The browser renderer draws front-facing sprites only, without rotations or sector lighting.
