# 20 What DAgger taught us

The first trained s1-pico labelled 82% of held-out Flappy questions correctly and then crashed in all 30 Flappy episodes, before the first pipe. This chapter is about that gap between answering questions and playing a game: why behaviour cloning leaves it open, the two fixes that closed it, the round-by-round results for s1-pico, s1-nano and s2-nano, and why the bigger s1 model did worse.

Every arena number below is `sky-arena` with 30 episodes per game and the default seed, from the `arena.json` file named next to it, unless the text says otherwise. Flappy is capped at 200 pipes.

## Behaviour cloning and its blind spot

The training data of [chapter 04](./04-game-cores-and-oracles) is behaviour cloning: the oracle plays, and every state it visits becomes a record labelled with the oracle's answer. `gen.rs` already adds noise (each episode draws an exploration rate from 0, 0.05, 0.15 or 0.3), so the data contains some off-course states and their recoveries.

It is still the oracle's distribution of states. Once the model plays, its own small mistakes take it to states the oracle rarely visits, where it is less accurate, which leads to more mistakes. Held-out accuracy cannot see this, because the held-out questions also come from the oracle's play.

DAgger (dataset aggregation) closes the loop:

1. the current model plays;
2. the oracle labels every state the model reaches;
3. the model is fine-tuned on the old data plus the new states;
4. repeat with the new model.

In skycmd, step 1 and 2 are `sky-arena <model> --dagger N --out DIR`. The model drives the episodes and the oracle writes the targets. 20% of the decisions are handed back to the oracle during collection (`DAGGER_ORACLE_SHARE` in `crates/sky-games/src/gen.rs`), so that episodes still reach the later parts of each course. `docs/research-night.md` lists the same recipe as the standard one for game control, with behaviour cloning then DAgger on Flappy Bird in Stanford's CS224R as an example.

## The Flappy failure

`runs/s1-pico/arena.json`, the first checkpoint out of `scripts/train-ladder.sh pico`:

| game | model | oracle |
|---|---|---|
| Flappy | 0.0 pipes, 30 crashes, 52 steps per episode on average | 200.0 pipes, 0 crashes |
| Rescue | rescue rate 0.63, completion 0.57, 130 vetoes | 1.00, 1.00 |
| Dodge | completion 0.67, 5,828 vetoes | 0.87 |

Flappy is the strict test. It asks for a decision every frame, and one wrong frame near a pipe ends the episode. Here the episodes did not even reach a pipe.

The model had learned the label prior. In the Flappy records, `glide` is the better action most of the time: in the current `data/games/flappy.jsonl`, 70.65% of the 63,000 `action` records put more than half of the target on `glide`, and the mean target on `thrust` is 0.295. A model that always leans toward `glide` gets most held-out questions right and falls out of the sky.

The per-frame trace from that debugging session, the model's p(thrust) frame by frame while the drone fell, was not saved in `runs/`, so this chapter does not quote it. To look at your own model the same way, send the successive states of an episode to `skycmd decide --model <file.gguf> --request -` and print `answers.action.probabilities.thrust`.

## Fix 1: the code measures, the model decides

The Flappy state used to give the gap edges as absolute heights. The doc comment above `state_text` still describes those older fields (`alt vy dx gt gb nc`: altitude, vertical speed, distance, gap top, gap bottom, next gap center). To know whether the drone is about to hit the bottom of the gap, the model had to compare two numbers, its altitude and the gap bottom. A 2-layer, 64-wide encoder reading binned integers as tokens is bad at that subtraction.

The fix moves the subtraction into Rust:

`crates/sky-games/src/flappy.rs`
```rust
        // Clearances are relative to the drone, so the model reads "room above / below" directly
        // instead of subtracting absolute heights (the code measures, the model decides).
        format!(
            "vy {} dx {} up {} dn {} nxt {} alt {}",
            bin(self.vy, 25.0, -16, 16),
            bin(dx, UNIT, -12, 60),
            bin(p.gap_top - (self.y + DRONE_HH), UNIT, -20, 20),
            bin((self.y - DRONE_HH) - p.gap_bottom, UNIT, -20, 20),
            bin(next.center() - p.center(), UNIT, -20, 20),
            bin(self.y, UNIT, 0, 64),
        )
```

`dn 2` now means "2 units of room below the drone" whatever the gap's height. The same rule shapes the Rescue and Dodge states and the 14 numbers of the Doom state ([chapter 18](./18-doom)): distances, bearings and clearances are computed in code, and the model only maps them to a decision.

The fix changed the data, so `data/games/flappy.jsonl` was regenerated with the new text. The base s1-pico row above predates it: its arena file is older than the current `flappy.rs` and `flappy.jsonl`, and its report counts 3,100 Flappy test questions against 4,604 in the regenerated data. Round 1 below was the first training on relative clearances and on DAgger states together, so the tables cannot separate the effect of the two fixes.

## Fix 2: DAgger rounds

```bash
scripts/specialize.sh pico 2 6      # s1: 2 rounds of 6 minutes
scripts/specialize-s2.sh nano 2 8   # s2: 2 rounds of 8 minutes, fully ternary
```

Each round of `scripts/specialize.sh`:

1. `sky-arena <model> --dagger 15000 --seed <round> --out data/dagger-<size>/r<round>`: 15,000 records per game from the model's own play;
2. `sky-train decide --init <previous checkpoint>` on `data/decisions,data/games,<dagger dir>` with the weights `games=0.5,r<round>=0.4,decisions=0.1`, for the given number of minutes;
3. calibrate, export, and play 30 episodes per game into `runs/s1-<size>-g<round>/arena.json`.

The s2 script does the same with `s2-decide --qat-frac 1.0`, so every step trains the ternary weights.

## Round by round

### s1-pico (202,305 parameters)

| checkpoint | decision steps | Flappy pipes (crashes) | Rescue rate | Dodge completion | Dodge vetoes | file |
|---|---:|---:|---:|---:|---:|---|
| base | 29,484 | 0.0 (30) | 0.63 | 0.67 | 5,828 | `runs/s1-pico/arena.json` |
| round 1 | 8,110 | 71.3 (28) | 1.00 | 0.70 | 4,756 | `runs/s1-pico-g1/arena.json` |
| **round 2 (shipped)** | 7,297 | **200.0 (0)** | **1.00** | **0.83** | 1,026 | `runs/s1-pico-g2/arena.json` |
| round 3 | 14,763 | 200.0 (0) | 1.00 | 0.80 | 1,021 | `runs/s1-pico-g3/arena.json` |
| oracle | | 200.0 (0) | 1.00 | 0.87 | 0 | |

Steps per round come from the `done:` lines of `runs/specialize-pico.log` and `runs/specialize-2.log`. Round 2 is the model on the site: `site/public/models/s1-pico.gguf` is byte-identical to a fresh `sky-convert models/s1-pico-g2`, and `runs/ship-s1-pico.arena.json` replays it with the same results. Round 3 got twice as many steps and came out slightly worse on Dodge (0.80 against 0.83), so it was not shipped.

The held-out report of round 2 (`runs/s1-pico-g2/report.md`) gives 0.8113 on Flappy questions, a little below the 0.8245 of the base report (on the older, smaller split). Closed-loop play went from 0 to 200 pipes while held-out accuracy stayed flat. That is the main lesson of this chapter in one line.

### s1-nano (1,134,209 parameters)

| checkpoint | decision steps | Flappy pipes (crashes) | Rescue rate | Dodge completion | Dodge vetoes | file |
|---|---:|---:|---:|---:|---:|---|
| base | 14,797 | 0.0 (30) | 0.23 | 0.53 | 5,082 | measured for this chapter |
| round 1 | 5,482 | 200.0 (0) | 0.97 | 0.70 | 14 | `runs/s1-nano-g1/arena.json` |
| round 2 | 1,104 | 75.8 (26) | 0.93 | 0.43 | 10,121 | `runs/s1-nano-g2/arena.json` |
| oracle | | 200.0 (0) | 1.00 | 0.87 | 0 | |

No arena file was written for the base s1-nano. Its row was measured for this chapter with `sky-arena models/s1-nano --episodes 30 --game flappy,rescue,dodge` (Flappy mean score 0.03).

### s2-nano (1,532,033 parameters, ternary)

The s2 rounds were played at the thresholds calibrated on validation questions, which turned out to be the wrong setting ([chapter 19](./19-s2-architecture), section 6).

| checkpoint | τ | Flappy pipes (crashes) | Rescue rate | Dodge completion | Dodge vetoes | file |
|---|---|---:|---:|---:|---:|---|
| round 1 (3,494 steps) | calibrated 0.57 / 0.61 / 0.72 | 4.8 (30) | 1.00 | 0.50 | 2,365 | `runs/s2-nano-g1/arena.json` |
| round 2 (3,233 steps) | calibrated 0.62 / 0.62 / 0.69 | 16.6 (30) | 1.00 | 0.50 | 2,163 | `runs/s2-nano-g2/arena.json` |
| **round 2 (shipped)** | **0.95** | **200.0 (0)** | **1.00** | **0.87** | **25** | `runs/ship-s2-nano.arena.json` |
| round 2 | 1.0 (full depth) | 200.0 (0) | 1.00 | 0.87 | 25 | measured for [chapter 19](./19-s2-architecture) |
| oracle | | 200.0 (0) | 1.00 | 0.87 | 0 | |

The DAgger rounds worked; the exit threshold hid it. At τ = 0.95, round 2 matches the oracle on all three games, and its reflex fires 25 times in Dodge against 1,026 for the shipped s1-pico.

Two more data points, from `scripts/train-doom.sh` at τ = 0.95 with 50 episodes and seed 7 (`runs/doom/`): the s2-nano from before DAgger flies 30.6 pipes with 50 crashes and completes 0.58 of Dodge, and round 2 flies 196.2 pipes with 1 crash and completes 0.74 of Dodge, against 0.78 for the oracle on that seed. Round 2 is near the oracle there too, but not equal to it on every seed.

The same procedure did not save s2-pico: after two rounds it still crashes in every Flappy episode at full depth (chapter 19, section 7).

## Why bigger was not better for s1

s1-nano has 5.6 times the parameters of s1-pico. Its best checkpoint, round 1, flies 200 pipes like pico but completes 0.70 of Dodge against 0.83 and rescues 0.97 against 1.00. Its second round collapsed. Four things in the logs explain most of it:

1. **The base model trained on the old state text.** The nano pretraining ended at 14:48 (`runs/s1-nano/pretrain/last/state.json`), and the regenerated `data/games/flappy.jsonl` is from 14:56, so the nano decision run most likely loaded the absolute-height Flappy states. Its first validation sample holds the same 1,881 game questions as the base pico run, while s1-micro, trained later, drew 1,989. Its report (written at 15:09, on the regenerated test split) gives 0.6351 on Flappy against 0.8202 for s1-micro on the same split. The base nano never had a fair start.
2. **Fewer updates in the same time.** A nano step costs about 2.5 times a pico step (`BENCH.md`). In its 20-minute budget nano made 14,797 decision steps against 29,484 for pico in 12 minutes.
3. **Round 2 barely trained.** It made 1,104 steps in its 8 minutes against 5,482 in round 1 (`runs/specialize-2.log`). The log does not say why; `BENCH.md` notes that the s2 runs of the same evening shared the GPU with an s1 training job. With so few steps, the round mostly shifted the model toward its newest DAgger data, and Flappy (75.8 pipes, 26 crashes) and Dodge (0.43, 10,121 vetoes) both regressed.
4. **It is 5 times slower.** 974 µs per question natively and 3,541 µs per WASM decision, against 188 and 694 µs for pico (`BENCH.md`).

None of this proves that nano cannot beat pico. It shows that, at a fixed wall-clock budget, the extra capacity was not the bottleneck, and that every round needs a closed-loop check before it replaces the previous model. The capacity question was answered by s2 instead, with an architecture that does less work per frame ([chapter 19](./19-s2-architecture)).

## What we keep

- **Gate every model on closed-loop play.** s1-pico's Flappy held-out accuracy moved by less than 2 points while its Flappy score went from 0 to 200 pipes.
- **Keep the best round, not the last one.** s1-pico round 3 and s1-nano round 2 were both worse than the round before.
- **Compute relations in code.** Relative clearances cost one subtraction in Rust and remove a comparison the model was failing to learn.
- **Calibrate early exits on played states**, not on i.i.d. questions (`scripts/calibrate-exit.sh`).
- **Keep the coded reflex.** The veto counts in the tables are the reflex catching the model. Fewer vetoes at equal completion means a better model, and the reflex is still there when it is not.

## Next

[Back to the tutorial index](./), or [play the games](/play/) with the published models.
