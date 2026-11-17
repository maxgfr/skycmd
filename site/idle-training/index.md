---
title: Train while you work
---

# Train while you work

skycmd's models are trained on one laptop, the one its owner works on, sometimes for days. At full
speed a training makes that laptop hot and loud, drains its battery, and can freeze it when it runs
out of memory. A small brake lets the training run hard when the machine is left alone and cool, and
gently when someone uses it, when it is on battery or when it heats, without stopping the run and
without changing what it computes. It is not tied to skycmd: any training loop can use it.

The full guide, with every setting, the guards and what went wrong along the way, is
[docs/idle-training.md](https://github.com/maxgfr/skycmd/blob/main/docs/idle-training.md).

## How it works

```text
 scripts/auto-brake.sh   macOS daemon: idle time, display, battery, heat, memory
        |  writes one line, only when it changes, by an atomic rename
        v
   runs/brake            0.85 50:10
        |  read at start, then at most every 5 s
        v
 the training loop       rests between steps; the rests stay out of the run clock
```

| the Mac is | brake | meaning |
|---|---|---|
| too hot, or on battery | `0.5 45:15` | works half the time, and rests 15 minutes in every hour |
| in use (input in the last 10 minutes, or a video or a call) | `0.5 45:15` | the same |
| left alone | `0.85 50:10` | works 85% of the time, rests 10 minutes in every hour |
| left alone for 30 minutes at night, nothing else working | `0.95 55:5` | nearly full speed |

The rests are taken out of the run clock, so a budget of 60 working minutes is the same computation in
every mode: only the wall-clock time changes. With these defaults it is about ×1.15 at night, ×1.4
when the Mac is left alone and ×2.7 when it is in use or on battery (the "Times" paragraph of
[TODO.md](https://github.com/maxgfr/skycmd/blob/main/TODO.md)).

## Quick start

On macOS, from the repository root:

```sh
scripts/auto-brake.sh --once                       # what the daemon would decide now, and why
scripts/train-when-idle.sh -- python3 train.py     # your training, braked (any command)
tail -f runs/brake.log                             # every change of the brake, with its reason
```

`train-when-idle.sh` starts the daemon for the brake file, exports `SKY_BRAKE_FILE` to the command,
keeps the Mac awake while it runs (`caffeinate -i`), and stops the daemon when the command ends, if it
started it. Its exit status is the command's. The command must read the brake file; one that ignores
it runs at full speed.

## The brake-file protocol

- **One line:** `<duty> [<work>:<rest>]`, as `0.85 50:10`, `0.5 45:15` or `1`.
- **duty**, in (0, 1], is the share of the time the training works: after each step it rests
  `work × (1 / duty − 1)`, where `work` is the time since its previous rest.
- **work:rest**, optional, in minutes, each part above 0, rounded to whole seconds: a cycle of long
  rests on the wall clock. While the Unix time in seconds, modulo `work + rest`, is past `work`, the
  training rests completely.
- **Writing:** write another file, then rename it over the brake file, so that a reader never sees
  half a line.
- **Reading:** at start, then at most every 5 seconds, also during a long rest. A valid line replaces
  `SKY_DUTY` and `SKY_COOL`; a missing or invalid file changes nothing.

Off macOS, anything that writes the line works, and `train-when-idle.sh --no-daemon` only exports the
variable and runs the command:

```sh
printf '0.5 45:15\n' > runs/brake.tmp && mv runs/brake.tmp runs/brake
```

## Use it with your own trainer

**Python**: copy [scripts/brake.py](https://github.com/maxgfr/skycmd/blob/main/scripts/brake.py)
(standard library only) next to your script.

```python
from brake import Brake

brake = Brake.from_env()                 # SKY_DUTY, SKY_COOL, SKY_BRAKE_FILE
for batch in loader:                     # a PyTorch, MLX or JAX loop alike
    loss = step(model, batch)
    brake.rest()                         # once per step
    if brake.working_seconds >= 60 * minutes:
        break                            # a budget in working minutes
```

**Rust**: copy
[crates/sky-train/src/brake.rs](https://github.com/maxgfr/skycmd/blob/main/crates/sky-train/src/brake.rs)
(`std` and `anyhow` only).

```rust
let mut clock = Instant::now();          // start of the run clock
let mut brake = Brake::from_env()?;      // SKY_DUTY, SKY_COOL, SKY_BRAKE_FILE
while clock.elapsed().as_secs_f64() < 60.0 * minutes {
    brake.rest(&mut clock);              // once per step; moves `clock` past the rest
    train_step()?;
}
```

**Any other language**:

```text
every step:
    if 5 s have passed since the last read: read the file; if it holds a valid line, take duty, cycle
    sleep (time since the last rest) × (1 / duty − 1)
    while cycle and (unix_seconds mod (work + rest)) >= work: sleep min(5, time left); read the file
    add the time slept to the time the run clock leaves out
```

## Limits

- The daemon is macOS only (`ioreg`, `pmset`, `top`); the protocol and the two readers work anywhere.
- "In use" means keyboard or pointer input, or an application that keeps the display awake.
- A trainer that does not read the file is not slowed down.
- The memory guard (13 GB by default) stops the training with `TERM`: the training must resume from
  its checkpoints.
- The CPU of the other programs, which gates the night mode, is one rough `ps` reading per tick. Name
  your trainer in `TRAIN_PROCS_RE`, or its own CPU keeps the night mode off.
