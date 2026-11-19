# Training while you work: the idle brake

skycmd trains its models for days on one laptop, the same one its owner works on. A training at full
speed makes that laptop hot and loud, drains its battery and can freeze it when it runs out of
memory. The brake described here lets a training run hard when the machine is left alone and cool,
and gently when someone uses it, when it runs on battery or when it heats, without stopping the run
and without changing what the run computes.

It is three small pieces that talk through one file:

```text
 scripts/auto-brake.sh  (macOS daemon, one tick every 30 s)
   reads: keyboard and pointer idle time (ioreg), display assertions, battery and
          thermal warnings (pmset), thermal state (scripts/thermal-state.swift),
          CPU of the other programs (ps), memory of the trainer (top)
   writes, only when the value changes, by an atomic rename:
        |
        v
   runs/brake            one line:  0.85 50:10
   runs/brake.log        one line per change, with its reason
        |
        |  read at start, then at most every 5 s
        v
 the training loop       rests between steps: Brake::rest (Rust, crates/sky-train/src/brake.rs)
                         or Brake.rest (Python, scripts/brake.py); the rests stay out of the
                         run clock, so a budget counts working time only
```

`scripts/train-when-idle.sh` starts the daemon, runs your command with `SKY_BRAKE_FILE` set and the
Mac kept awake, and stops the daemon when the command ends.

## Quick start

On macOS, from the repository root:

```sh
scripts/auto-brake.sh --once                       # what the daemon would decide now, and why
scripts/train-when-idle.sh -- python3 train.py     # your training, braked (any command)
tail -f runs/brake.log                             # every change of the brake, with its reason
```

The command must read the brake file: `sky-train` does, and so does any loop that calls
`scripts/brake.py` (below). A command that ignores the file runs at full speed.

`train-when-idle.sh` options: `-f FILE` (the brake file; default `$SKY_BRAKE_FILE`, else
`runs/brake`), `--set NAME=VALUE` (a daemon setting from the table below; the environment works the
same), `--no-daemon` (only export `SKY_BRAKE_FILE` and run the command), `--no-caffeinate`, `--help`.
If a daemon already serves the brake file, it is used and left running; otherwise the one started
here stops with the command. The exit status is the command's.

## The brake-file protocol

This is the whole interface between the daemon and a trainer. Anything that follows it can take
either side.

- **Content.** One line: `<duty> [<work>:<rest>]`, words separated by white space, a final newline
  allowed. Examples: `0.85 50:10`, `0.5 45:15`, `1`.
- **duty** is the share of the time the training works, a number in (0, 1]. After each step the loop
  rests `work × (1 / duty − 1)`, where `work` is the time since its previous rest: at 0.5 it rests as
  long as it worked; at 1 it does not rest.
- **work:rest** (optional) is a cycle of long rests, in minutes. Each part is rounded to whole
  seconds and must stay at least one second (`0.001` minutes is refused: a part of 0 seconds would
  divide by zero, or rest forever). The cycle follows the wall clock: while the Unix time in seconds, modulo
  `work + rest`, is past `work`, the loop rests completely. Following the wall clock keeps the cycle
  across the processes of a pipeline. Without it there are no long rests.
- **Writing.** Write another file in the same folder, then rename it over the brake file (`mv` is
  atomic on one file system), so that a reader never sees half a line. The daemon writes only when
  the value changes.
- **Reading.** A trainer reads the file when it starts, then at most every 5 seconds, between steps
  and between the naps of a long rest (so a new line can end a long rest within about 5 seconds).
- **Precedence.** A valid line replaces `SKY_DUTY` and `SKY_COOL`. A missing, unreadable or invalid
  file (empty, not a number, a duty of 0 or above 1, a part of 0 seconds, a third word) changes
  nothing: the last setting holds, and before any valid line that is `SKY_DUTY` (default 1) and
  `SKY_COOL` (default none). (A third reader, `experiments/lfm3-s1s2/s1s2.py`, written for one
  pre-registered experiment and left as it ran, differed: it read the file at every rest and went
  back to `SKY_DUTY` and `SKY_COOL` while the file was invalid. It was removed with the LFM3 code on
  19/11/2026 and is in the git history at commit 5011585: [`experiments/lfm3-s1s2/s1s2.py`](https://github.com/maxgfr/skycmd/blob/5011585/experiments/lfm3-s1s2/s1s2.py).)

Without the daemon, one line of shell is a valid writer:

```sh
printf '0.5 45:15\n' > runs/brake.tmp && mv runs/brake.tmp runs/brake
```

## The modes

The daemon checks these in order at every tick; the first that holds wins.

| mode | when | brake (setting, default) | reason in the log |
|---|---|---|---|
| too hot | macOS reports a thermal state of "serious" or worse, or `pmset -g therm` reports a speed limit or a thermal warning | `HOT`, default `IN_USE` | `thermal pressure (state N)` |
| on battery | the power source is the battery | `HOT` | `on battery` |
| in use | an application other than `caffeinate` keeps the display awake (a video, a call) | `IN_USE`, `0.5 45:15` | `a video or a call keeps the display awake` |
| in use | the idle time could not be read (the careful side) | `IN_USE` | `idle time unknown: treated as in use` |
| in use | a key or the pointer moved in the last `IDLE_MIN` minutes | `IN_USE` | `the Mac is in use` |
| night | no input for `FULL_IDLE_MIN` minutes, between the hours `FULL_HOURS`, and the other programs below `OTHER_CPU` percent of one core | `FULL`, `0.95 55:5` | `night: no input for 30 min, nothing else works` |
| left alone | none of the above | `AWAY`, `0.85 50:10` | `no input for 10 min` |

## Every setting

All are environment variables of `scripts/auto-brake.sh` (or `--set NAME=VALUE` of
`train-when-idle.sh`). The daemon checks them at start and refuses an invalid one (exit status 2).

| setting | default | meaning |
|---|---|---|
| `IN_USE` | `0.5 45:15` | brake while the Mac is in use |
| `AWAY` | `0.85 50:10` | brake while it is left alone |
| `FULL` | `0.95 55:5` | brake of the night mode; `FULL=""` turns the night mode off |
| `HOT` | the value of `IN_USE` | brake under thermal pressure and on battery |
| `IDLE_MIN` | `10` | minutes with no key or pointer input before the Mac counts as left alone |
| `FULL_IDLE_MIN` | `30` | minutes with no input before the night mode |
| `FULL_HOURS` | `23-8` | hours of the night mode, `from-to`: `23-8` is 23:00 to 08:00, `0-24` always, `9-17` 09:00 to 17:00 |
| `OTHER_CPU` | `60` | the night mode needs the other programs below this percent of one core; once it is on, below twice that, so that a short burst does not switch it off and on |
| `TRAIN_PROCS_RE` | `(sky-train\|sky-arena\|sky-eval\|[Pp]ython[0-9.]*)$` | awk regular expression over the whole command path of `ps -o comm=`, spaces included: the processes it matches are not "other programs" (`[Pp]` covers the framework build of Python, `.../MacOS/Python`) |
| `MEM_STOP_GB` | `13` | memory guard: stop the trainer processes that hold this many GB or more; `0` turns the guard off |
| `MEM_STOP_N` | `4` | readings in a row over the limit before the stop (4 ticks: two minutes) |
| `TRAIN_PROC` | `sky-train` | process name of the trainer, for the memory guard (`pgrep -x`) |
| `PERIOD` | `30` | seconds between two ticks |

The trainer side reads three variables:

| variable | default | meaning |
|---|---|---|
| `SKY_DUTY` | `1` | the duty before any valid brake-file line |
| `SKY_COOL` | none | the `work:rest` cycle before any valid brake-file line |
| `SKY_BRAKE_FILE` | none | the brake file; `train-when-idle.sh` sets it to an absolute path |

`sky-train` also reads `SKY_MLX_CACHE_GB`, a cap on MLX's cache of freed buffers, in units of 2^30
bytes (`crates/sky-train/src/main.rs`; unbounded when unset). It is not part of the brake, but the
training scripts of this repository set it (2 by default) because without a cap MLX keeps every
freed buffer for reuse on a 16 GB machine.

## The guards

- **Heat.** `pmset -g therm` records nothing on Apple silicon until the machine is already throttled,
  so the daemon also reads the thermal state that macOS reports to applications, through a
  six-line Swift helper (`scripts/thermal-state.swift`) that it builds once into `<file>.thermal`
  when `swiftc` is there. A state of 2 ("serious") or 3 ("critical") applies `HOT`.
- **Battery.** On battery the daemon applies `HOT` whatever else holds.
- **Memory.** When a process named `TRAIN_PROC` holds `MEM_STOP_GB` or more for `MEM_STOP_N`
  readings in a row, the daemon sends `TERM` to the processes of that name over the limit at that
  tick, and writes a line in the log with their pids (`sky-train holds 14 GB (limit 13): training
  stopped (TERM to pid 22)`). A run that swaps does almost no work and fills the disk; the guard
  trades it for a restart from the last checkpoint. A peak that lasts less than the window (the
  loading of a data set) does not count. The memory comes from one `top` call: every unit it prints
  (`950K`, `2587M`, `13G`, `16G+`) is converted, then truncated to whole GB of 1,024 MB, so `13500M`
  reads as 13 GB.
- **Readings that fail.** A probe that cannot be read takes the careful side: an unknown idle time
  counts as in use (`idle time unknown: treated as in use`), an unknown hour is not night, and an
  unknown CPU reading of the other programs is not quiet. The daemon never speeds a training up on
  a reading it could not make.

## Several trainers at once

The memory guard looks at every process named `TRAIN_PROC` in one `top` call. It counts a reading
as over the limit when the largest of them holds `MEM_STOP_GB` or more; after `MEM_STOP_N` such
readings in a row it sends `TERM` only to the processes over the limit at that tick. A second run's
evaluation or probe under the limit goes on, and the log names the pids it stopped. The count does
not follow one process: two trainers that take turns over the limit add up to one window. The
brake file itself is shared: every trainer that reads it gets the same duty and cycle, and each
counts its own working time. For two runs with different brakes, give each its own brake file (and
its own daemon: one daemon serves one file).

## Why budgets count working minutes

`Brake::rest` moves the start of the run clock forward by the time it rested, and `Brake.rest` in
Python keeps a `working_seconds` counter that leaves every rest out. A budget of 60 working minutes
is then the same 60 minutes of computation in every mode: the same number of steps on the same
machine, the same learning-rate schedule, the same model. Only the wall-clock duration changes. This
is what lets the pre-registered experiments of this repository run under a brake that changes with
the owner's day: their budgets are fixed before the run, and the brake cannot move them.

The cost in wall-clock time, with the defaults, is `(1 / duty) × (work + rest) / work`. TODO.md
("Times") gives it as about ×1.15 in the night mode (0.95, 55:5), ×1.4 when the Mac is left alone
(0.85, 50:10) and ×2.7 when it is in use or on battery (0.5, 45:15).

A budget in steps (`--steps` in `sky-train`) is not affected either.

## Use it with your own trainer

### Rust

`crates/sky-train/src/brake.rs` depends only on `std` and `anyhow` (its tests also use `tempfile`):
copy it into your crate.

```rust
use std::time::Instant;
mod brake; // the copied file
use brake::Brake;

let mut clock = Instant::now();          // start of the run clock
let mut brake = Brake::from_env()?;      // SKY_DUTY, SKY_COOL, SKY_BRAKE_FILE
while clock.elapsed().as_secs_f64() < 60.0 * minutes {
    brake.rest(&mut clock);              // once per step; moves `clock` past the rest
    train_step()?;
}
```

### Python

`scripts/brake.py` uses the standard library only: copy it next to your script.

```python
from brake import Brake

brake = Brake.from_env()                 # SKY_DUTY, SKY_COOL, SKY_BRAKE_FILE
for batch in loader:                     # a PyTorch, MLX or JAX loop alike
    loss = step(model, batch)
    brake.rest()                         # once per step: rests in proportion to the work
    if brake.working_seconds >= 60 * minutes:
        break                            # a budget in working minutes
```

`brake.rest(seconds)` takes the seconds of work when the loop knows them better than the time
between two calls. `python3 scripts/test_brake.py` runs its tests.

### Any other language

```text
every step:
    if 5 s have passed since the last read: read the file; if it holds a valid line, take duty, cycle
    sleep (time since the last rest) × (1 / duty − 1)
    while cycle and (unix_seconds mod (work + rest)) >= work: sleep min(5, time left); read the file
    add the time slept to the time the run clock leaves out
```

## Adapting the daemon

- **Your trainer's name.** `TRAIN_PROC=my-trainer` for the memory guard, and add it to
  `TRAIN_PROCS_RE`, or its CPU counts as "other programs" and keeps the night mode off. Check the
  names with `ps -A -o comm= | grep -i <name>`. The default pattern covers `python3.12` and the
  framework build of Python (Homebrew's, python.org's), which shows as `.../MacOS/Python` with a
  capital P.
- **Your hours.** `FULL_HOURS=0-24` allows the night mode at any hour; `FULL=""` turns it off.
- **Your thresholds.** For a day away from the machine, 0.85 whatever the input and the hour, and
  0.5 if it heats: `IN_USE="0.85 50:10" HOT="0.5 45:15" FULL="" scripts/auto-brake.sh`.
- **No memory guard.** `MEM_STOP_GB=0`.

## Diagnosis

`scripts/auto-brake.sh --once [file]` runs one tick and prints every reading (thermal state, pmset,
battery, display, idle seconds, hour, CPU of the other programs, memory of the trainer), the
decision with its reason, and whether a tick would change the file. It writes nothing. The log
`<file>.log` holds one dated line per change; `sky-train` prints `brake: duty …` when it takes a new
line and `cooling down for N s (SKY_COOL)` at the start of a long rest.

One daemon serves one brake file: it holds `<file>.pid` while it runs and removes it, with
`<file>.tmp`, on exit. A second start on the same file exits with status 3; a pid file whose process
is gone or is not `auto-brake.sh` is taken over. Exit statuses: 0 stopped (TERM or INT), 2 usage or
an invalid setting, 3 already running, 4 not macOS.

`bash scripts/test-auto-brake.sh` tests the decision table, the file writes, the pid file and
`train-when-idle.sh`, on Linux and macOS.

## Limits

- **The daemon is macOS only.** Its probes use `ioreg`, `pmset` and `top -l`; elsewhere it exits with
  a message. The protocol and the two readers work anywhere: write the file with your own tool and
  run `train-when-idle.sh --no-daemon`.
- **"In use" means input or a held display.** The daemon sees keys, the pointer and display
  assertions. Someone reading a long page without touching anything counts as away after
  `IDLE_MIN` minutes; a program that holds the display awake counts as use (`caffeinate` excepted).
- **The trainer must cooperate.** The brake is a request in a file. A trainer that does not read it
  is not slowed down, and the daemon does nothing to it but the memory guard.
- **The memory guard stops, it does not save.** It sends `TERM` to the trainer processes over the
  limit; the training must write checkpoints and resume from them. Whole GB are compared, by
  truncation: 13.9 GB is 13.
- **The CPU of the other programs is a rough signal.** It is one `ps` reading per tick, summed over
  every process the pattern does not match (the whole command path, spaces included).
- **The on-battery probe has a theoretical race.** Under `set -o pipefail`, `pmset -g batt | head -n
  1 | grep -q` could read "not on battery" if `pmset` were cut off by a closed pipe; its output is
  two short lines, so this has not been seen.
- **`experiments/lfm3-s1s2/s1s2.py` kept its own reader** until the LFM3 code was removed on
  19/11/2026 (it is in the git history at commit 5011585, [`experiments/lfm3-s1s2/s1s2.py`](https://github.com/maxgfr/skycmd/blob/5011585/experiments/lfm3-s1s2/s1s2.py)). It belonged to a
  finished pre-registered experiment and was left as it ran: besides the differences above (no
  5-second throttle, back to the environment on an invalid file), it accepted `nan` as a cycle
  part, which then made its `time.sleep` fail. `scripts/brake.py` and the Rust brake refuse it.

## What we learned

Each entry says where it comes from: an incident in the project's records, or a defect found when
the code was reviewed to make it reusable (November 2026), which no run is recorded to have hit.

- **A daemon that dies says nothing.** On 13/11/2026, from about 10:10 to 18:18, the daemon was not
  running: it had ended silently, under `set -e`, when a training process ended during one of its
  memory readings. The brake file kept `0.5 45:15` and the memory cap was not watched. Budgets count
  working time, so only the duration of those runs changed (`docs/big-run-report.md`, incidents).
  That reading was guarded the same day (commit `a284560`); every probe is now, and the daemon holds
  a pid file that `train-when-idle.sh` checks.
- **`caffeinate` is not a person.** Command-line tools hold display assertions for hours with nobody
  at the screen, so `caffeinate`'s assertion does not count as use (comment in
  `scripts/auto-brake.sh`). The long runs themselves are launched under `caffeinate -i`.
- **pmset is late on Apple silicon.** It records nothing until the machine is already throttled,
  hence the thermal-state helper (`scripts/thermal-state.swift`).
- **The memory guard catches real failures.** It stopped the first nano run of s6-sleep v3 on
  16/11/2026 at step 203 of its 12 working minutes: the trainer held 16 GB on a 16 GB machine (limit
  13 GB, four readings in a row). The fix cut each batch into parts with the same update, and the
  six nano runs then ran through the night without a stop (`docs/s6-sleep3-prereg.md`,
  amendment 3; TODO.md). On 13/11 it stopped a decision training at 15 GB, in swap, after 8.2 of its
  20 minutes (`docs/big-run-report.md`).
- **It does not catch all swapping.** On 14/11 a decision training of s6-micro was in swap from about
  its minute 92 at 9.3 GB of heap, below the limit, and answered 110 questions per second instead of
  205; it was stopped by hand (`docs/big-run-report.md`).
- **The night mode needs a margin.** Commit `27db95c` made it hold through a short burst of the
  other programs: once it is on, their limit doubles.
- **The project's own work counts as "other programs".** Incident: on the night of 16 to 17/11/2026
  the daemon left the night mode while other work of the project ran on the CPU (evaluations and
  builds counted as "other programs"), until it was restarted with `OTHER_CPU=250`. The brake log
  of that night is not in the repository. For such nights, raise `OTHER_CPU` or name those
  programs in `TRAIN_PROCS_RE`.
- **A framework Python is named `Python`.** Review: `ps -o comm=` names the framework build of
  Python that Homebrew installs `.../MacOS/Python`, with a capital P, which the first pattern
  (`python[0-9.]*`) did not match, so a Python trainer would have counted as another program. The
  pattern now reads `[Pp]ython`, and it is matched against the whole command path, so that a
  trainer in a folder with a space in its name is recognized too.
- **Megabytes are memory too.** Review: `top` prints a process's memory in K, M or G; the first
  guard read only the G values, so a trainer printed as `13500M` would have read as 0 GB. Every unit
  is now converted.
- **One run's guard must not stop another.** Review: the first guard sent `TERM` to every process
  named `sky-train` when one was over the limit, evaluations and probes of other runs included. It
  now stops only the processes over the limit.
- **Unknown is not idle.** Review: a failed idle reading used to count as "left alone", the faster
  brake; it now counts as in use.
- **Cycles under a second.** Review: the Rust brake accepted a cycle part that rounds to 0 seconds
  (`1 0.001:0.001` divided by zero; `0.001:10` never worked again), and after a stretch at duty 1
  its first rest at a lower duty was in proportion to the whole stretch. Both are fixed, with tests,
  in `crates/sky-train/src/brake.rs`; `scripts/brake.py` behaves the same.
- **Battery wins.** From 22:59 on 13/11 to 00:20 on 14/11 the Mac ran on battery (80% down to 47%)
  and the brake stayed at `0.5 45:15` (`docs/big-run-report.md`).
