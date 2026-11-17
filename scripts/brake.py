"""The thermal brake of a training loop: rest between steps so that the machine stays cool and usable.

Standard library only: copy this file into another project, or import it. The guide is
docs/idle-training.md; the reference implementation is crates/sky-train/src/brake.rs, and this module
follows its semantics.

    from brake import Brake

    brake = Brake.from_env()             # SKY_DUTY, SKY_COOL, SKY_BRAKE_FILE
    while brake.working_seconds < 60 * minutes:
        train_one_step()
        brake.rest()                     # after each step

Settings (environment):
  SKY_DUTY        the share of the time the loop works, in (0, 1] (default 1: no short rests). After
                  each step the loop rests in proportion to the work since its last rest:
                  work * (1 / duty - 1).
  SKY_COOL        `work:rest` in minutes, as 45:15: long rests on the wall clock. While the Unix time
                  (in whole seconds) modulo work + rest is past work, the loop rests. Following the
                  wall clock, the cycle holds across processes (the short stages of a pipeline).
  SKY_BRAKE_FILE  a file holding one line, `<duty> [<work>:<rest>]` (`0.85 50:10`, or `0.5` alone
                  for no long rests), written by scripts/auto-brake.sh or by any tool. It is read at
                  start, then again at most every FILE_PERIOD seconds (5), and a valid line replaces
                  SKY_DUTY and SKY_COOL. A missing, unreadable or invalid file changes nothing: the
                  last setting holds. During a long rest the loop naps FILE_PERIOD seconds at a time,
                  so that a new line can end or shorten the rest.

`working_seconds` counts the time since the brake was made, minus every rest: a budget in working
minutes then computes the same thing whatever the brake (only the wall-clock duration changes).

Agreement with the Rust brake: the same lines are valid (a cycle part that rounds to 0 seconds is
refused by both; Python's float() would also take underscores and surrounding spaces, which this
module refuses as Rust's parser does), the same messages are printed, and at duty 1 with no long
rests both move their mark of the last rest at every call, so that a later switch to a lower duty
rests in proportion to the steps after the switch only.

The one difference that remains: `rest()` measures the work since the previous call, as the Rust one
does, but it also takes the seconds of work as an argument (`rest(work_seconds)`), for a loop that
knows them better (experiments/lfm3-s1s2/s1s2.py passes the generation time).
"""
import math
import os
import sys
import time

FILE_PERIOD = 5  # seconds between two reads of the brake file


def _number(text):
    """A float written as Rust's f64 parser accepts it; None otherwise."""
    if not text or "_" in text or text != text.strip() or not text.isascii():
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _valid_duty(duty):
    return duty is not None and 0 < duty <= 1


def parse_cool(text):
    """`work:rest` in minutes as (work, rest) in whole seconds; None when it is not valid."""
    work, sep, rest = text.partition(":")
    if not sep:
        return None
    parts = []
    for m in (_number(work), _number(rest)):
        if m is None or not (m > 0 and math.isfinite(m)):
            return None
        s = math.floor(m * 60 + 0.5)  # rounded half away from zero, as Rust's round()
        if s < 1:  # a part of 0 seconds would divide by zero, or rest forever
            return None
        parts.append(s)
    return parts[0], parts[1]


def minutes(secs):
    """`secs` in minutes, as the brake file writes them: 45, 0.5, 0.02 (two decimals at most)."""
    if secs % 60 == 0:
        return str(secs // 60)
    return f"{secs / 60:.2f}".rstrip("0").rstrip(".")


def parse_brake_line(text):
    """(duty, (work, rest) in seconds or None) of a brake-file line; None when it is not valid."""
    words = text.split()
    if not words or len(words) > 2:
        return None
    duty = _number(words[0])
    if not _valid_duty(duty):
        return None
    if len(words) == 1:
        return duty, None
    cool = parse_cool(words[1])
    if cool is None:
        return None
    return duty, cool


def cool_wait(now, cycle):
    """Seconds left in the rest part of the cycle at Unix time `now` (whole seconds), 0 in its work part."""
    work, rest = cycle
    at = now % (work + rest)
    return 0 if at < work else work + rest - at


def _fmt(duty):
    return repr(duty)[:-2] if duty == int(duty) else repr(duty)


class Brake:
    """Rests between the steps of a training loop. Call `rest()` once per step."""

    def __init__(self, duty=1.0, cool=None, file=None, *, monotonic=time.monotonic,
                 wall=time.time, sleep=time.sleep, log=None):
        if not _valid_duty(duty):
            raise ValueError(f"SKY_DUTY must be in (0, 1], not {duty}")
        self._mono, self._wall, self._sleep = monotonic, wall, sleep
        self._log = log if log is not None else (lambda line: print(line, flush=True))
        self._rest_per_busy = 1 / duty - 1
        self.duty = duty
        self.cool = cool
        self.file = file
        self._created = self._busy_since = self._read_at = monotonic()
        self._rested = 0.0
        if file:
            self.read_file()

    @classmethod
    def from_env(cls, environ=None, **kwargs):
        """The brake of SKY_DUTY, SKY_COOL and SKY_BRAKE_FILE (each may be unset or empty)."""
        env = os.environ if environ is None else environ
        duty = 1.0
        v = env.get("SKY_DUTY", "")
        if v:
            duty = _number(v)
            if duty is None:
                raise ValueError(f"SKY_DUTY is not a number: {v}")
        cool = None
        v = env.get("SKY_COOL", "")
        if v:
            cool = parse_cool(v)
            if cool is None:
                raise ValueError(f"SKY_COOL must be work:rest in minutes, as 45:15, not {v}")
        return cls(duty, cool, env.get("SKY_BRAKE_FILE") or None, **kwargs)

    @property
    def working_seconds(self):
        """Seconds since the brake was made, minus every rest."""
        return self._mono() - self._created - self._rested

    @property
    def rested_seconds(self):
        """Seconds spent in `rest()` so far."""
        return self._rested

    def read_file(self):
        """Takes the duty and the long rests from the brake file, when it holds a valid line."""
        if not self.file:
            return
        self._read_at = self._mono()
        try:
            with open(self.file, encoding="utf-8") as f:
                parsed = parse_brake_line(f.read())
        except (OSError, UnicodeDecodeError):
            parsed = None
        if parsed is None:
            return
        duty, cool = parsed
        if 1 / duty - 1 != self._rest_per_busy or cool != self.cool:
            rests = ("no long rests" if cool is None
                     else f"{minutes(cool[0])} min of work, {minutes(cool[1])} of rest")
            self._log(f"brake: duty {_fmt(duty)}, {rests} ({self.file})")
            self.duty, self._rest_per_busy, self.cool = duty, 1 / duty - 1, cool

    def _refresh(self):
        if self.file and self._mono() - self._read_at >= FILE_PERIOD:
            self.read_file()

    def _wait(self):
        return 0 if self.cool is None else cool_wait(int(self._wall()), self.cool)

    def rest(self, work_seconds=None):
        """Rests after a step; returns the seconds rested.

        The short rest is `work * (1 / duty - 1)`, with `work` the seconds since the previous call
        (or `work_seconds` when given). Then, inside the rest part of the long cycle, it waits for
        the cycle's work part, in naps of FILE_PERIOD seconds when there is a brake file.
        """
        self._refresh()
        started = self._mono()
        if self._rest_per_busy == 0 and self.cool is None:
            self._busy_since = started
            return 0.0
        busy = started - self._busy_since if work_seconds is None else max(0.0, work_seconds)
        self._sleep(busy * self._rest_per_busy)
        if self._wait() > 0:
            self._log(f"cooling down for {self._wait()} s (SKY_COOL)")
        # A loop, because a machine put to sleep can wake up in another rest.
        while self._wait() > 0:
            nap = min(self._wait(), FILE_PERIOD) if self.file else self._wait()
            self._sleep(nap)
            self._refresh()
        rested = self._mono() - started
        self._rested += rested
        self._busy_since = self._mono()
        return rested


if __name__ == "__main__":
    # `python3 brake.py`: the brake this environment sets, and one short check of the file.
    try:
        b = Brake.from_env()
    except ValueError as e:
        sys.exit(str(e))
    cool = "none" if b.cool is None else f"{b.cool[0]} s of work, {b.cool[1]} s of rest"
    print(f"duty {_fmt(b.duty)}, long rests: {cool}, file: {b.file or 'none'}")
