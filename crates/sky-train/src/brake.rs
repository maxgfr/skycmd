//! The thermal brake of a training loop: rests between steps so that the machine stays cool and
//! usable, with the rests kept out of the run clock (docs/idle-training.md).
//!
//! # Contract
//!
//! A training loop makes one [`Brake`] with [`Brake::from_env`] before its first step, keeps an
//! [`Instant`] as the start of its run clock, and calls [`Brake::rest`] with that clock once per
//! step (before or after the step, always at the same place):
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use sky_train::brake::Brake;
//! use std::time::Instant;
//!
//! let budget_secs = 60.0 * 60.0; // a budget in working minutes
//! let mut clock = Instant::now(); // the start of the run clock
//! let mut brake = Brake::from_env()?; // SKY_DUTY, SKY_COOL, SKY_BRAKE_FILE
//! while clock.elapsed().as_secs_f64() < budget_secs {
//!     brake.rest(&mut clock); // rests, then moves `clock` forward by the time rested
//!     // one training step
//! }
//! # Ok(())
//! # }
//! ```
//!
//! `clock.elapsed()` is then the working time only: a budget, a learning-rate schedule or any
//! decision that reads it computes the same thing whatever the brake, and only the wall-clock
//! duration of the run changes. A loop that never calls `rest` is not slowed down.
//!
//! # The brake-file protocol
//!
//! `SKY_BRAKE_FILE` names a file of one line, `<duty> [<work>:<rest>]`: the duty in (0, 1], and
//! optional long rests in minutes (each part above 0). It is read when the brake is made, then at
//! most every 5 seconds, and a valid line replaces `SKY_DUTY` and `SKY_COOL`. A missing, unreadable
//! or invalid file changes nothing: the last setting holds. A writer should replace it atomically
//! (write another file, then rename it). `scripts/auto-brake.sh` writes it from what the machine is
//! doing; `scripts/brake.py` is the same reader in Python.

use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};

/// The thermal brake (`SKY_DUTY` in (0, 1], the share of the time the training loop works): before
/// each step, the loop rests in proportion to the work done since its last rest.
///
/// The rests are taken out of the run clock, so that `elapsed_secs`, a `--minutes` budget and the
/// schedule that follows it only ever see working time.
///
/// `SKY_COOL=45:15` adds long rests: 45 minutes of work, then 15 of complete rest. The cycle follows
/// the system clock, so that it holds across the short steps of a stage, each its own process.
///
/// `SKY_BRAKE_FILE` names a file that sets both while the run goes on, with no restart: one line,
/// `<duty> [work:rest]` (`0.85 50:10`, or `0.5` alone for no long rests). It is read again every
/// few seconds and replaces `SKY_DUTY` and `SKY_COOL`; a missing or unreadable file changes
/// nothing. `scripts/auto-brake.sh` writes it from what the machine is doing.
pub struct Brake {
    rest_per_busy: f64,
    busy_since: Instant,
    /// Seconds of work and of complete rest in each cycle (`SKY_COOL`).
    cool: Option<(u64, u64)>,
    /// `SKY_BRAKE_FILE` and when it was last read.
    file: Option<(PathBuf, Instant)>,
}

/// Seconds between two reads of `SKY_BRAKE_FILE`.
const BRAKE_FILE_PERIOD: Duration = Duration::from_secs(5);

/// The duty and the long rests of a `SKY_BRAKE_FILE` line; `None` when it is not valid.
fn parse_brake_file(text: &str) -> Option<(f64, Option<(u64, u64)>)> {
    let mut words = text.split_whitespace();
    let duty: f64 = words.next()?.parse().ok()?;
    if !(duty > 0.0 && duty <= 1.0) {
        return None;
    }
    let cool = match words.next() {
        None => None,
        Some(w) => Some(parse_cool(w).ok()?),
    };
    words.next().is_none().then_some((duty, cool))
}

/// `work:rest`, in minutes, as seconds.
fn parse_cool(v: &str) -> Result<(u64, u64)> {
    let secs = |m: &str| -> Option<u64> {
        let m: f64 = m.parse().ok()?;
        (m > 0.0 && m.is_finite()).then(|| (m * 60.0).round() as u64)
    };
    v.split_once(':')
        .and_then(|(w, r)| Some((secs(w)?, secs(r)?)))
        .with_context(|| format!("SKY_COOL must be work:rest in minutes, as 45:15, not {v}"))
}

/// Seconds left in the rest part of the cycle at `now`, 0 in its work part.
fn cool_wait(now: u64, (work, rest): (u64, u64)) -> u64 {
    let at = now % (work + rest);
    if at < work {
        0
    } else {
        work + rest - at
    }
}

impl Brake {
    /// The brake of `SKY_DUTY` (default 1: no short rests), `SKY_COOL` (default: no long rests)
    /// and `SKY_BRAKE_FILE` (read at once when it is set); an unset or empty variable is ignored.
    /// Fails on a `SKY_DUTY` that is not a number in (0, 1] or a `SKY_COOL` that is not
    /// `work:rest` in minutes.
    pub fn from_env() -> Result<Self> {
        let mut brake = match std::env::var("SKY_DUTY") {
            Ok(v) if !v.is_empty() => Self::new(
                v.parse()
                    .with_context(|| format!("SKY_DUTY is not a number: {v}"))?,
            ),
            _ => Self::new(1.0),
        }?;
        if let Ok(v) = std::env::var("SKY_COOL") {
            if !v.is_empty() {
                brake.cool = Some(parse_cool(&v)?);
            }
        }
        if let Ok(v) = std::env::var("SKY_BRAKE_FILE") {
            if !v.is_empty() {
                brake.file = Some((v.into(), Instant::now()));
                brake.read_file();
            }
        }
        Ok(brake)
    }

    /// Takes the duty and the long rests from `SKY_BRAKE_FILE`, when it holds a valid line.
    fn read_file(&mut self) {
        let Some((path, read_at)) = &mut self.file else {
            return;
        };
        *read_at = Instant::now();
        let Some((duty, cool)) = std::fs::read_to_string(&*path)
            .ok()
            .and_then(|t| parse_brake_file(&t))
        else {
            return;
        };
        let rest_per_busy = 1.0 / duty - 1.0;
        if rest_per_busy != self.rest_per_busy || cool != self.cool {
            let rests = cool.map_or("no long rests".to_string(), |(w, r)| {
                format!("{} min of work, {} of rest", w / 60, r / 60)
            });
            println!("brake: duty {duty}, {rests} ({})", path.display());
            self.rest_per_busy = rest_per_busy;
            self.cool = cool;
        }
    }

    /// [`Brake::read_file`], at most once every [`BRAKE_FILE_PERIOD`].
    fn refresh(&mut self) {
        if matches!(&self.file, Some((_, at)) if at.elapsed() >= BRAKE_FILE_PERIOD) {
            self.read_file();
        }
    }

    /// A brake of `duty` in (0, 1], with no long rests and no brake file.
    pub fn new(duty: f64) -> Result<Self> {
        if !(duty > 0.0 && duty <= 1.0) {
            anyhow::bail!("SKY_DUTY must be in (0, 1], not {duty}");
        }
        Ok(Self {
            rest_per_busy: 1.0 / duty - 1.0,
            busy_since: Instant::now(),
            cool: None,
            file: None,
        })
    }

    /// Rests, then moves `clock` (the start of the run clock) forward by the time rested.
    ///
    /// Call it once per step. The short rest is `(1 / duty - 1)` times the time since the previous
    /// rest; then, while the system clock is in the rest part of the `work:rest` cycle, it waits
    /// for the work part (in naps of 5 seconds with a brake file, which is read again between
    /// them). At duty 1 with no long rests it returns at once.
    pub fn rest(&mut self, clock: &mut Instant) {
        self.refresh();
        if self.rest_per_busy == 0.0 && self.cool.is_none() {
            return;
        }
        let started = Instant::now();
        std::thread::sleep(self.busy_since.elapsed().mul_f64(self.rest_per_busy));
        let now = || {
            let t = SystemTime::now().duration_since(UNIX_EPOCH);
            t.unwrap_or_default().as_secs()
        };
        let wait = |cool: Option<(u64, u64)>| cool.map_or(0, |cycle| cool_wait(now(), cycle));
        if wait(self.cool) > 0 {
            println!("cooling down for {} s (SKY_COOL)", wait(self.cool));
        }
        // A loop, because a Mac put to sleep can wake up in another rest. With a brake file the
        // rest is taken in short naps, so that a new line can end it or shorten it.
        while wait(self.cool) > 0 {
            let nap = match self.file {
                Some(_) => wait(self.cool).min(BRAKE_FILE_PERIOD.as_secs()),
                None => wait(self.cool),
            };
            std::thread::sleep(Duration::from_secs(nap));
            self.refresh();
        }
        *clock += started.elapsed();
        self.busy_since = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::{cool_wait, parse_brake_file, parse_cool, Brake};
    use std::time::{Duration, Instant};

    #[test]
    fn the_cycle_works_then_rests() {
        let cycle = (45 * 60, 15 * 60);
        assert_eq!(cool_wait(0, cycle), 0);
        assert_eq!(cool_wait(45 * 60 - 1, cycle), 0);
        assert_eq!(cool_wait(45 * 60, cycle), 15 * 60);
        assert_eq!(cool_wait(59 * 60, cycle), 60);
        assert_eq!(cool_wait(60 * 60, cycle), 0);
    }

    #[test]
    fn parses_work_and_rest_in_minutes() {
        assert_eq!(parse_cool("45:15").unwrap(), (2700, 900));
        assert_eq!(parse_cool("0.5:0.25").unwrap(), (30, 15));
        for bad in ["45", "0:15", "45:0", "a:b", "-1:5", "inf:5"] {
            assert!(parse_cool(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn refuses_a_duty_outside_0_1() {
        for d in [0.0, -0.5, 1.5, f64::NAN] {
            assert!(Brake::new(d).is_err(), "{d}");
        }
        assert!(Brake::new(1.0).is_ok());
    }

    #[test]
    fn the_rests_stay_out_of_the_run_clock() {
        let mut clock = Instant::now();
        let mut brake = Brake::new(0.5).unwrap();
        let wall = Instant::now();
        for _ in 0..3 {
            std::thread::sleep(Duration::from_millis(40));
            brake.rest(&mut clock);
        }
        let (wall, worked) = (wall.elapsed(), clock.elapsed());
        assert!(wall >= Duration::from_millis(240), "rested: {wall:?}");
        assert!(
            worked < wall.mul_f64(0.75),
            "clock moved: {worked:?} of {wall:?}"
        );
    }

    #[test]
    fn full_duty_does_not_rest() {
        let start = Instant::now();
        let mut clock = start;
        let mut brake = Brake::new(1.0).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        brake.rest(&mut clock);
        assert_eq!(clock, start);
    }

    #[test]
    fn parses_a_brake_file_line() {
        assert_eq!(
            parse_brake_file("0.85 50:10\n"),
            Some((0.85, Some((3000, 600))))
        );
        assert_eq!(parse_brake_file("0.5"), Some((0.5, None)));
        assert_eq!(
            parse_brake_file(" 1 45:15 "),
            Some((1.0, Some((2700, 900))))
        );
        for bad in ["", "fast", "0", "1.5", "0.5 45", "0.5 45:15 more"] {
            assert_eq!(parse_brake_file(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_brake_file_changes_the_duty_of_a_running_brake() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brake");
        let mut brake = Brake::new(1.0).unwrap();
        brake.file = Some((path.clone(), Instant::now()));
        // no file yet: nothing changes
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (0.0, None));
        std::fs::write(&path, "0.5 45:15\n").unwrap();
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (1.0, Some((2700, 900))));
        // a line that is not valid keeps the last setting
        std::fs::write(&path, "soon").unwrap();
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (1.0, Some((2700, 900))));
        std::fs::write(&path, "1").unwrap();
        brake.read_file();
        assert_eq!((brake.rest_per_busy, brake.cool), (0.0, None));
    }
}
