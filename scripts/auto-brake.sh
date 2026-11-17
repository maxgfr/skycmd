#!/usr/bin/env bash
# Sets the thermal brake of a running training from what the Mac is doing, with no restart:
# sky-train reads SKY_BRAKE_FILE again every few seconds (crates/sky-train/src/brake.rs), and so do
# scripts/brake.py and experiments/lfm3-s1s2/s1s2.py. The full guide is docs/idle-training.md; the
# usual way to start it is scripts/train-when-idle.sh, which starts and stops it around a command.
#
#   scripts/auto-brake.sh [runs/brake]        # runs until killed; start it next to the training
#   scripts/auto-brake.sh --once [runs/brake] # one pass: every probe reading and the decision, on
#                                             # stdout; writes nothing (for diagnosis)
#   scripts/auto-brake.sh --help
#
# The brake file holds one line, `<duty> [<work>:<rest>]`: the share of the time the training works,
# in (0, 1], and optional long rests in minutes on the wall clock (`0.85 50:10`). It is replaced
# atomically (written to <file>.tmp, then renamed), only when the value changes.
#
# Modes, checked in this order at every tick (the first that holds wins):
#   too hot       HOT (default: IN_USE) while macOS reports thermal pressure (`pmset -g therm`, or
#                 a thermal state of "serious" or worse, read by scripts/thermal-state.swift,
#                 built once into <file>.thermal when `swiftc` is there)
#   on battery    HOT, whatever else holds
#   in use        IN_USE="0.5 45:15"   an application other than `caffeinate` keeps the display
#                                      awake (a video, a call), or a key or the pointer moved in the
#                                      last IDLE_MIN minutes (10)
#   night         FULL="0.95 55:5"     no input for FULL_IDLE_MIN minutes (30), between the hours
#                                      FULL_HOURS (23-8, from 23:00 to 08:00; 0-24 is always), with
#                                      the other programs using less than OTHER_CPU percent of one
#                                      core (60; twice that once the night mode is on, so that a
#                                      short burst does not switch it off and on): the owner sleeps
#                                      and nothing else works. FULL="" turns this mode off
#   left alone    AWAY="0.85 50:10"    none of the above (the pre-registrations' "two modes")
#
#   Owner away for a day, 0.85 whatever the input and the hour, 0.5 if the machine heats:
#     IN_USE="0.85 50:10" HOT="0.5 45:15" FULL="" scripts/auto-brake.sh
#
# A reading that could not be made takes the careful side: an unknown idle time counts as in use
# ("idle time unknown: treated as in use"), an unknown hour is not night, and an unknown CPU reading
# of the other programs is not quiet (no night mode).
#
# Out of memory: when a process named TRAIN_PROC holds MEM_STOP_GB of memory or more (13; 0 turns the
# guard off) MEM_STOP_N readings in a row (4, two minutes), TERM goes to the TRAIN_PROC processes
# over the limit at that tick, and only to them (an evaluation or a probe under the limit goes on);
# the log names their pids. A run that swaps does almost no work and fills the disk. It resumes from
# its last checkpoint at the next launch. The loading of the decision sets peaks for about a minute
# and does not count. The memory is read with one `top` call; every unit it prints (K, M, G, with an
# optional + or -) is converted, then truncated to whole GB of 1024 MB (13500M reads as 13).
#
# Other settings: PERIOD (30) seconds between two ticks; TRAIN_PROC (sky-train) the trainer's
# process name for the memory guard; TRAIN_PROCS_RE (`(sky-train|sky-arena|sky-eval|[Pp]ython[0-9.]*)$`)
# an awk regular expression over the whole command path of `ps -o comm=` (spaces included): the
# processes it matches are not "other programs". It matches `Python` too, the framework build that
# Homebrew installs (`.../MacOS/Python`): without it a Python trainer counted as another program and
# kept the night mode off.
#
# Every change is a line in <file>.log, with the reason. The pre-registered budgets count working
# time only, so the brake changes how long a run takes, not what it computes.
#
# One daemon per brake file: it holds <file>.pid while it runs (removed on exit, with <file>.tmp).
# A second start on the same file exits with status 3; a pid file whose process is gone, or is not
# auto-brake.sh, is taken over. macOS only (ioreg, pmset, top -l): elsewhere it exits with status 4;
# write the brake file with your own tool (docs/idle-training.md, "The brake-file protocol").
# The settings are checked at start: whole numbers for the minutes, OTHER_CPU and the memory guard,
# `<from>-<to>` for FULL_HOURS, brake lines the trainers accept for IN_USE, AWAY, HOT and FULL.
# Exit status: 0 stopped by TERM or INT (or --once, --help), 2 usage or an invalid setting,
# 3 already running for this file, 4 not macOS.
#
# Sourcing this file defines the functions and the settings and runs nothing
# (scripts/test-auto-brake.sh does that); AUTO_BRAKE_OS replaces `uname -s`, for those tests.

# The settings, from the environment (called again by the tests after they change it).
ab_settings() {
  in_use=${IN_USE:-0.5 45:15}
  away=${AWAY:-0.85 50:10}
  full=${FULL-0.95 55:5}
  hot_brake=${HOT:-$in_use}
  full_idle_min=${FULL_IDLE_MIN:-30}
  full_hours=${FULL_HOURS:-23-8}
  other_cpu=${OTHER_CPU:-60}
  idle_min=${IDLE_MIN:-10}
  mem_stop_gb=${MEM_STOP_GB:-13}
  mem_stop_n=${MEM_STOP_N:-4}
  period=${PERIOD:-30}
  train_proc=${TRAIN_PROC:-sky-train}
  train_procs_re=${TRAIN_PROCS_RE:-(sky-train|sky-arena|sky-eval|[Pp]ython[0-9.]*)\$}
}
ab_settings
ab_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ----------------------------------------------------------------------------- probes
# Each prints a reading and never fails the caller. A probe that cannot run prints nothing (or 0
# for a yes/no reading), and `decide` takes the careful side for it.

# Seconds since the last key press or pointer move (empty when ioreg gives nothing).
idle_secs() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ { print int($NF / 1000000000); exit }'; }
# True when an application keeps the display awake (a video, a call, a presentation). `caffeinate`
# does not count: command-line tools hold it for hours with nobody at the screen.
display_held() {
  pmset -g assertions | awk '
    /pid [0-9]+\(.*PreventUserIdleDisplaySleep named/ && !/\(caffeinate\)/ { held = 1 }
    END { exit !held }'
}
on_battery() { pmset -g batt | head -n 1 | grep -q 'Battery Power'; }
# The thermal state of scripts/thermal-state.swift (0 when the helper could not be built).
thermal_state() {
  local v=0
  [ -x "$therm_bin" ] && v=$("$therm_bin" 2>/dev/null) || v=0
  case $v in [0-3]) echo "$v" ;; *) echo 0 ;; esac
}
# True when `pmset -g therm` reports a speed limit or a thermal warning.
pmset_hot() {
  pmset -g therm 2>/dev/null | awk -F'= *' '
    /CPU_Speed_Limit/ && $2 + 0 < 100 { hot = 1 }
    /Thermal Warning Level/ && $2 + 0 > 0 { hot = 1 }
    END { exit !hot }'
}
# Percent of one core used by the programs that TRAIN_PROCS_RE does not match, matched against the
# whole command path (everything after the %cpu field, spaces included). Printed exactly, so that
# `decide` compares the number the daemon always compared; nothing when `ps` gives nothing.
others_cpu() {
  LC_ALL=C ps -A -o %cpu=,comm= | AB_RE=$train_procs_re LC_ALL=C awk '
    NF { n++; cpu = $1; sub(/^[ \t]*[^ \t]+[ \t]+/, ""); if ($0 !~ ENVIRON["AB_RE"]) sum += cpu }
    END { if (n) printf "%.17g\n", sum + 0 }'
}
# Memory of the TRAIN_PROC processes from one `top` call: prints `<GB of the largest> [<pid>...]`,
# the pids being those that hold MEM_STOP_GB or more. `top` prints K, M or G (1G = 1024M), with an
# optional trailing + or - and a decimal point; each value is converted, then truncated to whole GB
# (13500M is 13). 0 when none runs, or when one ends during the reading (`top` leaves it out).
train_gb() {
  local pids p args=()
  pids=$(pgrep -x "$train_proc") || true
  [ -n "$pids" ] || { echo 0; return; }
  for p in $pids; do args+=(-pid "$p"); done
  top -l 1 "${args[@]}" -stats pid,mem 2>/dev/null | LC_ALL=C awk -v lim="$mem_stop_gb" '
    $1 == "PID" { rows = 1; next }
    rows && NF >= 2 && $1 ~ /^[0-9]+$/ {
      v = $2
      sub(/[+-]$/, "", v)
      unit = substr(v, length(v))
      n = substr(v, 1, length(v) - 1)
      if (n !~ /^[0-9]+([.][0-9]+)?$/) next
      if (unit == "B") gb = n / 1073741824
      else if (unit == "K") gb = n / 1048576
      else if (unit == "M") gb = n / 1024
      else if (unit == "G") gb = n + 0
      else if (unit == "T") gb = n * 1024
      else next
      gb = int(gb)
      if (gb > max) max = gb
      if (lim + 0 > 0 && gb >= lim + 0) over = over " " $1
    }
    END { print (max + 0) over }'
}

# Every reading of one tick, into p_* variables. None of them can stop the daemon under `set -e`.
read_probes() {
  p_state=$(thermal_state)
  p_pmset_hot=0
  pmset_hot 2>/dev/null && p_pmset_hot=1
  p_battery=0
  on_battery 2>/dev/null && p_battery=1
  p_display=0
  display_held 2>/dev/null && p_display=1
  p_idle=$(idle_secs 2>/dev/null || true)
  p_hour=$(date +%H 2>/dev/null || true)
  p_others=$(others_cpu 2>/dev/null || true)
  p_current=$(cat "$file" 2>/dev/null || true)
}

# ----------------------------------------------------------------------------- decision
# True when <value> is a whole number of zero or more.
is_count() { case $1 in '' | *[!0-9]*) return 1 ;; esac; }
# True when <hour> (00 to 23) is between the hours of FULL_HOURS (`23-8`: from 23:00 to 08:00,
# `0-24`: always, `9-17`: from 09:00 to 17:00).
night() { # <hour>
  local h from=${full_hours%-*} to=${full_hours#*-}
  case $1 in '' | *[!0-9]*) return 1 ;; esac
  h=$((10#$1))
  if [ "$from" -le "$to" ]; then
    [ "$h" -ge "$from" ] && [ "$h" -lt "$to" ]
  else
    [ "$h" -ge "$from" ] || [ "$h" -lt "$to" ]
  fi
}
# True when the other programs use less than OTHER_CPU percent of one core (twice that when the
# brake file already holds the night mode, so that a short burst does not switch it off and on).
# A reading that is not a number (`ps` gave nothing) is not quiet: no night mode on it.
others_quiet() { # <percent of one core> <current brake>
  local max=$other_cpu re='^[0-9]+([.][0-9]*)?([eE][-+]?[0-9]+)?$'
  [[ $1 =~ $re ]] || return 1
  [ "$2" = "$full" ] && max=$((other_cpu * 2))
  LC_ALL=C awk -v sum="$1" -v max="$max" 'BEGIN { exit !(sum + 0 < max) }'
}
# The brake for these readings, and why: prints `<brake>\t<reason>`. Pure: it reads only its
# arguments and the settings. An idle time that could not be read counts as in use: the daemon
# never speeds a training up on a reading it could not make.
#   decide <pmset hot 0|1> <thermal state 0-3> <on battery 0|1> <display held 0|1> <idle seconds>
#          <hour> <other programs, percent of one core> <current brake>
decide() {
  local pmset_hot=$1 state=$2 battery=$3 display=$4 idle=$5 hour=$6 others=$7 current=$8
  local t=$'\t'
  if [ "$state" -ge 2 ] 2>/dev/null || [ "$pmset_hot" = 1 ]; then
    echo "$hot_brake${t}thermal pressure (state $state)"
  elif [ "$battery" = 1 ]; then
    echo "$hot_brake${t}on battery"
  elif [ "$display" = 1 ]; then
    echo "$in_use${t}a video or a call keeps the display awake"
  elif ! is_count "$idle"; then
    echo "$in_use${t}idle time unknown: treated as in use"
  elif [ "$idle" -lt $((idle_min * 60)) ] 2>/dev/null; then
    echo "$in_use${t}the Mac is in use"
  elif [ -n "$full" ] && [ "$idle" -ge $((full_idle_min * 60)) ] 2>/dev/null &&
    night "$hour" && others_quiet "$others" "$current"; then
    echo "$full${t}night: no input for $full_idle_min min, nothing else works"
  else
    echo "$away${t}no input for $idle_min min"
  fi
}

# ----------------------------------------------------------------------------- effects
log_line() { echo "$(date '+%F %T') $*" >>"$log" 2>/dev/null || true; }

set_brake() { # <value> <reason>
  [ "$(cat "$file" 2>/dev/null)" = "$1" ] && return 0
  if { echo "$1" >"$file.tmp" && mv "$file.tmp" "$file"; } 2>/dev/null; then
    log_line "$1 ($2)"
  else
    # A full disk or a removed folder: the next tick tries again (this used to end the daemon).
    rm -f "$file.tmp" 2>/dev/null || true
    log_line "could not write $file: $1 ($2) not applied"
  fi
}

# One tick of the memory guard. After MEM_STOP_N readings in a row with a TRAIN_PROC process at
# MEM_STOP_GB or more, TERM goes to the processes over the limit at that tick, and only to them: an
# evaluation or a probe of another run, under the limit, goes on. ab_over counts the readings.
ab_over=0
mem_guard() {
  local out gb pids
  # A training that ends between the two commands of train_gb makes `top` fail: that reading is 0,
  # and the daemon goes on (it used to end there, silently, under `set -e`).
  out=$(train_gb) || out=0
  gb=${out%% *}
  pids=${out#"$gb"}
  pids=${pids# }
  is_count "$gb" || gb=0
  if [ "$gb" -ge "$mem_stop_gb" ] && [ -n "$pids" ]; then
    ab_over=$((ab_over + 1))
    if [ "$ab_over" -ge "$mem_stop_n" ]; then
      log_line "$train_proc holds $gb GB (limit $mem_stop_gb): training stopped (TERM to pid $pids)"
      # shellcheck disable=SC2086 # one word per pid
      kill -TERM $pids 2>/dev/null || true
      ab_over=0
    fi
  else
    ab_over=0
  fi
}

# True when <pid> is a running auto-brake.sh.
is_auto_brake() { # <pid>
  local cmd re='(^|[ /])auto-brake\.sh( |$)'
  case $1 in '' | *[!0-9]*) return 1 ;; esac
  cmd=$(ps -p "$1" -o command= 2>/dev/null) || return 1
  [[ $cmd =~ $re ]]
}
# The pid of the auto-brake.sh that serves <brake file> (from <file>.pid), or nothing.
daemon_pid() { # <brake file>
  local pid
  pid=$(cat "$1.pid" 2>/dev/null) || return 1
  is_auto_brake "$pid" || return 1
  echo "$pid"
}
# Writes <pid> into <pid file>, atomically and only if no live auto-brake.sh holds it. Returns 3
# (and prints its pid) when one does; a stale file is replaced.
claim_pidfile() { # <pid file> <pid>
  local tmp=$1.$2.tmp holder try
  echo "$2" >"$tmp" || return 1
  for try in 1 2; do
    # A hard link is created atomically, and only when the name is free.
    if ln "$tmp" "$1" 2>/dev/null; then
      rm -f "$tmp"
      return 0
    fi
    # A file system without hard links: an atomic rename, when the name is free.
    if [ ! -e "$1" ] && mv -f "$tmp" "$1" 2>/dev/null; then
      return 0
    fi
    holder=$(cat "$1" 2>/dev/null || true)
    if [ "$holder" != "$2" ] && is_auto_brake "$holder"; then
      rm -f "$tmp"
      echo "$holder"
      return 3
    fi
    [ "$try" = 1 ] && rm -f "$1"
  done
  rm -f "$tmp"
  return 1
}
# Removes <pid file> if it still holds <pid>.
release_pidfile() { # <pid file> <pid>
  [ "$(cat "$1" 2>/dev/null || true)" = "$2" ] && rm -f "$1"
  return 0
}

ab_usage() {
  cat <<'EOF'
Usage: scripts/auto-brake.sh [brake-file]          run until stopped (default brake file runs/brake)
       scripts/auto-brake.sh --once [brake-file]   one pass: the probe readings and the decision; writes nothing
       scripts/auto-brake.sh --help

Writes `<duty> [<work>:<rest>]` into the brake file from what the Mac is doing: too hot or on battery
(HOT), in use (IN_USE), at night with nothing else working (FULL), left alone (AWAY). Settings, by
environment, with their defaults:
  IN_USE="0.5 45:15"  AWAY="0.85 50:10"  FULL="0.95 55:5" (FULL="" turns the night mode off)
  HOT=$IN_USE  IDLE_MIN=10  FULL_IDLE_MIN=30  FULL_HOURS=23-8  OTHER_CPU=60
  MEM_STOP_GB=13 (0: off)  MEM_STOP_N=4  PERIOD=30  TRAIN_PROC=sky-train
  TRAIN_PROCS_RE='(sky-train|sky-arena|sky-eval|[Pp]ython[0-9.]*)$'
One daemon per brake file (<file>.pid). macOS only. Exit status: 0 stopped, 2 usage, 3 already
running for this file, 4 not macOS. Guide: docs/idle-training.md.
EOF
}

ab_require_macos() { # <program name>
  [ "${AUTO_BRAKE_OS:-$(uname -s)}" = Darwin ] && return 0
  cat >&2 <<EOF
$1: the brake daemon runs on macOS only. Its probes use ioreg (seconds since the last key or pointer
input), pmset (battery, display assertions, thermal warnings) and top -l (memory of the trainer).
On another system, write the brake file with your own tool: its format is specified in
docs/idle-training.md, "The brake-file protocol".
EOF
  exit 4
}

# --once: every reading, the decision, and what a tick would do; nothing is written or built.
ab_once() {
  local out value reason pid gb hours others over
  read_probes
  out=$(decide "$p_pmset_hot" "$p_state" "$p_battery" "$p_display" "$p_idle" "$p_hour" \
    "$p_others" "$p_current")
  value=${out%%$'\t'*}
  reason=${out#*$'\t'}
  pid=$(daemon_pid "$file" || true)
  printf '%-16s %s\n' "brake file" "$file (holds: ${p_current:-nothing})"
  printf '%-16s %s\n' "daemon" "${pid:+pid $pid}${pid:-none for this file}"
  printf '%-16s %s\n' "thermal state" "$p_state (0 nominal .. 3 critical; helper $therm_bin: $([ -x "$therm_bin" ] && echo present || echo absent))"
  printf '%-16s %s\n' "pmset therm" "$p_pmset_hot (1: a speed limit or a thermal warning)"
  printf '%-16s %s\n' "on battery" "$p_battery"
  printf '%-16s %s\n' "display held" "$p_display (by an application other than caffeinate)"
  printf '%-16s %s\n' "idle" "${p_idle:-unknown} s (in use below $((idle_min * 60)) s; night from $((full_idle_min * 60)) s)"
  hours="night hours $full_hours"
  [ -n "$full" ] || hours='night mode off: FULL=""'
  printf '%-16s %s\n' "hour" "${p_hour:-unknown} ($hours)"
  if [ -n "$p_others" ]; then
    others="$(LC_ALL=C awk -v s="$p_others" 'BEGIN { printf "%.1f", s }') % of one core"
  else
    others="unknown (not quiet: no night mode)"
  fi
  printf '%-16s %s\n' "other programs" "$others (quiet below $other_cpu, $((other_cpu * 2)) once the night mode is on)"
  if [ "$mem_stop_gb" -gt 0 ] 2>/dev/null; then
    gb=$(train_gb 2>/dev/null) || gb=0
    over=${gb#"${gb%% *}"}
    printf '%-16s %s\n' "trainer memory" "${gb%% *} GB, largest $train_proc${over:+; over the limit: pid$over} (stopped at $mem_stop_gb GB, $mem_stop_n readings in a row)"
  else
    printf '%-16s %s\n' "trainer memory" "not watched (MEM_STOP_GB=0)"
  fi
  printf '%-16s %s\n' "decision" "$value ($reason)"
  if [ "$p_current" = "$value" ]; then
    printf '%-16s %s\n' "a tick would" "change nothing"
  else
    printf '%-16s %s\n' "a tick would" "write \"$value\" and log it"
  fi
}

# Settings that the shell computes with must be whole numbers (a fraction would end the daemon in
# the middle of a run), and the brake values must be lines that the trainers accept.
ab_check_settings() {
  local name value
  for name in IDLE_MIN FULL_IDLE_MIN OTHER_CPU MEM_STOP_GB MEM_STOP_N; do
    value=$(ab_setting_value "$name")
    case $value in '' | *[!0-9]*)
      echo "auto-brake.sh: $name must be a whole number, not '$value'" >&2
      exit 2
      ;;
    esac
  done
  case $full_hours in
    [0-9]-[0-9] | [0-9]-[0-9][0-9] | [0-9][0-9]-[0-9] | [0-9][0-9]-[0-9][0-9]) ;;
    *)
      echo "auto-brake.sh: FULL_HOURS must be <from>-<to> in hours, as 23-8, not '$full_hours'" >&2
      exit 2
      ;;
  esac
  for name in IN_USE AWAY HOT FULL; do
    value=$(ab_setting_value "$name")
    [ "$name" = FULL ] && [ -z "$value" ] && continue
    valid_brake_line "$value" || {
      echo "auto-brake.sh: $name must be '<duty> [<work>:<rest>]' (duty in (0, 1], minutes" \
        "above 0), as '0.85 50:10', not '$value'" >&2
      exit 2
    }
  done
}
ab_setting_value() { # <name>: the value in use
  case $1 in
    IDLE_MIN) echo "$idle_min" ;;
    FULL_IDLE_MIN) echo "$full_idle_min" ;;
    OTHER_CPU) echo "$other_cpu" ;;
    MEM_STOP_GB) echo "$mem_stop_gb" ;;
    MEM_STOP_N) echo "$mem_stop_n" ;;
    IN_USE) echo "$in_use" ;;
    AWAY) echo "$away" ;;
    HOT) echo "$hot_brake" ;;
    FULL) echo "$full" ;;
  esac
}
# True when <line> is a brake-file line that the trainers accept (the usual forms of a number).
valid_brake_line() { # <line>
  LC_ALL=C awk -v line="$1" 'BEGIN {
    num = "^[+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?$"
    n = split(line, w, " ")
    if (n < 1 || n > 2 || w[1] !~ num || !(w[1] + 0 > 0 && w[1] + 0 <= 1)) exit 1
    if (n == 2) {
      if (split(w[2], c, ":") != 2 || c[1] !~ num || c[2] !~ num) exit 1
      # the trainers round each part to whole seconds: both must stay above 0
      if (int(c[1] * 60 + 0.5) < 1 || int(c[2] * 60 + 0.5) < 1) exit 1
    }
  }'
}

# On exit: the pid file (when it is still ours), the temporary brake file, the pending sleep.
ab_cleanup() {
  release_pidfile "$file.pid" "$ab_pid"
  rm -f "$file.tmp"
  [ -z "${ab_sleeper:-}" ] || kill "$ab_sleeper" 2>/dev/null || true
}

ab_main() {
  local once=0 out
  case ${1:-} in
    -h | --help) ab_usage; exit 0 ;;
    --once) once=1; shift ;;
    -*) ab_usage >&2; exit 2 ;;
  esac
  [ $# -le 1 ] || { ab_usage >&2; exit 2; }
  file=${1:-runs/brake}
  log=$file.log
  therm_bin=$file.thermal
  ab_require_macos auto-brake.sh
  ab_check_settings
  if [ "$once" = 1 ]; then
    ab_once
    exit 0
  fi

  mkdir -p "$(dirname "$file")"
  local holder status=0
  ab_pid=$$
  holder=$(claim_pidfile "$file.pid" "$ab_pid") || status=$?
  if [ "$status" = 3 ]; then
    echo "auto-brake.sh: already running for $file (pid $holder); stop it first, or let it serve" \
      "this training" >&2
    exit 3
  elif [ "$status" != 0 ]; then
    echo "auto-brake.sh: cannot write $file.pid" >&2
    exit 1
  fi
  ab_sleeper=
  trap ab_cleanup EXIT
  trap 'exit 0' TERM INT

  if [ ! -x "$therm_bin" ] && command -v swiftc >/dev/null; then
    nice -n 19 swiftc -O -o "$therm_bin" "$ab_dir/thermal-state.swift" 2>/dev/null ||
      log_line "thermal-state helper not built: pmset only"
  fi
  while :; do
    read_probes
    out=$(decide "$p_pmset_hot" "$p_state" "$p_battery" "$p_display" "$p_idle" "$p_hour" \
      "$p_others" "$p_current") || out=
    [ -z "$out" ] || set_brake "${out%%$'\t'*}" "${out#*$'\t'}"
    if [ "$mem_stop_gb" -gt 0 ] 2>/dev/null; then
      mem_guard
    fi
    # In the background, so that TERM ends the daemon at once and not at the end of the sleep.
    sleep "$period" &
    ab_sleeper=$!
    wait "$ab_sleeper" || true
    ab_sleeper=
  done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  ab_main "$@"
fi
