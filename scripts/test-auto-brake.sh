#!/usr/bin/env bash
# Tests of scripts/auto-brake.sh and scripts/train-when-idle.sh: bash scripts/test-auto-brake.sh
#
# No dependency beyond bash and the base tools; runs on Linux and macOS; one line per case, and a
# non-zero exit status when a case fails. The daemons it starts serve a brake file in a temporary
# folder, with the memory guard off (MEM_STOP_GB=0) and a trainer name that matches no process, so
# they cannot touch a training that runs on the machine. On Linux they run with AUTO_BRAKE_OS=Darwin:
# their probes find no ioreg or pmset and read as quiet, which is all the tests need.
# shellcheck disable=SC2016,SC2153,SC2329 # single quotes for child shells; variables set per call; stubs called by the sourced functions
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source-path=SCRIPTDIR source=auto-brake.sh
. "$here/auto-brake.sh"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/test-auto-brake.XXXXXX")
tmp=$(cd "$tmp" && pwd)
bg=()
finish() {
  local p
  for p in ${bg[@]+"${bg[@]}"}; do kill "$p" 2>/dev/null || true; done
  rm -rf "$tmp"
}
trap finish EXIT

n=0
fails=0
check() { # <case> <expected> <actual>
  n=$((n + 1))
  if [ "$2" = "$3" ]; then
    echo "ok   $1"
  else
    echo "FAIL $1: expected [$2], got [$3]"
    fails=$((fails + 1))
  fi
}
T=$'\t'
settings() { # [NAME=VALUE...]: the default settings, changed by the arguments
  unset IN_USE AWAY FULL HOT FULL_IDLE_MIN FULL_HOURS OTHER_CPU IDLE_MIN MEM_STOP_GB MEM_STOP_N \
    PERIOD TRAIN_PROC TRAIN_PROCS_RE
  local kv
  for kv in "$@"; do export "${kv%%=*}=${kv#*=}"; done
  ab_settings
}
wait_for() { # <shell condition>: true once it holds, within 10 s
  local i=0
  while [ "$i" -lt 50 ]; do
    eval "$1" && return 0
    sleep 0.2
    i=$((i + 1))
  done
  return 1
}
# Safe settings for every real daemon of these tests.
daemon_env=(AUTO_BRAKE_OS=Darwin MEM_STOP_GB=0 TRAIN_PROC=no-such-trainer-of-the-tests PERIOD=1)

# ----------------------------------------------------------------------------- decide
# decide <pmset hot> <thermal state> <on battery> <display held> <idle s> <hour> <others %> <current>
IN=0.5\ 45:15
AWAY_=0.85\ 50:10
NIGHT=0.95\ 55:5
settings
check "hot (pmset) beats battery, display, input and night" "$IN${T}thermal pressure (state 0)" \
  "$(decide 1 0 1 1 0 3 0 "")"
check "hot (thermal state 2)" "$IN${T}thermal pressure (state 2)" "$(decide 0 2 0 0 99999 3 0 "")"
check "hot (thermal state 3)" "$IN${T}thermal pressure (state 3)" "$(decide 0 3 0 0 99999 3 0 "")"
check "thermal state 1 is not hot" "$AWAY_${T}no input for 10 min" "$(decide 0 1 0 0 600 12 0 "")"
check "on battery beats display and input" "$IN${T}on battery" "$(decide 0 0 1 1 0 3 0 "")"
check "display held beats input and night" "$IN${T}a video or a call keeps the display awake" \
  "$(decide 0 0 0 1 99999 3 0 "")"
check "in use: input in the last 10 min" "$IN${T}the Mac is in use" "$(decide 0 0 0 0 599 3 0 "")"
check "left alone: 10 min with no input, by day" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 600 12 0 "")"
check "night: 30 min idle, at night, others quiet" \
  "$NIGHT${T}night: no input for 30 min, nothing else works" "$(decide 0 0 0 0 1800 3 10 "")"
check "night needs 30 min of idle" "$AWAY_${T}no input for 10 min" "$(decide 0 0 0 0 1799 3 10 "")"
check "night needs the night hours (08:00)" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 1800 8 10 "")"
check "night hours start at 23:00" "$NIGHT${T}night: no input for 30 min, nothing else works" \
  "$(decide 0 0 0 0 1800 23 10 "")"
check "night hours: 22:00 is day" "$AWAY_${T}no input for 10 min" "$(decide 0 0 0 0 1800 22 10 "")"
check "night needs others below 60 % (60 is not)" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 1800 3 60 "")"
check "night: others at 59.9 % are quiet" "$NIGHT${T}night: no input for 30 min, nothing else works" \
  "$(decide 0 0 0 0 1800 3 59.9 "")"
check "night: the battery still wins" "$IN${T}on battery" "$(decide 0 0 1 0 1800 3 0 "")"
check "night: a held display still wins" "$IN${T}a video or a call keeps the display awake" \
  "$(decide 0 0 0 1 1800 3 0 "")"
check "hysteresis: in night mode, 100 % is still quiet" \
  "$NIGHT${T}night: no input for 30 min, nothing else works" "$(decide 0 0 0 0 1800 3 100 "$NIGHT")"
check "hysteresis: in night mode, 119.9 % is still quiet" \
  "$NIGHT${T}night: no input for 30 min, nothing else works" \
  "$(decide 0 0 0 0 1800 3 119.9 "$NIGHT")"
check "hysteresis: in night mode, 120 % ends it" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 1800 3 120 "$NIGHT")"
check "hysteresis only from the night mode" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 1800 3 100 "$AWAY_")"
check "an idle time that could not be read counts as in use" \
  "$IN${T}idle time unknown: treated as in use" "$(decide 0 0 0 0 "" 3 0 "")"
check "an idle time that is not a number counts as in use" \
  "$IN${T}idle time unknown: treated as in use" "$(decide 0 0 0 0 "12x" 3 0 "")"
check "an unknown idle time still yields to heat, battery and display" \
  "$IN${T}on battery|$IN${T}a video or a call keeps the display awake" \
  "$(decide 0 0 1 0 "" 3 0 "")|$(decide 0 0 0 1 "" 3 0 "")"
check "an unknown hour is not night" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 1800 "" 0 "")"
check "an unknown CPU reading of the other programs is not quiet" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 1800 3 "" "")"
check "an unknown CPU reading ends the night mode too" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 1800 3 "" "$NIGHT")"

settings FULL=
check "FULL=\"\" turns the night mode off" "$AWAY_${T}no input for 10 min" \
  "$(decide 0 0 0 0 99999 3 0 "")"
settings HOT="0.3 30:30"
check "HOT applies under thermal pressure" "0.3 30:30${T}thermal pressure (state 2)" \
  "$(decide 0 2 0 0 0 12 0 "")"
check "HOT applies on battery" "0.3 30:30${T}on battery" "$(decide 0 0 1 0 0 12 0 "")"
check "HOT does not apply to input" "$IN${T}the Mac is in use" "$(decide 0 0 0 0 0 12 0 "")"
settings IN_USE="0.6 40:20" AWAY="0.9 50:10" FULL="1 55:5"
check "HOT defaults to IN_USE" "0.6 40:20${T}on battery" "$(decide 0 0 1 0 0 12 0 "")"
check "AWAY is the left-alone brake" "0.9 50:10${T}no input for 10 min" \
  "$(decide 0 0 0 0 600 12 0 "")"
check "FULL is the night brake" "1 55:5${T}night: no input for 30 min, nothing else works" \
  "$(decide 0 0 0 0 1800 3 0 "")"
settings IDLE_MIN=1 FULL_IDLE_MIN=45 OTHER_CPU=20 FULL_HOURS=0-24
check "IDLE_MIN=1: in use below 60 s" "$IN${T}the Mac is in use" "$(decide 0 0 0 0 59 12 0 "")"
check "IDLE_MIN=1: left alone from 60 s" "$AWAY_${T}no input for 1 min" \
  "$(decide 0 0 0 0 60 12 0 "")"
check "FULL_IDLE_MIN=45: not night at 2699 s" "$AWAY_${T}no input for 1 min" \
  "$(decide 0 0 0 0 2699 12 0 "")"
check "FULL_IDLE_MIN=45 and 0-24: night at 2700 s, at noon" \
  "$NIGHT${T}night: no input for 45 min, nothing else works" "$(decide 0 0 0 0 2700 12 0 "")"
check "OTHER_CPU=20: 30 % is not quiet" "$AWAY_${T}no input for 1 min" \
  "$(decide 0 0 0 0 2700 12 30 "")"
check "OTHER_CPU=20: 30 % is quiet once in night mode (below 40)" \
  "$NIGHT${T}night: no input for 45 min, nothing else works" "$(decide 0 0 0 0 2700 12 30 "$NIGHT")"

# ----------------------------------------------------------------------------- night
hours() { # <FULL_HOURS> <hour...>: t or f for each hour
  local h out=""
  full_hours=$1
  shift
  for h in "$@"; do
    if night "$h"; then out="${out}t"; else out="${out}f"; fi
  done
  echo "$out"
}
check "night 23-8 across midnight (22 23 00 07 08 12)" "ftttff" "$(hours 23-8 22 23 00 07 08 12)"
check "night 19-8 across midnight (18 19 23 0 7 8)" "fttttf" "$(hours 19-8 18 19 23 0 7 8)"
check "night 0-24 is every hour (0 1 12 23)" "tttt" "$(hours 0-24 0 1 12 23)"
check "night 9-17 the same day (8 9 16 17 23 0)" "fttfff" "$(hours 9-17 8 9 16 17 23 0)"
check "night: an hour that could not be read is not night" "f" "$(hours 0-24 "")"
settings

# ----------------------------------------------------------------------------- probes (parsing)
others_with() { # <ps output>: others_cpu over that output
  ps() { printf '%s\n' "$PS_OUT"; }
  PS_OUT=$1 others_cpu
  unset -f ps
}
ps_out=' 50.0 /usr/bin/foo
 30.0 /Users/x/target/release/sky-train
 20.5 /usr/bin/python3.12
 10.0 /opt/Python.framework/Versions/3.15/Resources/Python.app/Contents/MacOS/Python
  7.5 /Users/x/My Projects/target/release/sky-train
  2.0 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome Helper (Renderer)'
check "others: the trainers, python3.12 and a framework Python are left out" "52" \
  "$(others_with "$ps_out")"
settings TRAIN_PROCS_RE='(sky-train)$'
check "others: TRAIN_PROCS_RE replaces the pattern" "82.5" "$(others_with "$ps_out")"
settings
check "others: a trainer in a path with spaces is still the trainer" "0" \
  "$(others_with '  7.5 /Users/x/My Projects/target/release/sky-train')"
check "others: no ps output reads as unknown" "" "$(others_with "")"

gb_with() { # <pgrep output> <top output>: train_gb over them
  pgrep() { [ -n "$PG_OUT" ] && printf '%s\n' "$PG_OUT"; }
  top() { printf '%s\n' "$TOP_OUT"; }
  PG_OUT=$1 TOP_OUT=$2 train_gb
  unset -f pgrep top
}
top_rows() { printf 'Processes: 700 total\nPhysMem: 15G used (2000M wired), 900M unused.\n\nPID    MEM\n'; printf '%s\n' "$@"; }
check "memory: 2587M is 2 GB" "2" "$(gb_with 11 "$(top_rows '11  2587M')")"
check "memory: 13G is 13 GB, over the limit of 13" "13 11" "$(gb_with 11 "$(top_rows '11  13G')")"
check "memory: 13500M is 13 GB" "13 11" "$(gb_with 11 "$(top_rows '11  13500M')")"
check "memory: 16G+ is 16 GB" "16 11" "$(gb_with 11 "$(top_rows '11  16G+')")"
check "memory: 12G- is 12 GB, under the limit" "12" "$(gb_with 11 "$(top_rows '11  12G-')")"
check "memory: 950K is 0 GB" "0" "$(gb_with 11 "$(top_rows '11  950K')")"
check "memory: 13.9G is 13 GB (truncated)" "13 11" "$(gb_with 11 "$(top_rows '11  13.9G')")"
check "memory: the largest of several, and every pid over the limit" "16 22 33" \
  "$(gb_with $'11\n22\n33\n44' "$(top_rows '11  2587M' '22  16G+' '33  13500M' '44  950K')")"
check "memory: the PhysMem line before the table is not a process" "0" \
  "$(gb_with 11 "$(top_rows '11  900M')")"
check "memory: no trainer reads as 0" "0" "$(gb_with "" "$(top_rows '22  14G')")"
check "memory: a trainer that ended during the reading reads as 0" "0" "$(gb_with 11 "$(top_rows)")"
settings MEM_STOP_GB=20
check "memory: MEM_STOP_GB sets which pids are over" "16" \
  "$(gb_with $'22\n33' "$(top_rows '22  16G' '33  13500M')")"
settings TRAIN_PROC=my-trainer
check "memory: TRAIN_PROC names the process" "my-trainer" \
  "$(pgrep() { echo "$2"; }; top() { :; }; train_gb >/dev/null; pgrep -x "$train_proc")"
settings

# ----------------------------------------------------------------------------- memory guard
# mem_guard with `pgrep`, `top` and `kill` stubbed: the kills are recorded, never sent.
log=$tmp/guard.log
guard() { # <ticks> <pgrep output> <top rows...>: runs the guard, prints the kill calls
  local ticks=$1 i=0
  GUARD_PG=$2
  shift 2
  GUARD_TOP=$(top_rows "$@")
  pgrep() { printf '%s\n' "$GUARD_PG"; }
  top() { printf '%s\n' "$GUARD_TOP"; }
  kill() { echo "kill $*"; }
  while [ "$i" -lt "$ticks" ]; do
    mem_guard
    i=$((i + 1))
  done
  unset -f pgrep top kill
}
ab_over=0
check "guard: three readings over the limit stop nothing" "" \
  "$(guard 3 $'11\n22' '11  2587M' '22  14G')"
ab_over=0
check "guard: the fourth stops only the process over the limit" "kill -TERM 22" \
  "$(guard 4 $'11\n22' '11  2587M' '22  14G')"
check "guard: the log names the pids stopped" "yes" \
  "$(grep -q 'sky-train holds 14 GB (limit 13): training stopped (TERM to pid 22)$' "$log" && echo yes)"
ab_over=0
check "guard: every process over the limit, none under it" "kill -TERM 22 33" \
  "$(guard 4 $'11\n22\n33' '11  950K' '22  16G+' '33  13500M')"
ab_over=0
check "guard: a reading under the limit starts the count again" "" \
  "$(guard 3 22 '22  14G'; guard 1 22 '22  12G'; guard 3 22 '22  14G')"
ab_over=0
check "guard: the count restarts after a stop" "kill -TERM 22" \
  "$(guard 7 22 '22  14G')"
settings

# ----------------------------------------------------------------------------- set_brake
mkdir -p "$tmp/sb"
file=$tmp/sb/brake
log=$file.log
lines() { if [ -f "$1" ]; then wc -l <"$1" | tr -d ' '; else echo 0; fi; }
# shellcheck disable=SC2012 # ls -i is the portable inode
inode() { ls -i "$1" | awk '{ print $1 }'; }
set_brake "0.5 45:15" "first"
check "set_brake writes the line" "0.5 45:15" "$(cat "$file")"
check "set_brake logs the change with its reason" "yes" \
  "$(grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:]{8} 0\.5 45:15 \(first\)$' "$log" && echo yes)"
check "set_brake leaves no temporary file" "no" "$([ -e "$file.tmp" ] && echo yes || echo no)"
i1=$(inode "$file")
set_brake "0.5 45:15" "again"
check "set_brake: the same value is not logged again" "1" "$(lines "$log")"
check "set_brake: the same value is not rewritten" "$i1" "$(inode "$file")"
set_brake "0.85 50:10" "second"
check "set_brake: a new value is logged" "2" "$(lines "$log")"
check "set_brake: a new value replaces the file by a rename" "different" \
  "$([ "$(inode "$file")" != "$i1" ] && echo different || echo same)"
check "set_brake: a folder that is gone does not end the daemon (set -e)" "alive" \
  "$(set -e; file=$tmp/gone/brake; log=$file.log; set_brake "1" "x" 2>/dev/null; echo alive)"

# ----------------------------------------------------------------------------- settings check
ok_lines=("0.5 45:15" "1" "0.85 50:10" " 1 45:15 " "0.95 55:5" ".5 1:1")
bad_lines=("" "0" "1.5" "fast" "0.5 45" "0.5 45:0" "0.5 45:15 more" "0.5 a:b" "0.5 0.001:5")
out=""
for l in "${ok_lines[@]}"; do valid_brake_line "$l" && out="${out}t" || out="${out}f"; done
check "valid brake lines are accepted" "tttttt" "$out"
out=""
for l in "${bad_lines[@]}"; do valid_brake_line "$l" && out="${out}t" || out="${out}f"; done
check "invalid brake lines are refused" "fffffffff" "$out"
check "a fractional IDLE_MIN is refused (exit 2)" "2" \
  "$(IDLE_MIN=1.5 AUTO_BRAKE_OS=Darwin bash "$here/auto-brake.sh" --once "$tmp/x/brake" >/dev/null 2>&1; echo $?)"
check "an invalid IN_USE is refused (exit 2)" "2" \
  "$(IN_USE="0.5 45" AUTO_BRAKE_OS=Darwin bash "$here/auto-brake.sh" --once "$tmp/x/brake" >/dev/null 2>&1; echo $?)"
check "an unknown option is refused (exit 2)" "2" \
  "$(bash "$here/auto-brake.sh" --bogus >/dev/null 2>&1; echo $?)"

# ----------------------------------------------------------------------------- pid file
mkdir -p "$tmp/pid" "$tmp/fake"
pf=$tmp/pid/brake.pid
printf 'while :; do sleep 0.2; done\n' >"$tmp/fake/auto-brake.sh"
bash "$tmp/fake/auto-brake.sh" >/dev/null 2>&1 &
fake=$!
bg+=("$fake")
sleep 60 >/dev/null 2>&1 &
other=$!
bg+=("$other")
sleep 0 &
dead=$!
wait "$dead"
wait_for "is_auto_brake $fake" >/dev/null
check "is_auto_brake: a process named auto-brake.sh" "yes" \
  "$(is_auto_brake "$fake" && echo yes || echo no)"
check "is_auto_brake: another process" "no" "$(is_auto_brake "$other" && echo yes || echo no)"
check "is_auto_brake: a pid that is gone" "no" "$(is_auto_brake "$dead" && echo yes || echo no)"
claim_pidfile "$pf" 4242 >/dev/null
check "claim: a free pid file is written" "4242" "$(cat "$pf")"
echo "$fake" >"$pf"
holder=$(claim_pidfile "$pf" 4242)
status=$?
check "claim: a second instance is refused (status 3)" "3 $fake" "$status $holder"
check "claim: the refused instance leaves the pid file alone" "$fake" "$(cat "$pf")"
check "daemon_pid: the live daemon of a brake file" "$fake" "$(daemon_pid "$tmp/pid/brake")"
echo "$dead" >"$pf"
claim_pidfile "$pf" 4242 >/dev/null
check "claim: a pid file whose process is gone is taken over" "0 4242" "$? $(cat "$pf")"
echo "$other" >"$pf"
claim_pidfile "$pf" 4242 >/dev/null
check "claim: a pid file of another program is taken over" "0 4242" "$? $(cat "$pf")"
check "claim: no temporary file is left" "" "$(find "$tmp/pid" -type f ! -name brake.pid)"
release_pidfile "$pf" 999
check "release: a pid file of another instance stays" "4242" "$(cat "$pf")"
release_pidfile "$pf" 4242
check "release: our pid file is removed" "no" "$([ -e "$pf" ] && echo yes || echo no)"
kill "$fake" "$other" 2>/dev/null
wait "$fake" "$other" 2>/dev/null

# ----------------------------------------------------------------------------- real daemons
helper() { # <folder>: a thermal helper that prints 0, so that no swiftc build is attempted
  mkdir -p "$1"
  printf '#!/bin/sh\necho 0\n' >"$1/brake.thermal"
  chmod +x "$1/brake.thermal"
}
helper "$tmp/d"
env "${daemon_env[@]}" bash "$here/auto-brake.sh" "$tmp/d/brake" 2>"$tmp/d/err1" &
d1=$!
bg+=("$d1")
wait_for "[ \"\$(cat '$tmp/d/brake.pid' 2>/dev/null)\" = $d1 ] && [ -s '$tmp/d/brake' ]"
check "daemon: holds its pid file and writes the brake file" "$d1 yes" \
  "$(cat "$tmp/d/brake.pid" 2>/dev/null) $(valid_brake_line "$(cat "$tmp/d/brake" 2>/dev/null)" && echo yes)"
env "${daemon_env[@]}" bash "$here/auto-brake.sh" "$tmp/d/brake" >/dev/null 2>"$tmp/d/err2"
check "daemon: a second one on the same file exits with status 3" "3" "$?"
check "daemon: and says why" "yes" "$(grep -q 'already running' "$tmp/d/err2" && echo yes)"
env "${daemon_env[@]}" bash "$here/train-when-idle.sh" --no-caffeinate -f "$tmp/d/brake" -- \
  sh -c 'exit 5' 2>/dev/null
check "train-when-idle: reuses a running daemon, returns the command's status" "5" "$?"
check "train-when-idle: leaves the daemon it did not start running" "yes" \
  "$(kill -0 "$d1" 2>/dev/null && echo yes)"
once=$(env "${daemon_env[@]}" bash "$here/auto-brake.sh" --once "$tmp/d/brake" 2>/dev/null)
check "--once names the live daemon's pid, once" "pid $d1" \
  "$(printf '%s\n' "$once" | sed -n 's/^daemon  *//p')"
kill -TERM "$d1"
wait "$d1"
check "daemon: TERM stops it with status 0" "0" "$?"
check "daemon: removes its pid file and its temporary file on exit" "no no" \
  "$([ -e "$tmp/d/brake.pid" ] && echo yes || echo no) $([ -e "$tmp/d/brake.tmp" ] && echo yes || echo no)"

helper "$tmp/e"
env "${daemon_env[@]}" bash "$here/train-when-idle.sh" --no-caffeinate -f "$tmp/e/brake" -- \
  sh -c 'cat "$SKY_BRAKE_FILE.pid" >"$SKY_BRAKE_FILE.seen"; cat "$SKY_BRAKE_FILE" >"$SKY_BRAKE_FILE.line"; exit 6' \
  2>/dev/null
check "train-when-idle: starts a daemon, returns the command's status" "6" "$?"
check "train-when-idle: the brake file is written before the command starts" "yes" \
  "$(valid_brake_line "$(cat "$tmp/e/brake.line" 2>/dev/null)" && echo yes)"
seen=$(cat "$tmp/e/brake.seen" 2>/dev/null)
wait_for "! is_auto_brake '$seen'" >/dev/null
check "train-when-idle: stops the daemon it started" "stopped" \
  "$([ -n "$seen" ] && ! is_auto_brake "$seen" && echo stopped)"
check "train-when-idle: the pid file is gone with it" "no" \
  "$([ -e "$tmp/e/brake.pid" ] && echo yes || echo no)"

once=$(env "${daemon_env[@]}" bash "$here/auto-brake.sh" --once "$tmp/o/brake" 2>/dev/null)
check "--once prints the decision" "yes" "$(printf '%s\n' "$once" | grep -q '^decision ' && echo yes)"
check "--once says when no daemon serves the file" "none for this file" \
  "$(printf '%s\n' "$once" | sed -n 's/^daemon  *//p')"
check "--once writes nothing" "no" "$([ -e "$tmp/o" ] && echo yes || echo no)"

# ----------------------------------------------------------------------------- train-when-idle
mkdir -p "$tmp/nd"
check "--no-daemon: exports an absolute SKY_BRAKE_FILE" "$tmp/nd/rel/brake" \
  "$(cd "$tmp/nd" && env -u SKY_BRAKE_FILE bash "$here/train-when-idle.sh" --no-daemon -f rel/brake -- \
    sh -c 'printf %s "$SKY_BRAKE_FILE"' 2>/dev/null)"
check "--no-daemon: the default is \$SKY_BRAKE_FILE, made absolute" "$tmp/nd/x/b" \
  "$(cd "$tmp/nd" && SKY_BRAKE_FILE=x/b bash "$here/train-when-idle.sh" --no-daemon -- \
    sh -c 'printf %s "$SKY_BRAKE_FILE"' 2>/dev/null)"
check "--no-daemon: else runs/brake" "$tmp/nd/runs/brake" \
  "$(cd "$tmp/nd" && env -u SKY_BRAKE_FILE bash "$here/train-when-idle.sh" --no-daemon -- \
    sh -c 'printf %s "$SKY_BRAKE_FILE"' 2>/dev/null)"
bash "$here/train-when-idle.sh" --no-daemon -f "$tmp/nd/b" -- sh -c 'exit 9' 2>/dev/null
check "--no-daemon: returns the command's status" "9" "$?"
check "--no-daemon: creates nothing" "no" "$([ -e "$tmp/nd/b" ] && echo yes || echo no)"
check "train-when-idle: the daemon is refused off macOS (exit 4)" "4" \
  "$(AUTO_BRAKE_OS=Linux bash "$here/train-when-idle.sh" -f "$tmp/nd/b" -- true >/dev/null 2>&1; echo $?)"
check "auto-brake: refused off macOS (exit 4)" "4" \
  "$(AUTO_BRAKE_OS=Linux bash "$here/auto-brake.sh" "$tmp/nd/b" >/dev/null 2>&1; echo $?)"
check "train-when-idle: no command is a usage error (exit 2)" "2" \
  "$(bash "$here/train-when-idle.sh" --no-daemon >/dev/null 2>&1; echo $?)"
check "train-when-idle: an unknown --set name is refused (exit 2)" "2" \
  "$(bash "$here/train-when-idle.sh" --no-daemon --set NOPE=1 -- true >/dev/null 2>&1; echo $?)"
check "--help works anywhere" "0 0" \
  "$(AUTO_BRAKE_OS=Linux bash "$here/auto-brake.sh" --help >/dev/null; echo -n "$? ")$(AUTO_BRAKE_OS=Linux bash "$here/train-when-idle.sh" --help >/dev/null; echo $?)"

echo "$((n - fails)) of $n cases passed"
[ "$fails" = 0 ]
