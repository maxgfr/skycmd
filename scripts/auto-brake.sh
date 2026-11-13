#!/usr/bin/env bash
# Sets the thermal brake of a running training from what the Mac is doing, with no restart:
# sky-train reads SKY_BRAKE_FILE again every few seconds (crates/sky-train/src/session.rs).
#
#   scripts/auto-brake.sh [runs/brake]        # runs until killed; start it next to the training
#
#   in use        IN_USE="0.5 45:15"   a key or the pointer moved in the last IDLE_MIN minutes, a video
#                                      or a call keeps the display awake, or the Mac is on battery
#   left alone    AWAY="0.85 50:10"    none of the above (the pre-registrations' "two modes")
#   night         FULL="0.95 55:5"     left alone for FULL_IDLE_MIN minutes (30), between the hours
#                                      FULL_HOURS (23-8, from 23:00 to 08:00), with the other
#                                      programs using less than OTHER_CPU percent of one core
#                                      (60): the owner sleeps and nothing else works. FULL="" turns
#                                      this mode off
#   too hot       back to IN_USE while macOS reports thermal pressure
#   out of memory the training is stopped when it holds more than MEM_STOP_GB of memory MEM_STOP_N
#                 times in a row (4, two minutes): a run that swaps does almost no work and fills
#                 the disk. It resumes from its last checkpoint at the next launch. The loading of
#                 the decision sets peaks for about a minute and does not count.
#
# Every change is a line in <file>.log, with the reason. The pre-registered budgets count working
# time only, so the brake changes how long a run takes, not what it computes.
set -euo pipefail
file=${1:-runs/brake}
log=$file.log
in_use=${IN_USE:-0.5 45:15}
away=${AWAY:-0.85 50:10}
full=${FULL-0.95 55:5}
full_idle_min=${FULL_IDLE_MIN:-30}
full_hours=${FULL_HOURS:-23-8}
other_cpu=${OTHER_CPU:-60}
idle_min=${IDLE_MIN:-10}
mem_stop_gb=${MEM_STOP_GB:-13}
mem_stop_n=${MEM_STOP_N:-4}
period=${PERIOD:-30}

# Seconds since the last key press or pointer move.
idle_secs() { ioreg -c IOHIDSystem | awk '/HIDIdleTime/ { print int($NF / 1000000000); exit }'; }
# True when an application keeps the display awake (a video, a call, a presentation). `caffeinate`
# does not count: command-line tools hold it for hours with nobody at the screen.
display_held() {
  pmset -g assertions | awk '
    /pid [0-9]+\(.*PreventUserIdleDisplaySleep named/ && !/\(caffeinate\)/ { held = 1 }
    END { exit !held }'
}
on_battery() { pmset -g batt | head -n 1 | grep -q 'Battery Power'; }
hot() {
  pmset -g therm 2>/dev/null | awk -F'= *' '
    /CPU_Speed_Limit/ && $2 + 0 < 100 { hot = 1 }
    /Thermal Warning Level/ && $2 + 0 > 0 { hot = 1 }
    END { exit !hot }'
}
# True between the hours of FULL_HOURS (`23-8`: from 23:00 to 08:00, `0-24`: always).
night() {
  local h from=${full_hours%-*} to=${full_hours#*-}
  h=$((10#$(date +%H)))
  if [ "$from" -le "$to" ]; then
    [ "$h" -ge "$from" ] && [ "$h" -lt "$to" ]
  else
    [ "$h" -ge "$from" ] || [ "$h" -lt "$to" ]
  fi
}
# True when the programs other than the training use less than OTHER_CPU percent of one core.
others_quiet() {
  LC_ALL=C ps -A -o %cpu=,comm= | LC_ALL=C awk -v max="$other_cpu" '
    $2 !~ /(sky-train|sky-arena|sky-eval|python[0-9.]*)$/ { sum += $1 }
    END { exit !(sum < max) }'
}
# Memory of the largest sky-train process, in GB (0 when none runs).
train_gb() {
  local pid
  pid=$(pgrep -x sky-train | head -n 1) || true
  [ -n "$pid" ] || { echo 0; return; }
  top -l 1 -pid "$pid" -stats mem 2>/dev/null | tail -n 1 | awk '
    /G/ { print int($1); next } /M/ { print 0; next } { print 0 }'
}

set_brake() { # <value> <reason>
  [ "$(cat "$file" 2>/dev/null)" = "$1" ] && return 0
  echo "$1" >"$file.tmp"
  mv "$file.tmp" "$file"
  echo "$(date '+%F %T') $1 ($2)" >>"$log"
}

mkdir -p "$(dirname "$file")"
over=0
while :; do
  if hot; then
    set_brake "$in_use" "thermal pressure"
  elif on_battery; then
    set_brake "$in_use" "on battery"
  elif display_held; then
    set_brake "$in_use" "a video or a call keeps the display awake"
  elif [ "$(idle_secs)" -lt $((idle_min * 60)) ]; then
    set_brake "$in_use" "the Mac is in use"
  elif [ -n "$full" ] && [ "$(idle_secs)" -ge $((full_idle_min * 60)) ] && night && others_quiet; then
    set_brake "$full" "night: no input for $full_idle_min min, nothing else works"
  else
    set_brake "$away" "no input for $idle_min min"
  fi
  gb=$(train_gb)
  if [ "$gb" -ge "$mem_stop_gb" ]; then
    over=$((over + 1))
    if [ "$over" -ge "$mem_stop_n" ]; then
      echo "$(date '+%F %T') sky-train holds $gb GB (limit $mem_stop_gb): training stopped" >>"$log"
      pkill -TERM -x sky-train || true
      over=0
    fi
  else
    over=0
  fi
  sleep "$period"
done
