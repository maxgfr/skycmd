#!/usr/bin/env bash
# Sets the thermal brake of a running training from what the Mac is doing, with no restart:
# sky-train reads SKY_BRAKE_FILE again every few seconds (crates/sky-train/src/session.rs).
#
#   scripts/auto-brake.sh [runs/brake]        # runs until killed; start it next to the training
#
#   in use        IN_USE="0.5 45:15"   a key or the pointer moved in the last IDLE_MIN minutes, a video
#                                      or a call keeps the display awake, or the Mac is on battery
#   left alone    AWAY="0.85 50:10"    none of the above (the pre-registrations' "two modes")
#   too hot       back to IN_USE while macOS reports thermal pressure
#   out of memory the training is stopped when it holds more than MEM_STOP_GB of memory twice in a
#                 row: a run that swaps does almost no work and fills the disk. It resumes from its
#                 last checkpoint at the next launch.
#
# Every change is a line in <file>.log, with the reason. The pre-registered budgets count working
# time only, so the brake changes how long a run takes, not what it computes.
set -euo pipefail
file=${1:-runs/brake}
log=$file.log
in_use=${IN_USE:-0.5 45:15}
away=${AWAY:-0.85 50:10}
idle_min=${IDLE_MIN:-10}
mem_stop_gb=${MEM_STOP_GB:-13}
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
  else
    set_brake "$away" "no input for $idle_min min"
  fi
  gb=$(train_gb)
  if [ "$gb" -ge "$mem_stop_gb" ]; then
    over=$((over + 1))
    if [ "$over" -ge 2 ]; then
      echo "$(date '+%F %T') sky-train holds $gb GB (limit $mem_stop_gb): training stopped" >>"$log"
      pkill -TERM -x sky-train || true
      over=0
    fi
  else
    over=0
  fi
  sleep "$period"
done
