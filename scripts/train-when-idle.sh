#!/usr/bin/env bash
# Runs a training command politely: it works hard when the Mac is left alone and cool, and gently
# when someone uses it, when it is on battery or when it heats (docs/idle-training.md).
#
#   scripts/train-when-idle.sh [options] -- <command> [args...]
#
# What it does, in order:
#   1. exports SKY_BRAKE_FILE, the absolute path of the brake file (default: $SKY_BRAKE_FILE when it
#      is set, else runs/brake), to the command;
#   2. starts scripts/auto-brake.sh for that file, unless one already serves it (its <file>.pid, or
#      an older daemon whose command line names the file): that one is used and left running;
#   3. waits until the daemon has claimed the file (at most TW_WAIT seconds, 60), then runs the
#      command under `caffeinate -i` when it exists (no idle sleep while it runs);
#   4. when the command ends, stops the daemon if and only if step 2 started it;
#   5. exits with the command's status.
#
# Options:
#   -f, --brake-file FILE   the brake file (default: $SKY_BRAKE_FILE, else runs/brake)
#   --set NAME=VALUE        a setting of the daemon it starts, exported (repeatable): IN_USE, AWAY,
#                           FULL, HOT, IDLE_MIN, FULL_IDLE_MIN, FULL_HOURS, OTHER_CPU, MEM_STOP_GB,
#                           MEM_STOP_N, PERIOD, TRAIN_PROC, TRAIN_PROCS_RE (scripts/auto-brake.sh
#                           --help). Setting them in the environment works the same. A daemon that
#                           already runs keeps its own settings
#   --no-daemon             start no daemon: only export SKY_BRAKE_FILE and run the command (for a
#                           system other than macOS, where your own tool writes the file)
#   --no-caffeinate         do not wrap the command in `caffeinate -i`
#   -h, --help              this text
#
# The command must read the brake file: sky-train does (crates/sky-train/src/brake.rs), and so does
# any loop that calls scripts/brake.py; a command that ignores it is not slowed down. Stop the
# training with Ctrl-C: the daemon started here stops with it. A daemon left behind by a killed
# train-when-idle.sh keeps serving its file, and the next run uses it.
# Exit status: the command's; 2 for a usage error or an invalid setting, 4 when the daemon is
# needed and the system is not macOS.

tw_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The daemon's functions (daemon_pid, is_auto_brake, ab_require_macos); sourcing runs nothing.
# shellcheck source-path=SCRIPTDIR source=auto-brake.sh
. "$tw_here/auto-brake.sh"

tw_settings=" IN_USE AWAY FULL HOT IDLE_MIN FULL_IDLE_MIN FULL_HOURS OTHER_CPU MEM_STOP_GB MEM_STOP_N PERIOD TRAIN_PROC TRAIN_PROCS_RE "

tw_usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$tw_here/train-when-idle.sh"; }
tw_say() { echo "train-when-idle: $*" >&2; }

# The absolute brake file that an auto-brake.sh process serves (from its command line and, for a
# relative path, its working folder), or nothing when it cannot be told.
tw_served_file() { # <pid> <command line>
  local arg cwd
  arg=${2##* }
  case $arg in auto-brake.sh | */auto-brake.sh) arg=runs/brake ;; esac
  case $arg in
    /*) echo "$arg" ;;
    *)
      command -v lsof >/dev/null 2>&1 || return 0
      cwd=$(lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1) || true
      [ -z "$cwd" ] || echo "$cwd/$arg"
      ;;
  esac
}
# The pid of an auto-brake.sh with no pid file (an older version) that serves <file>, or nothing.
tw_legacy_daemon() { # <absolute brake file>
  local line pid cmd re='(^|[ /])auto-brake\.sh( |$)'
  while IFS= read -r line; do
    pid=${line%% *}
    cmd=${line#* }
    [[ $cmd =~ $re ]] || continue
    case " $cmd " in *" --once "* | *" --help "* | *" -h "*) continue ;; esac
    if [ "$(tw_served_file "$pid" "$cmd")" = "$1" ]; then
      echo "$pid"
      return 0
    fi
  done < <(ps -A -o pid=,command= 2>/dev/null | sed 's/^ *//')
  return 0
}

tw_cleanup() {
  if [ -n "${tw_daemon:-}" ] && kill -0 "$tw_daemon" 2>/dev/null; then
    kill -TERM "$tw_daemon" 2>/dev/null || true
    wait "$tw_daemon" 2>/dev/null || true
  fi
}

# Starts auto-brake.sh for $tw_file and waits until it holds the pid file and the brake file exists.
# Leaves tw_daemon empty when another daemon claimed the file first.
tw_start_daemon() {
  local waited=0 limit=$((${TW_WAIT:-60} * 5)) status
  "$tw_here/auto-brake.sh" "$tw_file" </dev/null &
  tw_daemon=$!
  while :; do
    if [ "$(cat "$tw_file.pid" 2>/dev/null || true)" = "$tw_daemon" ] && [ -e "$tw_file" ]; then
      tw_say "auto-brake.sh started for $tw_file (pid $tw_daemon); it stops with the command"
      return 0
    fi
    if ! kill -0 "$tw_daemon" 2>/dev/null; then
      status=0
      wait "$tw_daemon" || status=$?
      tw_daemon=
      if [ "$status" = 3 ]; then
        tw_say "another auto-brake.sh claimed $tw_file first: using it, it keeps running after the command"
        return 0
      fi
      tw_say "auto-brake.sh ended with status $status before the command started (see above)"
      exit "$status"
    fi
    if [ "$waited" -ge "$limit" ]; then
      tw_say "auto-brake.sh (pid $tw_daemon) has not written $tw_file after ${TW_WAIT:-60} s:" \
        "starting the command anyway"
      return 0
    fi
    sleep 0.2
    waited=$((waited + 1))
  done
}

tw_main() {
  local no_daemon=0 caffeinate=1 set=0 pid status runner=""
  tw_file=${SKY_BRAKE_FILE:-runs/brake}
  tw_daemon=
  while [ $# -gt 0 ]; do
    case $1 in
      -f | --brake-file)
        [ $# -ge 2 ] || { tw_say "$1 needs a file"; exit 2; }
        tw_file=$2
        shift 2
        ;;
      --brake-file=*) tw_file=${1#*=}; shift ;;
      --set)
        [ $# -ge 2 ] || { tw_say "--set needs NAME=VALUE"; exit 2; }
        case $2 in *=*) ;; *) tw_say "--set needs NAME=VALUE, not '$2'"; exit 2 ;; esac
        case $tw_settings in
          *" ${2%%=*} "*) ;;
          *) tw_say "unknown daemon setting '${2%%=*}' (one of:$tw_settings)"; exit 2 ;;
        esac
        export "${2%%=*}=${2#*=}"
        set=1
        shift 2
        ;;
      --no-daemon) no_daemon=1; shift ;;
      --no-caffeinate) caffeinate=0; shift ;;
      -h | --help) tw_usage; exit 0 ;;
      --) shift; break ;;
      *) tw_say "unknown option '$1' (the command comes after --; see --help)"; exit 2 ;;
    esac
  done
  [ $# -gt 0 ] || { tw_say "no command: train-when-idle.sh [options] -- <command> [args...]"; exit 2; }
  [ -n "$tw_file" ] || { tw_say "the brake file is empty"; exit 2; }
  case $tw_file in /*) ;; *) tw_file=$PWD/$tw_file ;; esac
  export SKY_BRAKE_FILE=$tw_file
  if [ "$caffeinate" = 1 ] && command -v caffeinate >/dev/null 2>&1; then
    runner=caffeinate
  fi

  if [ "$no_daemon" = 1 ]; then
    tw_say "no daemon; SKY_BRAKE_FILE=$SKY_BRAKE_FILE"
  else
    if [ "${AUTO_BRAKE_OS:-$(uname -s)}" != Darwin ]; then
      (ab_require_macos train-when-idle.sh) || {
        tw_say "or run with --no-daemon: SKY_BRAKE_FILE is exported and nothing else is done"
        exit 4
      }
    fi
    mkdir -p "$(dirname "$tw_file")"
    if pid=$(daemon_pid "$tw_file"); then
      tw_say "auto-brake.sh already serves $tw_file (pid $pid): using it, it keeps running after the command"
    elif pid=$(tw_legacy_daemon "$tw_file") && [ -n "$pid" ]; then
      tw_say "auto-brake.sh already serves $tw_file (pid $pid, an older version with no pid file):" \
        "using it, it keeps running after the command"
    else
      pid=
    fi
    if [ -n "$pid" ] && [ "$set" = 1 ]; then
      tw_say "the --set values do not apply: the running daemon keeps its own settings"
    fi
    if [ -z "$pid" ]; then
      trap tw_cleanup EXIT
      trap 'exit 130' INT
      trap 'exit 143' TERM
      tw_start_daemon
    fi
    tw_say "SKY_BRAKE_FILE=$SKY_BRAKE_FILE${runner:+; under caffeinate -i}"
  fi

  status=0
  if [ -n "$runner" ]; then
    caffeinate -i "$@" || status=$?
  else
    "$@" || status=$?
  fi
  exit "$status"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  tw_main "$@"
fi
