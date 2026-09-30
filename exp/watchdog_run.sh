#!/usr/bin/env bash
# Run a command with a hang watchdog that watches OUR process, not the card.
#
# Failure mode observed repeatedly on this shared box: the vLLM engine core
# hangs during model load (main thread in D, wchan=0, no further reads) while
# the process never exits.  A retry loop that only checks the exit status waits
# forever.
#
# The first version of this watchdog sampled *card* utilisation.  That is wrong
# on a shared GPU: when another tenant is busy the card reads ~90% while our
# engine is stalled, so the watchdog never fires.  This version samples our own
# process tree instead -- if neither I/O (read+write bytes) nor CPU time of any
# process in the tree has advanced for STUCK_MIN minutes, the run is hung.
#
# Usage: watchdog_run.sh <stuck_minutes> <logfile> <cmd...>
set -u
STUCK_MIN="$1"; LOG="$2"; shift 2
: > "$LOG"
setsid "$@" >> "$LOG" 2>&1 &
PID=$!

tree_metric() {
  # sum read+write bytes and cpu ticks over the whole process tree
  local total_io=0 total_cpu=0 p
  for p in $(pgrep -P "$PID" 2>/dev/null; echo "$PID"); do
    [ -r "/proc/$p/io" ] && total_io=$((total_io + $(awk '/^(read|write)_bytes/{s+=$2} END{print s+0}' /proc/$p/io 2>/dev/null)))
    [ -r "/proc/$p/stat" ] && total_cpu=$((total_cpu + $(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null)))
  done
  echo "$total_io $total_cpu"
}

prev=""
stuck=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
  cur=$(tree_metric)
  if [ "$cur" = "$prev" ]; then
    stuck=$((stuck+1))
  else
    stuck=0
  fi
  prev="$cur"
  if [ "$stuck" -ge "$STUCK_MIN" ]; then
    echo "!!! watchdog: no I/O or CPU progress for ${stuck}m (pid $PID) -> killing tree" >> "$LOG"
    kill -9 -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
    pkill -9 -P "$PID" 2>/dev/null
    sleep 3
    exit 124
  fi
done
wait "$PID"
exit $?
