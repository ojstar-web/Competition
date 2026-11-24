#!/bin/bash
# Shared Kaggle GPU-session gate for this account (limit: 2 concurrent GPU sessions).
# Usage:
#   gpu_slot.sh run <kernel-ref> -- <kaggle push args...>   # push under a slot, hold until terminal
#   gpu_slot.sh status                                      # show who holds the slots
#   gpu_slot.sh free                                        # release stale slots whose kernel is terminal
set -u
DIR="$HOME/jyx/.gpu_slots"
KAGGLE="$HOME/jyx/Competition/EnvedaCASMI2026/.venv/bin/kaggle"
SLOTS=("$DIR/slot1" "$DIR/slot2")
mkdir -p "$DIR"

status() {
  for s in "${SLOTS[@]}"; do
    if [ -f "$s" ]; then
      printf '%s: %s\n' "$(basename "$s")" "$(cat "$s")"
    else
      printf '%s: free\n' "$(basename "$s")"
    fi
  done
}

case "${1:-}" in
  status) status; exit 0;;
  free) rm -f "${SLOTS[@]}"; echo "cleared"; exit 0;;
esac

[ "${1:-}" = "run" ] || { echo "usage: gpu_slot.sh run <kernel-ref> -- <kaggle args>"; exit 2; }
REF="$2"; shift 3   # drop run, ref, --

acquire() {
  for attempt in $(seq 1 240); do
    for s in "${SLOTS[@]}"; do
      exec 9>"$s"
      if flock -n 9; then
        printf 'pid=%s ref=%s since=%s\n' "$$" "$REF" "$(date -u +%FT%TZ)" > "$s"
        SLOT_FILE="$s"; SLOT_FD=9; return 0
      fi
      exec 9>&-
    done
    sleep 15
  done
  return 1
}

acquire || { echo "no slot after wait"; exit 3; }

push_ok=0
for attempt in $(seq 1 60); do
  if "$KAGGLE" "$@" > /tmp/gpu_slot_push.out 2>&1; then push_ok=1; break; fi
  if grep -q "Maximum batch GPU session" /tmp/gpu_slot_push.out; then
    sleep 120; continue
  fi
  echo "push failed: $(tail -1 /tmp/gpu_slot_push.out)"; break
done
if [ "$push_ok" -ne 1 ]; then
  rm -f "$SLOT_FILE"; exec 9>&-; exit 4
fi
echo "pushed $REF under $(basename "$SLOT_FILE")"
for i in $(seq 1 480); do
  st=$("$KAGGLE" kernels status "$REF" 2>/dev/null | tail -1)
  case "$st" in
    *COMPLETE*|*ERROR*|*FAILED*|*CANCEL*) echo "terminal: $st"; break;;
  esac
  sleep 60
done
rm -f "$SLOT_FILE"; exec 9>&-; exit 0
