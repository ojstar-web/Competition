#!/bin/bash
# Shared Kaggle GPU-session gate (account limit: 2 concurrent GPU sessions).
#   gpu_slot.sh sessions                 # list our kernels that are queued/running
#   gpu_slot.sh status                   # which gate slots are held
#   gpu_slot.sh free                     # clear stale gate slots
#   gpu_slot.sh run <slug> -- <kaggle push args...>
set -u
DIR="$HOME/jyx/.gpu_slots"; KAGGLE="$HOME/jyx/Competition/EnvedaCASMI2026/.venv/bin/kaggle"
SLOTS=("$DIR/slot1" "$DIR/slot2"); mkdir -p "$DIR"

sessions() {
  "$KAGGLE" kernels list -m --page-size 25 --sort-by dateRun 2>/dev/null | tail -n +3 | awk '{print $1}' | while read -r ref; do
    case "$ref" in *-*) : ;; *) continue;; esac
    [ -z "$ref" ] && continue
    st=$("$KAGGLE" kernels status "$ref" 2>/dev/null | tail -1)
    case "$st" in
      *RUNNING*|*QUEUED*) printf '%-55s %s\n' "$ref" "${st##*status }";;
    esac
  done
}

case "${1:-}" in
  sessions) sessions; exit 0;;
  status) for s in "${SLOTS[@]}"; do [ -f "$s" ] && printf '%s: %s\n' "$(basename "$s")" "$(cat "$s")" || printf '%s: free\n' "$(basename "$s")"; done; exit 0;;
  free) rm -f "${SLOTS[@]}"; echo cleared; exit 0;;
esac
[ "${1:-}" = run ] || { echo "usage: gpu_slot.sh sessions|status|free|run <slug> -- <args>"; exit 2; }
REF="$2"; shift 3

acquire() {
  for a in $(seq 1 480); do
    for s in "${SLOTS[@]}"; do
      exec 9>"$s"
      if flock -n 9; then
        printf 'pid=%s ref=%s since=%s\n' "$$" "$REF" "$(date -u +%FT%TZ)" > "$s"
        SLOT_FILE="$s"; return 0
      fi
      exec 9>&-
    done
    sleep 10
  done
  return 1
}
acquire || { echo "gate: no local slot"; exit 3; }
echo "gate: holding $(basename "$SLOT_FILE") for $REF"

ok=0
for a in $(seq 1 120); do
  out=$("$KAGGLE" "$@" 2>&1); rc=$?
  if [ $rc -eq 0 ] && ! grep -q "Kernel push error\|Maximum batch GPU session" <<<"$out"; then ok=1; break; fi
  if grep -q "Maximum batch GPU session" <<<"$out"; then
    echo "gate: Kaggle slots full (foreign session), backing off"; sleep 180; continue
  fi
  echo "gate: push failed rc=$rc: $(tail -1 <<<"$out")"; break
done
[ $ok -eq 1 ] || { rm -f "$SLOT_FILE"; exit 4; }
echo "gate: pushed $REF"
for i in $(seq 1 600); do
  st=$("$KAGGLE" kernels status "$REF" 2>/dev/null | tail -1)
  case "$st" in *COMPLETE*|*ERROR*|*FAILED*|*CANCEL*) echo "gate: terminal $st"; break;; esac
  sleep 60
done
rm -f "$SLOT_FILE"; exit 0
