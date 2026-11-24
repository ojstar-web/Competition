# Kaggle GPU session gate (account-wide limit: 2 concurrent GPU sessions)

Two agent sessions share account `okkojbjj`. Kaggle rejects a third concurrent GPU kernel with
`Maximum batch GPU session count of 2 reached`. Use the gate instead of pushing directly.

## Install once (server)
    cp ~/jyx/state_repo/Competition/shared/gpu_slot.sh ~/jyx/.gpu_slots/gpu_slot.sh
    chmod +x ~/jyx/.gpu_slots/gpu_slot.sh

## Use
    ~/jyx/.gpu_slots/gpu_slot.sh run <owner/kernel-slug> -- kernels push -p <kernel-dir>

It takes one of two slots (flock, no polling race), pushes, waits for the kernel to reach a terminal
state, then frees the slot. If Kaggle still refuses because a foreign session (e.g. a notebook started
from the web UI) holds a slot, it backs off 2 minutes and retries.

Inspect / reset:
    ~/jyx/.gpu_slots/gpu_slot.sh status
    ~/jyx/.gpu_slots/gpu_slot.sh free      # only when no kernel of yours is actually running

## What counts against the 2
Every GPU kernel in QUEUED/RUNNING state, from any client, including web-UI "Save & Run All" runs.
CPU kernels and committed-but-finished kernels do not.

## Coordination rule
Never call `kaggle kernels push` for a GPU kernel directly. Always go through `gpu_slot.sh run`.
