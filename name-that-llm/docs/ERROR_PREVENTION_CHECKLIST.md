# NameThatLLM project error prevention

## 2026-11-09: public notebook port failed before inference

- Observed: `okkojbjj/ntl-native-0-680-private-probe-256/1` ended ERROR in source cell 6 at about 11 seconds. Eleven expected dataset roots, including the public artifact hub, were absent from `/kaggle/input`; no model was loaded and no prediction was produced. The pulled cloud metadata still lists the 21 dataset sources. Representative missing public datasets remain readable through the Kaggle files API. The mount cause remains unresolved; the private CPU mount audit `okkojbjj/ntl-native-0680-mount-audit/1` was uploaded once to record actual roots.
- Separately observed: the probe builder added `execution_count` and `outputs` to markdown cells, violating the notebook JSON schema. This was not the cell 6 runtime failure, but could break conversion and must be fixed before any revised probe. Validate full notebook schema, not just Python AST and metadata.
- Impact: no scientific inference, no BA evidence, and no public submission. Keep exact v1 log, cloud source and metadata under `outputs/native_0680_candidate_20261109_127/`.
- Next action: use the mount audit to distinguish missing attachment from path changes; change only the demonstrated cause. Do not silently remove required members or relabel a reduced ensemble as the public 0.680 recipe.

## 2026-11-10: Kaggle source-combination queue stall

- Observed: two NameThatLLM jobs with 21 datasets, 8 notebook outputs and 6 models (35 total sources) stayed QUEUED for over 16 hours without failure text or output; the same account completed unrelated jobs. Read-only CPU comparisons with 32, 33 and two different 34-source combinations reached RUNNING/COMPLETE. A 34-source v3 private T4 probe also reached RUNNING immediately.
- Inference, not a published platform rule: the 35-source combination appears to cross an admission boundary. Do not claim a universal numerical Kaggle limit without platform evidence.
- Fix: inspect actual source references; `nuridudak/ntl-q14-lora-v4` was used only in the preflight existence list, so v3 removed that attachment and its check while retaining computation. Source hashes and job identities are in `outputs/native_0680_candidate_20261109_127/`. Do not delete a source used by runtime computation merely to clear a queue.

## 2026-11-10: fake input writer ran before temporary directory creation

- Observed: v3 passed all input-path preflight checks, then cell 7 failed while writing its fake CSV because `/kaggle/temp` did not exist yet. No model inference occurred. This is distinct from v1's mount path failure and the v2 queue issue.
- Fix: v4 writes the private fake CSV under Kaggle's existing `/kaggle/working`; no model, input text selection or fusion setting changed. Before another notebook port, check write-path creation order as well as read-path existence. Evidence: `outputs/native_0680_candidate_20261109_127/native_probe_v3/` and v4 source hash in `run.json`.
