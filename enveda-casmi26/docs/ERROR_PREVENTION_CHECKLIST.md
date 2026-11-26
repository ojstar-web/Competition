# Project Error Prevention

## 2026-11-26: Verify comparable submission rows before Top-25 statistics

- **Observed:** the first v7 difference audit stopped at its 25-candidate assertion for the public477 reproduction CSV; no audit report was written by that attempt.
- **Cause and impact:** the file has 12 genuine 25-candidate rows and 388 single-`CCO` placeholders. Treating every row as a complete ranking would make 400-row Top-1/Top-25 agreement misleading and could falsely explain the public-score gap.
- **Repair and evidence:** `outputs/v7_plan_20261126_01/audit_diff.py` classifies placeholder rows first and compares the 12 valid rows separately; the corrected run exited 0 and wrote `diff_audit.json` with per-row results and input SHA. Check per-row list cardinality and placeholder semantics before comparing submissions.

## 2026-11-19: Read candidate tuple shape before a new probe

- **Observed:** the first GLACIER margin probe exited while unpacking `a_top25` entries as two values; no scoring or report occurred.
- **Cause and impact:** frozen candidate entries have more than two fields, while the new probe used two-value tuple unpacking. The fault was confined to input parsing.
- **Repair and evidence:** the probe now indexes the documented key and score positions (`item[0]`, `item[1]`); `.venv/bin/python src/glacier_margin_probe.py confirm` exited 0 and wrote `outputs/glacier_margin_20261119_01/confirm.json`. Inspect one actual producer row before coding a new consumer of frozen candidate records.

## 2026-11-19: Use the repository Python for S1 scoring probes

- **Observed:** a new precursor-window probe stopped before reading or scoring data when launched with the other competition's Python: first `pyarrow` was missing, then a dependency path supplied PyArrow but its RDKit version failed the official `2026.03.3` check. Neither attempt produced a scientific result.
- **Cause and impact:** the launcher used `/home/yuanyn/jyx/Competition/AI/.venv/bin/python` instead of this repository's registered `.venv/bin/python`.
- **Repair and evidence:** rerunning with `.venv/bin/python src/precursor_window_probe.py --output outputs/precursor_window_20261119_dev_01` exited 0, replayed all 64 baseline Top25 lists, and wrote the paired report. Use this repository's `.venv` for S1 scoring; the separate candidate-generator and ICEBERG environments retain their own pinned entrypoints.

## 2026-11-19: Kaggle dataset download omits publisher metadata

- **Observed:** the first full-file ICEBERG package check failed on `dataset-metadata.json`; all 39 downloaded model, code and wheel files matched `MANIFEST.json` byte counts and SHA256. The Kaggle CLI `datasets download --unzip` had exited 0; no model inference had started.
- **Cause and impact:** Kaggle's downloadable dataset archive did not include the publisher's metadata file listed in the publisher manifest. `ice_runner.py` and `fuse.py` do not reference that file. This is a packaging difference, not a model-file mismatch.
- **Action and evidence:** retain the failed preflight result, verify every actual inference file against the manifest, and document this one explicit omission; do not weaken checks for weights or code. The 39-file check found no other missing or mismatched files.

## 2026-11-19: Respect simulator adduct coverage during input preflight

- **Observed:** the first ICEBERG input preparation stopped before writing an input because 12 of 128 frozen control queries had no spectrum in the runner's positive `[M+H]+`/`[M+Na]+` coverage. No model inference ran.
- **Cause and impact:** the preflight incorrectly required every evaluation molecule to have a covered spectrum, although the runner explicitly leaves unsupported molecules unscored. This was an input-contract error, not missing source data.
- **Repair and evidence:** the corrected preparation keeps all 128 controls, passes an empty spectrum list for unsupported cases, and reports 116 covered queries, 649 covered spectra and 2,760 same-formula candidate SMILES in `outputs/orthogonal_simulator_20261119_01/confirm_probe/prepare.json`. Keep unsupported molecules at their frozen baseline rank in the overall metric; report simulator coverage separately.

## 2026-11-19: Join evaluation spectra from the actual candidate IDs

- **Observed:** the first bond-fragment probe reported exactly zero rank changes on 128 confirmation controls; 125 truth fragment scores were zero. The report is marked invalid in `outputs/bond_fragment_20261119_01/INVALID.md`.
- **Cause and impact:** the probe selected spectra with the old fold confirmation ID file while evaluating a different campaign's confirmation candidates. The join silently yielded no spectra for those candidates; no model or submission used the invalid result.
- **Repair and evidence:** `src/bond_fragment_probe.py` now derives needed IDs from the candidate records and requires a usable spectrum for every evaluation query. The corrected `outputs/bond_fragment_20261119_02/report.json` has nonzero confirmation rank changes and its source/input SHA values. For cross-campaign comparisons, assert complete join cardinality before scoring.

## 2026-11-19: Candidate generator needs the existing Parquet dependency path

- **Observed:** the first local campaign generator stopped while opening `query_features.parquet` with `ImportError: Unable to find a usable engine`; it did not create a candidate row or reach model inference.
- **Cause and impact:** the selected `mypoltype2` Python has pandas and torch but lacks `pyarrow`; this project's earlier fold0 dependency directory contains it. The campaign's first attempted confirmation run produced no scientific result.
- **Repair and evidence:** launch generator and queue with `PYTHONPATH=outputs/fold0_fp_prior_validation_20261110_01/pydeps`; a direct import/read check succeeded, and the restarted generator is writing resumable candidate rows. Preflight file-format modules using the project's registered dependency path before the next Parquet-backed run. The global checklist's 2026-11-09 Parquet item records the same environment-level cause.

## 2026-11-19: Fixed-candidate replay must preserve empty lists

- **Observed:** the first candidate-fusion replay filled a short pool-only list with PubChem candidates, falsely giving the pool-only OOD baseline nonzero recall. Its report and group rows are retained under `outputs/candidate_fusion_20261119_01/invalid_first_pass/`; they are invalid results.
- **Cause and impact:** the list builder applied a common fill rule to policies with different semantics. No training or Kaggle submission used the invalid output.
- **Repair and evidence:** `evaluate.py` now keeps `pool25` strictly pool-only and asserts that replayed `pool20_pc5` exactly matches all 257 frozen confirmation `b_top25` lists. The corrected `report.json` has OOD `pool25` MRR 0 and the replay exits 0.

## 2026-11-19: PubChem shortlist can be empty

- **Observed:** the first adaptive-gate run exited on `assert pc`; confirmation query IDs 706, 197434 and 206056 have `n_window=0` and an empty PubChem shortlist. No report was written by that attempt.
- **Repair and evidence:** `adaptive.py` assigns an empty shortlist a score gap of negative infinity, so it follows the pool-side policy. The rerun exited 0 and wrote `adaptive_report.json`. Check actual candidate-list cardinality before defining any score-gap or list-fusion rule.

## 2026-11-19: Skip empty Parquet selections before `take`

- **Observed:** the first collision-energy probe failed in PyArrow `RecordBatch.take` with `ArrowNotImplementedError` for input types `(string, null)` when a batch contained no target-source rows. It stopped before scoring and wrote no output report.
- **Cause and impact:** an empty Python index list became a null-typed Arrow index array. No scientific result was produced by that attempt.
- **Repair and evidence:** `src/collision_energy_probe.py` now skips batches with no selected indices. The rerun exited 0, replayed all 64 baseline Top25 lists exactly, and wrote `outputs/collision_energy_20261119_01/report.json`.

## 2026-11-19: Exact baseline replay must reuse the original vector

- **Observed:** the first channel-weight probe stopped at `Baseline vector replay differs for q000` before scoring or writing a report.
- **Cause and impact:** scaling the nominal 70/30 vector and normalizing it again introduced floating-point differences. The frozen baseline had not changed; no scientific result was produced by that attempt.
- **Repair and evidence:** `src/channel_weight_probe.py` returns the original vector for 70/30. The rerun exited 0, replayed all 64 frozen baseline Top25 lists, and wrote `outputs/channel_weight_20261119_dev_01/report.json`. Reuse the exact baseline calculation when an identity control is required.

## 2026-11-26: Fold1 ranker feature worker environment and notebook Numba cache

- **Observed:** the first feature worker failed during FPNet model loading with `AssertionError: Torch not compiled with CUDA enabled`; no query feature was written. The first confirm worker then failed at its first spectral call while restoring a Numba disk cache for functions extracted from an `.ipynb` (`AttributeError` in `importlib.import_module(None)`). The immediate cache fix failed before library loading with `TypeError: required field "lineno" missing from expr`.
- **Cause and impact:** this project's `.venv` has CPU-only PyTorch; an extracted notebook function lacks a stable importable module identity for Numba's disk cache. Replacing an AST decorator value also needs location metadata. The failed attempts created no scientific feature artifact; logs remain in `outputs/v7_plan_20261126_01/features_train.log` and `features_confirm.log`.
- **Repair and evidence:** use `device="cpu"`; set `cache=False` on extracted Numba decorators and call `ast.fix_missing_locations` before compilation. The resumed train worker wrote query features; the restarted confirm worker loaded the masked reference library and proceeded to scoring. Retain per-query files for bounded resumption.
