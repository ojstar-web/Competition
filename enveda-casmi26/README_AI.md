# Enveda CASMI 2026 — shared state for other agents

Machine: Net-4090D (100.64.21.76), user yuanyn, project `/home/yuanyn/jyx/Competition/EnvedaCASMI2026`.
This folder is a read-only mirror, refreshed every 15 minutes. Do not expect data or model weights here.

## Layout
- `docs/` — project documents (NOW.md is the entry point, then AGENTS.md, PLAN.md)
- `src/` — pipeline scripts
- `receipts/` — frozen experiment receipts (ruler_v2 = new validation partition, candidate fusion report)
- `notes/` — per-agent notes

## Verified leaderboard state (2026-11-23)
5845 teams. Ours 0.477 → rank ~2113 (top 36%). Bronze 0.500 (rank 584), silver 0.502 (292), gold 0.513 (21), best 0.554.
Best public notebooks claim 0.484-0.486 (≈top 29%), so reproducing public work does not reach a medal.

## Current work (this session)
1. Ruler v2: 4096 untouched fold1 identity groups frozen, label-blind features extracted (4118 queries, 37961 spectra).
2. Prevalence sweep on the old confirm set showed pool20_pc5's +0.0121 was an artifact of the artificial 50/50 OOD mix: at p=0.10 the delta is +0.0025 with CI [-0.066, +0.072].
3. Candidate generation on ruler v2 is running (6 shards, CPU FPNet); scoring script `score_v2.py` will report p=5..50% deltas.
4. Found: 18.8% of queries under pool20_pc5 emit fewer than 25 candidates, while the PubChem 10ppm window has median 11329 entries. Fill-to-25 policies are added in `fill_policies.py`.

## Rules for other agents
- Do not write to this repo; open a PR or leave it to the owning session.
- Do not submit to Kaggle from here; submission budget is 5/day and belongs to the owning session.
