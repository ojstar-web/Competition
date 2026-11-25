# Handoff: ruler-v2 measurements for the candidate-recall line (2026-11-25)

Owner of this note: the session that ran the ruler-v2 experiment. The v6/v7 PubChem-depth work is
owned by another session. These numbers are offered as evidence, not as a competing plan.

## What ruler v2 is
- 4,096 untouched fold1 identity groups (2,048 OOD + 2,048 control), frozen by SHA256 salt
  `CASMI26-ruler-v2`; 52,737 eligible groups remained after excluding the earlier 2,048 and the NP panel.
- 4,118 queries / 37,961 spectra; features are label-blind, truth keys in a separate file.
- The OOD arm's structures were physically removed from the candidate pool (710,701 -> 708,653 rows),
  so pool25 scores exactly 0.0000 on OOD by construction.
- Files: `outputs/ruler_v2_20261123_01/{selection_v2.json,query_features_v2.parquet,labels_v2.sealed.parquet,score_v2.json}`.

## Headline numbers (group-level, 2,048 vs 2,047 groups)
- In-pool control MRR (pool25) = 0.8091.
- Our real LB is 0.477, which implies an OOD prevalence of roughly 41% if the hidden set only differed
  by "structure in pool vs not". Treat that as an upper bound: the deployed pipeline also loses to
  instrument/asset mismatch, so the true unknown-structure share may be lower.

## Policy deltas vs pool25 (paired group bootstrap, 20k resamples)
| policy | OOD MRR | control MRR | p=.10 | p=.30 | p=.50 |
|---|---|---|---|---|---|
| pool20_pc5 | 0.0271 | 0.8090 | +0.0026 | +0.0080 | +0.0135 |
| pool15_pc10 | 0.0368 | 0.8088 | +0.0034 | +0.0108 | +0.0182 |
| pool10_pc15 | 0.0500 | 0.8081 | +0.0041 | +0.0143 | +0.0245 |
| alternate4_1 | 0.0699 | 0.8081 | +0.0061 | +0.0202 | +0.0344 |
| alternate2_1 | 0.1151 | 0.8022 | +0.0053 | +0.0297 | +0.0541 |
| alternate1_1 | 0.1728 | 0.7753 | -0.0132 | +0.0282 | +0.0695 |
| raw_score | 0.3276 | 0.4318 | -0.3068 | -0.1659 | -0.0249 |

Reading: pool15_pc10 is a strict improvement over pool20_pc5 at every prevalence, and costs no control
MRR (0.8088 vs 0.8090). alternate2_1/alternate1_1 only pay off above roughly p=0.15-0.20.

## Two practical findings
1. **Lists shorter than 25.** 18.8% of queries under pool20_pc5 emit fewer than 25 candidates because the
   policy is `pool[:20] + pc[:5]` and the pool window is often smaller than 20 (median 77, but the tail is
   small). The PubChem 10 ppm window has median 11,329 entries, so appending PubChem candidates to fill
   the tail is free: it cannot lower an existing rank. See `fill_policies.py`
   (`pool20_pc_fill`, `pool10_pc_fill`, `pool25_fill`).
2. **Cross-source raw scores are unusable.** Merging pool and PubChem by raw f.z drops control MRR from
   0.809 to 0.432. Any merge must be rank-based or gated.

## Caveat on the v7 gate
An OOD-coverage gate (e.g. >= 0.5625) is a proxy. What scores is MRR@25. At p=0.30 the whole coverage
gain from pool20_pc5 to pool15_pc10 is worth about +0.003 MRR, so coverage targets should be tied to an
MRR delta before being treated as a promotion criterion.

## Coordination
The GPU-session gate lives in `shared/gpu_slot.sh` (account limit is 2 concurrent GPU sessions). Push GPU
kernels only through `gpu_slot.sh run <slug> -- kernels push -p <dir>`.
