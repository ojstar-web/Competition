# 当前状态（2026-11-23 21:40）

- 本站已验收最高公榜仍 0.651（58023439）。旧路线“复现公开 0.680”已作废：2026-11-23 实测公开 notebook 已达 0.716–0.719，来源是 GLM-5.3 likelihood + user-turn prefix 阶段（见 HANDOFF 顶部“战略纠偏”）。
- 已原样复制并私有化 alexchilton 的 0.717 管线（`okkojbjj/ntl-public-0717-glm-prefix` v1），45 源 mount 检查通过、commit COMPLETE；已提交 scoring rerun，ref **58780390**，2026-11-23 13:34 UTC，PENDING，今日余 4 次。等待 ≤7 h，不重复提交。
- 0.680 线的工程结果保留：v5 256 行私有 COMPLETE 并产出合法 `submission.parquet`（18 类、256 唯一 ID、5199 s）；v6 512 行测时版仍 QUEUED，跑完只记测时，不提交。
- 权威证据：`HANDOFF.md` 顶部两节、`outputs/native_0680_candidate_20261109_127/{run.json,native_probe_v5/}`、`public_top_20261123/{alex716,areeb719,ours717}/`。
- 不动 EnvedaCASMI2026 目录、不抢其 GPU；本站 9 h/5 次/禁网/kernel-only 约束继续有效。
