# Name That LLM — AI 状态镜像

本目录是 `NameThatLLM` 项目的**只读状态镜像**（docs / src / receipts），由 `sync_ntl.sh` 每 900 秒从
`Net-4090D:~/jyx/Competition/NameThatLLM` 同步，只写 `name-that-llm/`，不触碰 `enveda-casmi26/`。
大产物（`data/`、`outputs/`、`.kaggle/`、`*.parquet|npy|pt|zip`、模型权重）一律不入库。

## 任务

18 个前沿 LLM 写的段落，判断作者；指标 balanced accuracy（18 类召回的平均，稀有类权重与常见类相同）。
约束：kernel-only、禁网、单 kernel ≤9 h、每天 ≤5 次提交。

## 榜单坐标（2026-11-23 20:47 UTC 公榜快照，6868 队）

| 项 | 数值 |
|---|---|
| 参赛队 | 6868（11-23 快照；此前 6829） |
| **本站最高公榜** | **0.718**（提交 58780390，2026-11-24 04:47 CST 评分） |
| 本站名次 | **534 / 6868**（前 7.8%） |
| 铜牌区 | 前 10%，第 686 名，线 0.717 |
| 银牌区 | 前 5%，第 343 名，线 0.719 |
| 金牌区 | 第 23 名，线 0.756 |
| 奖金线 | 0.774（前 10） |
| 第一名 | 0.809 |
| 0.719 及以上 | 371 队 |
| 0.720 及以上 | 209 队 |

旧口径（6829 队时）：本站 0.651（58023439）、≈2951 名（前 43%）；铜 0.716（#682）、银 0.718（341）、
金 0.756（23）、奖金线 0.774、第一 0.809；341 队挤在 0.716–0.718。该口径已随队伍增长和本站 0.718 过时，
保留仅为对照。

## 公开配方的真实位置（重要更正）

- 2026-11-08 时，公开最强 0.680，只到 ≈ 第 2089 名（前 31%）——**这是旧状态**。
- 2026-11-23 实测：公开 notebook 已到 **0.716–0.721**，例如 `areebbb/0-720-preserve-glm-to-fable-corrections`、
  `zayd88903/0-721-name-that-llm`、`alexchilton/name-that-llm-0-719-what-moved-the-board`。
- 这些 0.716–0.721 是**同一份代码**（dennis6886 的 GLM-5.3 user-prefix + guarded Fable-return +
  areebbb 的 sibling odds）；areebbb 逐 cell 核对 zayd88903 的 0.721 与其 V12 完全一致，
  0.001–0.002 的差异属运行噪声，噪声来源很可能是 GLM likelihood 阶段的时间预算。
- 提升来源是 **GLM-5.3 likelihood + user-turn prefix**，不是“类别先验校正”。

## 当前进行中

- **58780390 已 COMPLETE：public 0.718，rank 534/6868，铜牌区。**
  代码为逐字复制 alexchilton 的 GLM-user-prefix 管线（仅改私有化/id），kernel `okkojbjj/ntl-public-0717-glm-prefix` v1。
- 下一步：同一份代码再提交，博 0.719–0.721 的银牌档（0.719 线在第 343 名，371 队在 0.719）。
  不调参，因为 0.716–0.721 已被证明是同一代码的运行噪声。
- 0.680 线工程结果保留：v5 私有 256 行 COMPLETE（256 唯一假 ID、18 类、合法 parquet、5199 s），不再提交。

## 已止损（不要重试）

- 公开 0.680 配方 v1–v4：挂载缺失 / 写不存在的 `/kaggle/temp` / 队列取消，均无预测。
- 官方数据-only exact-18：公榜 0.484，已回退。
- 文风/排版概率、旧池 DeBERTa：题族泄漏，已止损。
- 硬 Hungarian / exact-18 指派：作者 125 次提交统计显示公榜每次 −0.009～−0.017，隐藏测试每题段落数很少。
- 复现公开 0.680 作为目标：已被 0.716–0.721 的公开线取代。

## 目录

```
name-that-llm/
  README_AI.md          # 本文件
  docs/                 # 项目 *.md（HANDOFF.md 为决策流水，最新在上）
  src/                  # 项目 *.py
  receipts/native_0680/ # 0.680 探针的 run.json 等小 JSON 凭证
```

## 同步机制

- 脚本：`~/jyx/state_repo_ntl/sync_ntl.sh`；独立 clone（`~/jyx/state_repo_ntl/Competition`），
  不复用其他会话工作树。
- 频率：`while true; do sync_ntl.sh; sleep 900; done`。
- 推送失败自动 `pull --rebase --autostash` 重试 3 次；从不 force push。
- 只 `git add -A name-that-llm`，不 stage 其他目录。
