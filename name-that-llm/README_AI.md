# Name That LLM — AI 状态镜像

本目录是 `NameThatLLM` 项目的**只读状态镜像**（docs / src / receipts），由 `sync_ntl.sh` 每 900 秒从
`Net-4090D:~/jyx/Competition/NameThatLLM` 同步，只写 `name-that-llm/`，不触碰 `enveda-casmi26/`。
大产物（`data/`、`outputs/`、`.kaggle/`、`*.parquet|npy|pt|zip`、模型权重）一律不入库。

## 任务

18 个前沿 LLM 写的段落，判断作者；指标 balanced accuracy（18 类召回的平均，稀有类权重与常见类相同）。
约束：kernel-only、禁网、单 kernel ≤9 h、每天 ≤5 次提交。

## 榜单坐标（2026-11-24 12:39 UTC 公榜快照，6943 队）

| 项 | 数值 |
|---|---|
| 参赛队 | 6943 |
| **本站最高公榜** | **0.720**（提交 58811392，2026-11-24 评分） |
| 本站名次 | **406 / 6943**（前 5.8%） |
| 铜牌区 | 前 10%，第 694 名，线 0.718 |
| 银牌区 | 前 5%，第 347 名，线 0.720 |
| 金牌区 | 第 23 名，线 0.758 |
| 0.721 及以上 | 147 队（0.721 即可进银牌区） |
| 0.720 及以上 | 409 队 |

旧口径（6829 队时）：本站 0.651（58023439）、≈2951 名（前 43%）；铜 0.716（#682）、银 0.718（341）、
金 0.756（23）、奖金线 0.774、第一 0.809；341 队挤在 0.716–0.718。保留仅为对照。

## 公开配方的真实位置（重要更正）

- 2026-11-08 时公开最强 0.680，只到 ≈ 第 2089 名（前 31%）——**旧状态**，已不能作为目标或天花板。
- 2026-11-24 实测：公开 notebook 到 **0.716–0.721**（`areebbb/0-720`、`zayd88903/0-721`、
  `alexchilton/0-719`）。这些是同一族代码：dennis6886 的 GLM-5.3 user-prefix + guarded Fable-return
  + areebbb 的 sibling odds；0.001–0.002 差异是运行噪声。
- 提升来源是 **GLM-5.3 likelihood + user-turn prefix**，不是“类别先验校正”。

## 当前进行中（自有改动）

- 已评分：`58780390`/`58794719`（0.717 原样线，同代码两次）都是 **0.718**；`58811392`
  （0.720 sibling-odds 原样线）**0.720，rank 406**，仍差银线 2 名区间。
- 已改、待发：`okkojbjj/ntl-public-0720-c20`，唯一改动是 V1 阶段 `--max-cover 0.30 → 0.20`。
  依据是 `v1_stage.py` 自带 train OOF 曲线：15% +0.030、**20% +0.0315**、全量 +0.029；
  默认 0.30 不是最优点。预期冲击 0.721，进银牌区。
- 0.680 线工程结果保留：v5 私有 256 行 COMPLETE（256 唯一假 ID、18 类、合法 parquet、5199 s），不再提交。

## 已止损（不要重试）

- 公开 0.680 配方 v1–v4：挂载缺失 / 写不存在的 `/kaggle/temp` / 队列取消，均无预测。
- 官方数据-only exact-18：公榜 0.484，已回退。
- 文风/排版概率、旧池 DeBERTa：题族泄漏，已止损。
- 硬 Hungarian / exact-18 指派：作者 125 次提交统计显示公榜每次 −0.009～−0.017，隐藏测试每题段落数很少。
- 提高 `RUN_LIMIT_S`：查源码后否掉，`V1_DEADLINE_H=8.8`、4000 行 GLM 成本约 55 分钟，时间不是瓶颈。

## 目录

```
name-that-llm/
  README_AI.md          # 本文件
  docs/                 # 项目 *.md（HANDOFF.md 为决策流水，最新在上）
  src/                  # 项目 *.py
  receipts/native_0680/ # 0.680 探针的 run.json 等小 JSON 凭证
  receipts/leaderboard_20261124/ # 公榜快照汇总
```

## 同步机制

- 脚本：`~/jyx/state_repo_ntl/sync_ntl.sh`；独立 clone（`~/jyx/state_repo_ntl/Competition`），
  不复用其他会话工作树。
- 频率：`while true; do sync_ntl.sh; sleep 900; done`。
- 推送失败自动 `pull --rebase --autostash` 重试 3 次；从不 force push。
- 只 `git add -A name-that-llm`，不 stage 其他目录。
