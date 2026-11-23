# Name That LLM — AI 状态镜像

本目录是 `NameThatLLM` 项目的**只读状态镜像**（docs / src / receipts），由 `sync_ntl.sh` 每 900 秒从
`Net-4090D:~/jyx/Competition/NameThatLLM` 同步，只写 `name-that-llm/`，不触碰 `enveda-casmi26/`。
大产物（`data/`、`outputs/`、`.kaggle/`、`*.parquet|npy|pt|zip`、模型权重）一律不入库。

## 任务

18 个前沿 LLM 写的段落，判断作者；指标 balanced accuracy（18 类召回的平均，稀有类权重与常见类相同）。
约束：kernel-only、禁网、单 kernel ≤9 h、每天 ≤5 次提交。

## 榜单坐标（2026-11-23 实测）

| 项 | 数值 |
|---|---|
| 参赛队 | 6829 |
| 本站最高公榜 | **0.651**（提交 58023439） |
| 本站名次 | ≈ 2951 / 6829（前 43%） |
| 铜牌线 | 0.716（第 682 名） |
| 银牌线 | 0.718（341 队） |
| 金牌线 | 0.756（23 队） |
| 奖金线 | 0.774（前 10） |
| 第一名 | 0.809 |

关键结构：**341 队挤在 0.716–0.718**，说明这一档是“公开配方 + 少量后处理/更强成员”的平台层，
不是靠单点调参能穿过的；再往上是 0.756 金线，需要新信号。

## 公开配方的真实位置（重要更正）

- 2026-11-08 时，公开最强 0.680，只到 ≈ 第 2089 名（前 31%）——**这是旧状态**。
- 2026-11-23 21:10 实测（`kaggle kernels list --competition name-that-llm --sort-by scoreDescending`）：
  公开 notebook 已到 **0.716–0.719**，例如 `areebbb/0-719-preserve-glm-to-fable-corrections`(0.719)、
  `alexchilton/name-that-llm-0-719-what-moved-the-board`(0.719)、
  `lightningv08/0-717-user-prefix-glm-centering-qwen3-8-27b`(0.717)。
- 提升来源是 **GLM-5.3 likelihood + user-turn prefix** 阶段（见 `docs/HANDOFF.md` 顶部“战略纠偏”），
  不是“类别先验校正”。公开 0.680 已不能作为目标或天花板。

## 当前进行中

- 提交 **58780390**：逐字复制 alexchilton 的 0.717 GLM-user-prefix 管线（仅改私有化/id），
  2026-11-23 13:34 UTC 提交，状态 **PENDING**，今日余 4 次。
- 0.680 线工程结果保留：v5 私有 256 行 COMPLETE，256 唯一假 ID、18 类、合法 `submission.parquet`，
  墙钟 5199 s；但该线已不提交。

## 已止损（不要重试）

- 公开 0.680 配方 v1–v4：挂载缺失 / 写不存在的 `/kaggle/temp` / 队列取消，均无预测。
- 官方数据-only exact-18：公榜 0.484，已回退。
- 文风/排版概率、旧池 DeBERTa：题族泄漏，已止损。
- 硬 Hungarian / exact-18 指派：作者 125 次提交统计显示公榜每次 −0.009～−0.017，隐藏测试每题段落数很少。

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
