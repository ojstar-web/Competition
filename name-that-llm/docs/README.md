# Name That LLM

整理日期：2026-10-21（北京时间）。本目录与 AI 同级，用于此比赛的资料与后续工作。

## 比赛摘要

识别短文本来自哪个模型：18 类，隐藏测试约 4,000 条，文本约 40–250 词。评分为 Balanced Accuracy（各类召回率平均值），公私榜类别比例不同。

- 启动：2026-10-14。
- 报名/合队截止：2027-01-06 23:59 UTC，即北京时间 01-07 07:59。
- 最终提交：2027-01-13 23:59 UTC，即北京时间 01-14 07:59。
- 总奖金 50,000 美元，前十名各 5,000 美元。
- Notebook 提交；CPU/GPU 运行上限均为 9 小时，禁用网络，允许公开免费外部数据及预训练模型。
- 输入：`/kaggle/input/competitions/name-that-llm/test.csv`。
- 输出：`submission.parquet`，列为 `id, model`；ID 唯一、完整，无空值，标签须合法。

以上为[官方概览](https://www.kaggle.com/competitions/name-that-llm/overview/citation)的摘要，来源检索缓存约 5 天前；比赛页面可能更新。

## 标签

```text
gpt-5.5
gpt-5.6-luna
gpt-5.6-sol
gpt-5.6-terra
gpt-6-astra
gemini-3.1-pro-preview
gemini-3.6-flash
gemini-3.7-flash
gemini-3.8-flash
claude-fable-5
claude-opus-4-6
claude-opus-4-8
claude-opus-5
claude-sonnet-5
qwen3.8-max
glm-5.3
glm-5.3-flash
kimi-k3
```

## 当前状态（2026-10-25，持续推进中）

[项目审查](outputs/project_review_20261025_44/REPORT.md)之后，已完成[全行上下文评价与CPU对照](outputs/context_evaluation_20261025_45/REPORT.md)：每场景完整2699行，旧v8严格重建通过；普通LR+style=.435976，Gaussian=.423008、收缩LDA=.395976，固定配方停止。

三折外层方案的正式代码与首折私有禁网待上传包已完成CPU准备；61项相关测试通过，真实22层重载及三折GPU尚未运行。详见[准备报告](outputs/qwen_outer_preparation_20261025_49/REPORT.md)。

Qwen LoRA私有v34已按新增25分钟授权完成固定1epoch：1439行、90次更新、内层20题同运行冻结BA .394444→适配BA .411111，NLL 1.786144→1.748359；题级BA区间跨零，暂属探索性信号。完整权重和逐题配对证据见[运行报告](outputs/qwen_lora_epoch1_20261025_48/REPORT.md)。三折outer协议已做CPU核对，需额外GPU授权；v8默认与公榜.472不变。外部池技术候选3587/146完整块仍因许可口径冲突不训练，见[审计](outputs/external_pool_audit_20261025_47/REPORT.md)。

公榜最佳仍v8=.472（57516416）；原有Windows Kaggle CLI可用且已认证，不需重装或复制凭据。公开0.607大型方案仍缺LoRA24依赖，未复现。隐藏组结构未知，旧mixed不是盲测。最新运行状态以HANDOFF.md顶部为准。

以下带日期的结果、额度与“当前/下一步”均为历史。

首轮优化已完成：char_case 分组 OOF 0.29818，比原字符 0.28633 提高 1.184 个百分点。详见 [OPTIMIZATION.md](OPTIMIZATION.md)。

2026-10-21 更新：已完成 Windows 本机 Kaggle CLI 认证、官方数据下载、规则与数据页面读取，以及首轮 CPU 基线验证。

- [首轮数据与基线结果](BASELINE.md)：2,699 行、150 个 prompt、18 类；字符级 prompt 分组 OOF Balanced Accuracy 为 0.28633。
- [运行与提交配置](SUBMISSION.md)：本轮4次追加提交均已发出，最终最佳公开0.447（v6，提交57482280，COMPLETE）；2026-10-21剩余0次；不自动跨日补交。详见 OPTIMIZATION.md。
- 原始官方页面：official_pages.json；数据：data/；唯一源码入口：src/baseline.py；运行产物：outputs/。
- 用户身份验证链接中的查询参数未保存；账号凭据留在官方工具管理的本机位置。
- 最新规则以本轮 API 取得的 official_pages.json 为依据，原先数据/规则正文缺口已补齐。README 上方简介为初始摘要。

## 工作区归属

- workspace_id：net-4090d-jyx；server_id：net-4090d；SSH 别名：Net-4090D。
- Windows：Z:\Competition\NameThatLLM。
- 远端：/home/yuanyn/jyx/Competition/NameThatLLM。
- 本轮严格主机密钥 SSH 已确认 ubuntu / yuanyn；挂载来源为 yuanyn@Net-4090D:/home/yuanyn/jyx。
- CPU 分析只读复用 AI/.venv 的已有依赖，不修改 AI 或 BigData；代码入口为 src/baseline.py。








## 夜间候选（2026-10-22，尚未提交排行榜）

正式公开最佳仍为v6=0.447。私有第8版已完成真实特征提取、预测及Parquet写入读回，完整prompt OOF=0.55194134；这不是公开分数。两套混合覆盖验证中，原策略→候选为0.449058→0.470501、0.489909→0.525236；统一6条场景仍有折间/类别退步，按prompt重采样的三套区间有两套跨0，不能保证泛化提升。

候选只改两项：按推断组大小降低小组去中心强度；仅推断4–18条组将4B/8B比例由60/40改为30/70。单条和超大组回退；无真实test prompt参与策略。模型、训练数据和style权重不变。无效的尺度修正、配对专用头已放弃，证据见OPTIMIZATION.md。

统一入口的显式参数：

- 候选：`--mode predict --neural center --center-policy adaptive --feature-set group_middle`
- 原v6策略：`--mode predict --neural center --center-policy constant075 --feature-set all`
- CPU缓存诊断：`--mode cached-analysis --data data --cache outputs/kaggle_kernel_v7/frozen_train_features.npz --output <新的输出目录>`；混合覆盖另加`--mixed-coverage --fold-seed 20261022 --sampling-seed 20261024`。

真实预测需Kaggle已有模型挂载/GPU与Parquet支持，不能直接在CPU服务器提取特征。本地无参数默认仍为export-features；检测到官方Kaggle比赛train.csv路径时，使用源码中的固定候选验收profile，显式参数优先。CPU分析只读复用已有环境，不安装依赖。

第8版完整配置、源码核验、模型身份和日志：outputs/kaggle_kernel_v8/run_record.json。混合覆盖记录：outputs/night_mixed_coverage_20261022_23与night_mixed_confirm_20261022_24。GPU累计观察0.89小时，用户授权累计上限8小时；本夜无排行榜提交。当前阶段停止增加调参分支，保持候选可复核，10点统一交付。
2026-10-22 DeepSeek Harness交接时官方查询确认：v8提交57516416 COMPLETE，公开0.472，较v6=0.447提高0.025；当前公开最佳更新为v8。今日此前已用1次/剩余4次，后续需重查。两个自动任务已PAUSED，无本轮Python任务。最新统一交接HANDOFF.md；旧PENDING/不再提交仅为历史状态，不据此重提。


2026-10-22 晚（DeepSeek Harness 第二轮）：第9版 v9b（每 prompt 唯一作者匈牙利指派 + 组大小条件正则）私有 Kaggle 运行 OOF 0.6486，但正式提交 57527432 公开仅 **0.462**，比 v8 的 0.472 退步 0.010。后续审查更正：该组合完整 prompt OOF 未迁移到公榜；稀疏结构只是开发假设，隐藏组组成未知。混合覆盖仅是压力场景之一，且已反复用于开发，不能作为唯一选型依据。已把 Kaggle 默认 profile 回退为 v8 配置，源码 SHA `830f98d8…`。今日已用 2/5 次，剩余 3 次。详见 HANDOFF.md 与 OPTIMIZATION.md。


2026-10-23双异构筛选已完成：Mistral与Gemma单模均不及v8，固定50:50融合均一套升/一套退，全部未达到预定晋级门槛，保留v8公开0.472。本轮GPU约新增.24h（累计观察1.43h），18项测试通过，未提交排行榜。Gemma访问问题经用户自行同意后已解决，旧“403待处理”是历史状态。完整结果见outputs/hetero_20261023_27/REPORT.md及summary.json。
