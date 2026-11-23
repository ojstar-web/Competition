# Enveda CASMI 2026 分子结构识别

整理日期：2026-10-23（北京时间）。本目录与 NameThatLLM、AI、BigData 平级；用于本比赛资料与后续开发。

## 比赛要点

根据 LC-MS/MS 质谱预测小分子的二维结构，输出 SMILES 候选列表。主办方 Enveda，总奖金 50,000 美元。

| 事项 | 时间 |
| --- | --- |
| 开赛 | 2026-09-14 |
| 报名及合队截止 | 2026-12-07 23:59 UTC，即北京时间 12-08 07:59 |
| 最终提交截止 | 2026-12-14 23:59 UTC，即北京时间 12-15 07:59 |

评分为 MRR@25：每个分子首个正确候选位于第 r 名得 1/r，未命中得 0，再对分子平均。匹配使用 RDKit 2026.03.3 互变异构规范化后的 InChIKey14；不要求立体化学一致。

Notebook 离线提交，CPU/GPU 运行均限 9 小时，输出 submission.csv。允许免费公开外部数据和预训练模型。每个 molecule_id 恰好一行，列为 molecule_id、smiles；候选由分号连接、最佳在前，最多 25 个，不允许重复 ID 或空值。

来源：[官方概览](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)。完整页面已保存 official_pages.json；详情及核验限制见 [SOURCES.md](SOURCES.md)。

## 数据概况

官方列出 train.parquet、test.parquet、sample_submission.csv，总计约 3.04 GB；训练汇集约 254 万条谱、27.6 万个结构。预测单位是分子，同一分子的多条谱需要聚合。

主要输入包括碎片 m/z 与归一化强度数组、precursor_mz、adduct、ionization_mode、仪器及碰撞能量。训练额外含结构、分子式、来源库和质量误差信息。注意：可下载测试数据来自训练集，评分时会被隐藏集替换。测试仪器为 timsTOF；训练来源及仪器混杂。官方将 enveda-np-examples 描述为 250 个天然产物；本次文件实测 1,184 条谱，与页面的 1,151 条不同，详见数据审计。

数据页标注 CC BY-NC 4.0；实际使用还须核对比赛规则和各外部资产许可。实际文件已确认：训练 18 列、测试 12 列；训练不含 molecule_id 和 spectrum_id，解释了初轮字段计数疑问。

来源：[官方数据页](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)。

## 当前状态与入口

用户已报名并授权下载数据、核验与首个 CPU 检索基线。三个官方文件现已下载；已创建本项目独立 .venv，固定 RDKit 2026.03.3。未训练神经网络或提交排行榜；正式运行 outputs/baseline_20261023_03 已完成；40 分子已知结构检索 MRR@25 为 0.961111，结构不相交纯检索为 0。详见 [BASELINE.md](BASELINE.md)，不代表隐藏测试成绩。

- [SOURCES.md](SOURCES.md)：来源、证据状态与待核实规则。
- [PLAN.md](PLAN.md)：首轮数据审计、验证及基线建议。
- [BASELINE.md](BASELINE.md)：实际数据、验证结果、预测产物和重跑入口。
- [REVIEW.md](REVIEW.md)：独立科学审查及已修复问题。

data/ 保存官方文件，src/baseline.py 是初轮基线入口，tests/ 保存针对性检查，outputs/ 保存运行记录和预测；references/ 保存官方评分参考，requirements.txt 固定本轮直接依赖。不修改邻近比赛文件或环境。

## 工作区归属

- workspace_id：net-4090d-jyx；server_id：net-4090d；SSH 别名：Net-4090D。
- Windows：Z:\Competition\EnvedaCASMI2026。
- 远端：/home/yuanyn/jyx/Competition/EnvedaCASMI2026。
- 本轮写入前严格 SSH 验证：ubuntu / yuanyn，Competition realpath 符合预期；SSHFS 实际来源为 yuanyn@Net-4090D:/home/yuanyn/jyx。
- 接入表及挂载脚本状态记录滞后，本轮未修改它们；以实际身份核验为依据。SSHFS 用于资料读写，未执行训练环境或 GPU 验证。

## 阶段一更新

已完成冻结结构划分、64 个未曝光开发分子上的参考谱检索比较，以及官方 COCONUT 外部候选的质量排序控制。正式结果、S2/S3 隔离审计、资源限制、失败记录和回退见 [PHASE1.md](PHASE1.md)；公开资源和模型适用性见 [SOURCES.md](SOURCES.md)。目前 A 分箱检索仍为已知结构有参考谱时的默认基线；未训练或调用神经模型，未提交 Kaggle，70 个封存结构未评分。新阶段入口为 `src/phase1.py`（固定清单）、`src/phase1_eval.py`（A/B 与参考隔离）、`src/coconut_candidates.py`（外部候选与质量控制）；旧 `src/baseline.py` 未改。

## 阶段二更新

已在独立 CPU 环境完成候选分子式审计及 MIST main_v2 小规模适配。固定 64 个开发结构中仅 9 个满足候选池唯一中性式和 positive `[M+H]+` 谱，20 条谱用于模型；其余 55 个保持原质量排序。9 结构同池 MIST MRR@25 为 0.87037，对照 mass_only 0.10159；全 64 的 9+55 回退系统为 0.27716，对照 0.16905。预测先冻结后评分，70 个 sealed 未运行。结果是选定开发子集的描述性证据，非隐藏集、泛化或新结构生成成绩；详见 [PHASE2.md](PHASE2.md)。有参考谱的 S1 仍用 A 分箱基线；外部候选排序分支入口为 `src/mist_phase2.py`，环境为 `.venv_mist_cpu`，不改原 `.venv` 或 `src/baseline.py`。


## 阶段三更新

剩余 76 个开发结构已按冻结方案使用同一原始 COCONUT 快照独立构建质量窗候选。6 个预定单公式/H+ 结构使用 17 条谱完成 MIST CPU 排序，其余 70 个保留质量排序；全 76 的 MRR@25 为 0.23653，对照 0.19698。技术复现成功，但 6<预设 8 的效果支持门槛，只能描述观察结果。预测先冻结后评分、70 个 sealed 未评分；证据、资源和适用边界见 [PHASE3.md](PHASE3.md)。仍用 `src/mist_phase2.py` 的 `--members` 路径和既有独立环境，不影响 S1 A 检索或阶段二历史产物。

## 阶段四更新

在已曝光的阶段三 76 个开发结构上，仅对其中 69 个多分子式结构做无真值公式解释力审计，59 个可处理结构完成 759 个公式×H+ 谱配对；固定 0.50/0.05 门槛使 7 个通过、52 个因公式间隔不足回退。通过者的 18 条谱用原 MIST 权重做所选公式内部重排，116 个候选指纹跨 RDKit 逐位一致。全 76 的原 hybrid→公式层 F→分层 FM 的 MRR@25 为 0.23653→0.23912→0.28760，Top1 为 10→10→15，Top25 始终 53；其余 69 个 qid 逐列表保留原 hybrid。FM 是开发候选，F 是分解对照；不是独立验证或隐藏集成绩。协议、成本、完整性、资源口径和回退见 [PHASE4.md](PHASE4.md)；70 sealed 继续封存。

## 阶段五统一预测阻断（2026-10-23）

140 个 fresh_dev 的统一预测输入和固定 61 补位候选已冻结，但历史逐列表硬回归未通过：原 76 的 FM 预测 76/76 相同，旧 9 个单公式 MIST 仅 4/9 相同。原因指向官方 attention 对补零峰的掩码未生效，使旧批长 58 与新批长 61 的向量不同；固定 61 只稳定批次上下文，不修复模型。**没有开展阶段五真值效果评价，也未将固定 61 采用为新基线。**证据、哈希、资源和回退见 [PHASE5.md](PHASE5.md)。70 sealed 仍封存；阶段二至四历史结果保留原批次条件。
## 固定 61 开发基线更新（2026-10-23）

用户授权后，已对冻结的 140 个 fresh_dev 预测按固定 61 口径一次评分：FM / 单公式分支 / 质量排序的 MRR@25 为 0.29773598 / 0.25510398 / 0.18420817。原阶段五历史回归失败及旧结果继续保留；本轮是已曝光开发集基线，70 sealed 未评分。协议、三表、两例下降、资源与回退见 PHASE5.md 和 outputs/phase5_20261023_13/。

## Kaggle 提交状态（2026-10-23）

已完成无真值 test-only 入口、便携 CPU 资产和本地 4/16 分子 CSV 验收；私人资产在 Kaggle 上暂不可作为 Notebook 数据源，故**尚未正式提交、没有比赛分数**。资产/Notebook 路径、哈希、403 与失败云端日志、待用户核实事项见 [SUBMISSION.md](SUBMISSION.md)。
