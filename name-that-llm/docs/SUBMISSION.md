# 2026-10-25 持续推进说明

三折Qwen outer已完成CPU代码、合成全流程汇总验证与首折私有禁网待上传包，尚无新的GPU授权或外层结果；详见`outputs/qwen_outer_preparation_20261025_49/REPORT.md`。

当前正式源码保留v8默认推断。私有GPU v34完成固定1epoch（单作业1028.93秒），仅fold0内层BA .394444→.411111，区间仍跨零；已保存但未真实重载22层adapter，未外层OOF，不能作为推断提交版本。独立审查建议固定方案三折开发OOF；额外GPU尚未授权。详见HANDOFF.md顶部和outputs/qwen_lora_epoch1_20261025_48/REPORT.md。公榜最佳仍.472，本轮无新榜交。

---

# 当前运行入口与提交事实（2026-10-25）

本轮实测下述原有本机 Kaggle CLI 可用并已认证；查询返回 7 次正式提交，最佳 v8=0.472（57516416），最近 v9b=0.462（57527432）。不需要重装 CLI 或复制凭据。

公开强基线私有复现 ntl-atlas-public-repro v2 缺少 LoRA24 权重而失败，无新提交。正式 src metadata 虽保留 CPU 标题，实际 enable_gpu=true；源码当前 Kaggle profile 是 v8 adaptive/group_middle。云端诊断版本与本地默认不能混为一谈。

下文 CPU/300秒/v1-v6 命令、额度和状态均为当时历史，**不应直接作为当前GPU作业运行方案**。提交必须绑定已验收的具体kernel/version/file。--timeout 的现有日志不足以证明整项作业硬时限；运行预算要包含模型加载、所有分支、隐藏规模和输出。

完整历史审查、失败原因与下一步见 [项目审查报告](outputs/project_review_20261025_44/REPORT.md)；实时提交证据在同目录 submissions_20261025.json。本轮没有新上传、训练或榜交。

---

# Kaggle 运行与提交

本机工具：C:\Users\JYX\AppData\Local\Programs\KaggleCLI\Scripts\kaggle.exe
已认证账号：okkojbjj。认证文件由官方工具管理，不复制到服务器或项目。

配置：src/kernel-metadata.json；源码：src/baseline.py。
目标：okkojbjj/name-that-llm-cpu-baseline，私有，CPU，禁网，绑定 name-that-llm。
默认入口在 Kaggle 官方数据路径读取 train.csv/test.csv，以字符 TF-IDF 全量拟合，输出 submission.parquet，并回读核验。
不把本地可见三行 test 当作完整评分数据。

## 命令（PowerShell）

先设置本次会话工具变量：
```powershell
$kg = "$env:LOCALAPPDATA\Programs\KaggleCLI\Scripts\kaggle.exe"
```

上传并私有运行（300 秒上限）：
```powershell
& $kg kernels push -p 'Z:\Competition\NameThatLLM\src' -t 300
& $kg kernels status okkojbjj/name-that-llm-cpu-baseline
```

运行完成后，应检查日志及生成的 submission.parquet，并核对这次上传的具体版本。
实际排行榜提交需要显式指定版本，下面的 $version 必须来自已验收版本：
```powershell
& $kg competitions submission-limits name-that-llm
& $kg competitions submit name-that-llm -k okkojbjj/name-that-llm-cpu-baseline -v $version -f submission.parquet -m 'Character TF-IDF CPU baseline'
& $kg competitions submissions name-that-llm
```

只上传/运行不会消耗比赛排行榜提交次数；submit 才是正式评分请求。此文件中的命令尚不代表已执行成功。

## 当前验收结果

2026-10-21 用户明确授权上传本轮源码并私有运行后，已上传第 1 版：
https://www.kaggle.com/code/okkojbjj/name-that-llm-cpu-baseline

官方状态 COMPLETE。CPU、禁网、无 GPU，300 秒运行上限；日志显示约 18.6 秒时完成 3 条可见测试预测的写入及 parquet 回读校验。Kaggle 系统转换组件有 SyntaxWarning，不影响完成状态。

产物保存在 outputs/kaggle_kernel_v1/：
- submission.parquet：已下载的预测文件。
- name-that-llm-cpu-baseline.log：Kaggle 运行日志。
- run_record.json：版本、源代码/配置/预测文件哈希和验收记录。

首次下载连接被重置，核对目录后重试成功，没有重复上传或重跑。
本轮只验收私有运行；未请求排行榜评分，账号累计提交 0 次、今日剩余 5 次。
下一步正式提交可指定 -v 1，仍需用户授权排行榜提交。本地运行完成不代表隐藏测试已通过，也不代表有排行榜分数。

## 首轮优化更新

私有第 2 版已完成。默认预测特征为 char_case，具体结果和限制见 OPTIMIZATION.md，产物与源码哈希见 outputs/kaggle_kernel_v2/。上面的第 1 版验收记录保留为历史；当前候选是第 2 版。尚未提交排行榜。

## 首次正式评分：2026-10-21

用户明确要求先提交确认正确性。已提交私有第 2 版，submission ID=57472229，状态 COMPLETE，Public Score=0.308。隐藏测试重跑和输出格式已通过；这不是科学正确性或泛化性能的全面保证。
查询时公开榜首 0.667，差 0.359（35.9 个百分点）；榜单前15末位约 0.622，与当前基线仍有明显差距。本地分组 OOF=0.29818，公开榜比其高约 0.00982，两者评测样本不同，不能据此宣称交叉验证无偏或过拟合不存在。

榜单只提供分数，不能确定领先方案的训练方法、数据来源或逐类收益。当前特征只是稀疏字符片段；应先审核可复现的公开强基线及其许可、数据和验证，再决定语义表征或预训练模型实验。下一轮不宜只围绕0.01级的字符参数改动反复耗费提交额度。
记录与榜单快照：outputs/submission_57472229/。本次使用1次提交额度，剩余4次；没有追加其他提交。

## 本轮4次追加提交汇总

| 版本 | 改动 | 分组OOF | 提交ID | 公开榜 |
|---|---|---:|---:|---:|
| v2（起点） | 保留大小写字符 | 0.29818 | 57472229 | 0.308 |
| v3 | 多视图风格 | 0.35523 | 57480764 | 0.361 |
| v4 | 冻结Qwen4B浅层+style | 0.39743* | 57481072 | 0.398 |
| v5 | 加入Qwen8B中间层 | 0.43375 | 57481530 | 0.421 |
| v6 | 加入内容去中心头 | 0.53267 | 57482280 | 0.447 |

* v4使用Kaggle当时重建的分组折，同环境style对照为0.36413；v5/v6已固定原始prompt折，style精确复现0.35523，不直接将v4与v5的OOF视为严格同折比较。
原始已提交1次，本轮按授权再提交4次；2026-10-21当前提交5次，剩余0。不得再提交或以新的一天自动补交。

### v6实际发现与验证

在相同冻结特征、样本和折上，普通头控制概率与v5逐元素一致（max abs diff=0）。去中心混合的OOF由0.43375升至0.53267，增加0.09892（9.892个百分点）；已从保存的OOF概率独立重算核对。
去中心头自身OOF=0.53416；实际部署保留普通头和20%style，以覆盖不完整组和单条。未按公开分数调权，未强制类别数量平衡。
v6私有运行COMPLETE，约866秒，3条可见预测的parquet回读通过，证据在outputs/kaggle_kernel_v6/。最后正式提交57482280待隐藏评分，不能将0.53267当公开榜成绩。

截至最后提交，Kaggle quota累计GPU used=0.52h（本轮初始0），后续最后评分仍会消耗配额；远低于8h上限。每次GPU作业设置3600秒上限，没有安装依赖或修改AI项目。
正式最佳已验证公开成绩为v5的0.421。最后候选是否成为最佳需等最终评分；如退步，保留v5作为最佳公开成绩，不擅自撤销或改最终入选提交。

## 最后评分完成（2026-10-21 北京时间21:05核查）

提交57482280（v6内容去中心），官方状态COMPLETE，公开榜0.447。
相对v5=0.421提高0.026（2.6个百分点）；相对起点0.308提高0.139（13.9个百分点）。当前已验证公开最佳为v6。
当日总提交5次，剩余0。本轮额外4次均有正式结果：0.361、0.398、0.421、0.447。
本地OOF0.53267与公开0.447相差约0.08567；不能将本地9.89个百分点提升外推为公开提升，实际仅2.6个百分点。下一轮应优先校验不完整prompt组、误分组和反复使用开发验证的选择偏差，不继续叠加模型追分。
本次查询GPU quota累计used=0.52h；官方配额可能有更新延迟，该观察不是逐作业计费明细。
只完成记录与通知，停止评分监控，不启动新实验、上传或提交；后续遵循STRATEGY.md，每日默认0–2次，不以用完额度为目标。

2026-10-22 用户明确授权后已提交冻结私有v8，提交ID57516416，UTC08:08:39，当前PENDING，公开分数未出，今日剩余4次。只提交一次，未重训/新上传；上一公开最佳仍v6=0.447。本地OOF0.55194134不是公开分数。记录outputs/submission_57516416/record.json。


2026-10-22 DeepSeek Harness交接时官方查询确认：v8提交57516416 COMPLETE，公开0.472，较v6=0.447提高0.025；当前公开最佳更新为v8。今日此前已用1次/剩余4次，后续需重查。两个自动任务已PAUSED，无本轮Python任务。最新统一交接HANDOFF.md；旧PENDING/不再提交仅为历史状态，不据此重提。

