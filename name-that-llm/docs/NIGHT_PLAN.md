# 夜间优化：2026-10-21 至 2026-10-22 10:00（北京时间）

用户已明确要求持续优化至明天10点。此授权覆盖旧记录中暂停实验的限制；仍遵循STRATEGY.md节约提交要求。本夜及跨日不自动提交排行榜，10点交付后等待用户决定。

## 当前真实状态

- 当前公开最佳v6：0.447，提交57482280，已COMPLETE；本地固定prompt折OOF=0.532667。
- 当前存在约8.57个百分点验证-公开差距，优先做验证结构诊断。
- 今日5次额度已用完，不能以补交为夜间目标。
- Kaggle第7版已启动：仅导出训练冻结特征，默认模式export-features，不产生排行榜提交。
- 同一源码src/baseline.py；夜间CPU分析模式cached-analysis已实现；四项行为检查通过。新增分析入口尚未实际完成运行，不能声称已验证结果。
- 作业记录outputs/kaggle_kernel_v7/run_record.json。

## 身份和环境

workspace net-4090d-jyx → server net-4090d → SSH Net-4090D → ubuntu/yuanyn。
Z:\Competition\NameThatLLM → /home/yuanyn/jyx/Competition/NameThatLLM。
CPU依赖只读复用/home/yuanyn/jyx/Competition/AI/.venv/bin/python，不修改AI项目或环境。
Kaggle CLI：C:\Users\JYX\AppData\Local\Programs\KaggleCLI\Scripts\kaggle.exe，账号okkojbjj，私有脚本okkojbjj/name-that-llm-cpu-baseline。
不读或复制凭据，不向其他服务器迁移代码/数据。

## 下一步操作

1. 查询私有第7版状态。已完成则下载frozen_train_features.npz、frozen_feature_manifest.json和日志至outputs/kaggle_kernel_v7/。预计特征约170MB，为本项目派生产物，保留行ID、模型源版本和训练数据SHA256。网络重置先检查已有文件/产物再有界重试，不能重跑已完成GPU。
2. 在已核验项目根目录运行：
   /home/yuanyn/jyx/Competition/AI/.venv/bin/python src/baseline.py --mode cached-analysis --data data --cache outputs/kaggle_kernel_v7/frozen_train_features.npz --output outputs/night_partial_20261021_01
   CPU2线程，timeout1200。日志outputs/night_partial_20261021_01.log。首次验证与分析必须检查退出码和关键错误。
3. 若以SSH后台运行，使用项目内唯一锁/记录PID，后续先检查PID、日志和产物，避免重复。不能通过远端Codex或子代理运行。
4. 分析已实现每prompt仅1/2/3/6/18条，普通头与去中心权重0/0.25/0.5/0.75/1，固定原始150个prompt折。真实prompt仅用于采样和诊断，预测分组来自无标签文本。oracle组结果只作诊断参考，既不可部署，也不是数学上保证的性能上界。
5. 根据诊断提出单一改进，优先复用缓存：分组置信度回退、组大小相关混合、去中心训练组大小、冻结层消融、分类头正则化；强候选在预先约定第二组prompt划分上复核，不声称是未见过的盲测。
6. 保留失败及退步场景；除非有证据需要新表示，不重复跑GPU。不得为填满时间或额度做无依据实验。

## 资源与截止

Kaggle GPU总授权先限8h，包含此前使用；本夜起始官方quota used约0.53h（可能有更新延迟），每个新作业必须核对累计。
用户单次上限2h；实际采用3600秒。仅KaggleGPU，不使用本服务器GPU、不安装依赖。
08:00后不启动新GPU；09:30后不启动新CPU；所有任务timeout必须落在10点之前。10点停止启动新实验并汇总实际结果，未完成明确列出，不编造。

## 自动跟进

- name-that-llm：每20分钟继续有意义的实验与检查，无实质变化时安静，完成改进或实质失败时报告。
- name-that-llm-2：明天北京时间10:00汇总；完成后暂停这两个任务，不能自动开启新一轮。
- 旧“最终评分监控”已被夜间任务替代，最终评分已完成，不再重复查询或通知。
