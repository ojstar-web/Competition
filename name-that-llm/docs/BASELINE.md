# 首轮检查与 CPU 基线

日期：2026-10-21。账号认证由 Windows 本机 Kaggle CLI 2.2.4 完成；不在项目保存凭据。

## 已取得官方信息

原始页面保存在 official_pages.json，由已认证 CLI 读取，包含 data-description、rules、foundational-rules、Evaluation 等页面。每日 5 次提交、最多 5 人组队、可选 2 次最终提交。数据访问和使用栏标记 CC BY-SA 4.0，获奖许可栏标记 CC-BY 4.0；完整条款另含数据访问保护和获奖源码交付要求，应一并阅读原文。

专属规则允许公开可获取或符合合理成本条件的外部数据/模型；本比赛代码要求还明确免费公开外部数据、禁网、CPU/GPU 最多 9 小时。本轮不使用外部数据。

## 数据实测

- train.csv：2,699 行，字段 id、prompt_id、text、model。
- test.csv：3 行可见示例，字段 id、text；不是完整评分集。
- 18 个模型：claude-fable-5 为 149 条，其余每类 150 条。
- 150 个 prompt：149 组各 18 条，1 组 17 条。
- 无空字段、重复 ID、完全重复文本；转小写并合并空白后也无重复文本。
- 可见训练/测试无 ID 或完全相同文本交集。
- 空白分词长度为 36–274；这不等同于官方词数处理方式。
- 尚未做语义近重复检测；不能据精确匹配断言所有 prompt 独立。
- 官方数据说明明确隐藏测试 prompt 不出现在训练集，采用按 prompt 分组验证。

## 实际运行

项目：/home/yuanyn/jyx/Competition/NameThatLLM。
服务器：Net-4090D，hostname=ubuntu，user=yuanyn。
解释器：/home/yuanyn/jyx/Competition/AI/.venv/bin/python，仅复用已安装依赖，不修改 AI 源码或环境。
资源：CPU 2 线程、超时 300 秒、无 GPU；实际分析耗时约 15.6 秒。

统一入口 src/baseline.py。3 折，seed=20261021；TF-IDF 最多 50,000 特征，词 1–2 gram/字符 3–5 gram，min_df=2；分类器 LinearSVC，C=1，balanced 类别权重。
向量器仅在每折训练数据拟合，分组划分断言训练/验证 prompt 无交集。

| 划分 | 词级 OOF Balanced Accuracy | 字符级 OOF Balanced Accuracy |
|---|---:|---:|
| 随机分层 | 0.09556 | 0.13556 |
| prompt 分组 | 0.25149 | 0.28633 |

本轮按分组指标选择字符级。随机划分分数更低，因此没有证据说它在本数据上高估了成绩；可能与同 prompt 的多作者回答结构有关，但此解释未经验证。
这些是本地交叉验证分数，不是 Kaggle 排行榜成绩，也不是性能上限。

## 产物与复核

outputs/cpu_baseline_20261021_01/ 含：
- data_profile.json：数据统计。
- config.json：最终参数、环境版本、服务器、源代码及小数据文件 SHA256。
- summary.json：折分数、OOF 分数与所选特征。
- *_recall.csv：每类召回率。
- *_confusion.csv：混淆矩阵。
- *_oof.csv：对应 ID/prompt 的折外预测。

运行日志为 outputs/cpu_baseline_20261021_01.log，退出码 0。
源码检查覆盖训练折预处理、分组隔离、ID/标签输出断言和禁止覆盖已有输出。
本机未安装 parquet 分析依赖；预测入口已在 Kaggle 私有第 1 版完成 parquet 写入/回读验证，详见 SUBMISSION.md 和 outputs/kaggle_kernel_v1/。
目前只有一个基线入口，不建立模型管理框架，不复制源码迭代，不提交或推送 Git。


