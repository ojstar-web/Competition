# 首轮数据审计与 CPU 检索基线

2026-10-23 完成。正式验收运行：outputs/baseline_20261023_03。仅为本地基线，没有 Kaggle 排行榜成绩。

## 数据实测

| 文件或内容 | 实测 |
| --- | --- |
| train.parquet | 2,539,608 条谱，18 列，3,033,286,496 字节 |
| 训练结构 | 按文件提供的 inchikey14 共 275,810 个；不等于重新按评分规范化后的精确去重数 |
| test.parquet | 1,213 条谱，400 个 molecule_id，12 列，4,848,729 字节 |
| sample_submission.csv | 43,619 字节，400 个 ID |
| enveda-np-examples | 1,184 条谱；与官方文字说明中的 1,151 不同，以上为本次文件实测 |

可下载测试是训练集样例，评分重跑时会被隐藏集替换。不能用其可检索性声称隐藏测试准确率。

测试样例七种 adduct 的谱数：质子化 959、去质子化 193、甲酸加合 31、钠加合 22、铵加合 4、钾加合 2、氯加合 2。代码支持官方隐藏测试列出的全部十种加合物；额外训练加合物中未支持的类别被显式计数跳过，见 data_audit.json。

## 方法与验证划分

使用官方训练谱做谱库检索，不拟合神经网络。先按前体及 adduct 换算中性质量预筛，候选结构须为 RDKit 可解析的中性单组分；最终以结构精确质量做 15 ppm、最低 0.003 Da 过滤。强度开方、最多 64 峰，碎片及中性丢失按 0.02 Da 分箱，余弦证据权重为 0.7/0.3；每个查询谱取同候选结构最优参考谱得分，再对该分子的全部查询谱平均。

规范化顺序与官方一致：先去立体化学，随后规范互变异构，比较 InChIKey14；固定 RDKit 2026.03.3。

按种子 20261023 从 enveda-np-examples 的规范结构选择 40 个验证分子，同结构所有查询谱作为一组，清单见 split.json。整个 enveda-np-examples 来源都不进入参考库，避免查询谱自匹配。

| 场景 | 分子数 | MRR@25 | Top1 | Top5 | Top25 | 解释 |
| --- | --- | --- | --- | --- | --- | --- |
| 已知结构谱库检索 | 40 | 0.961111 | 95% | 97.5% | 100% | 其他来源可以保留相同结构的参考谱；最终质量过滤后的真值覆盖率为 100% |
| 结构不相交检查 | 40 | 0 | 0 | 0 | 0 | 从所有实际使用的参考谱排除这 40 个规范结构；纯检索无法返回不存在的真值 |

第一组衡量天然产物已知结构检索；该来源刻意包含常见产物，不能代表隐藏测试三类新颖度的整体分布。第二组建立了结构分组清单与排除检查，但尚无候选库外结构预测能力，也没有非零的结构留出泛化基线。小样本成绩不提供完整性能保证。

## 运行与产物

- 服务器 net-4090d，实际 ubuntu / yuanyn；项目 /home/yuanyn/jyx/Competition/EnvedaCASMI2026。
- 环境 .venv，Python 3.12，直接依赖 requirements.txt，全依赖 outputs/environment_20261023.txt。
- 原预筛 518,562 条谱；最终参考库 355,296 条、65,553 个规范结构。每结构/adduct 最多接纳文件顺序前 8 条有效谱，跳过数量见 data_audit.json。
- 运行 246.14 秒，峰值 RSS 4,755.40 MiB；CPU 线程配置为 2，无 GPU，进程超时 900 秒、虚拟内存上限 12 GiB。
- submission.csv 共 400 行，ID 顺序及集合与样例一致；质量窗口外回退 0 次。所有候选已解析，最多 25 个，规范结构无重复。
- 验收包含源码及预测 SHA256、真实 CSV 回读、验证指标重算、8 个案例与下载的官方规范化函数对照；verification.json 状态 passed。

主要文件均在 outputs/baseline_20261023_03/：run_record.json、data_audit.json、split.json、metrics.json、diagnostics.json、validation_predictions.json、submission.csv、verification.json。

## 重跑入口

在已重新核验身份和根目录的普通 SSH 会话内运行，不通过 Windows 挂载执行 Linux 计算：

```bash
cd /home/yuanyn/jyx/Competition/EnvedaCASMI2026
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 ARROW_NUM_THREADS=2 \
  .venv/bin/python src/baseline.py --data data --output outputs/<新的唯一运行标识>
```

输出目录已存在时会拒绝覆盖。单元检查入口为 `.venv/bin/python -m pytest -q tests/test_baseline.py`，现为 6 项通过。`tests/verify_run.py` 用于核验本次指定运行，不能替代未来运行的路径调整。

## 失败与审查记录

- baseline_20261023_01：预筛超过 50 万参考谱保护，停止；run_record 保存 failed。
- baseline_20261023_02：独立审查发现限额先于有效性检查的问题，主动停止并标记 aborted_for_review_fix。
- 已修正为仅对通过结构和峰检查的谱计数，真实微型 Parquet 回归先失败后通过。详见 REVIEW.md 和 outputs/tests_20261023.log。

固定分箱会漏掉邻箱近质量峰，前 8 条有效谱也有来源/能量顺序偏差。后续优先补充公开天然产物候选库与无参考谱结构的验证，之后再考虑指纹预测或重排序；本轮未下载外部库、上传 Notebook、安装 GPU 依赖或提交排行榜。

## 回退

本轮未改动相邻比赛、全局配置或原有环境；没有提交、推送或发布。保留当前数据和失败记录即可检查与重跑。若不采用该基线，停止使用 src/baseline.py 即可；不要覆盖或删除历史运行来回退。
