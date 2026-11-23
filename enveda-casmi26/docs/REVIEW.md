# 科学语义与代码审查记录

日期：2026-10-23。审查依据为本轮源码、针对性测试、下载的官方 metric 内容以及主代理提供的运行证据。独立审查者未执行 SSH、访问数据或修改文件。

## 已处理问题

审查发现，先截断每个身份的前 8 条谱、再检查有效性，会让早期无效谱挤掉后续有效谱。旧实现通过真实微型 Parquet 回归复现失败，日志为 outputs/test_invalid_early_red.log。

正式实现改为先通过结构、单体/电荷和峰检查，再按规范结构键及 adduct 对实际接纳谱计数。已有六项测试覆盖：加合物、官方规范化回归、首命中排名、峰对齐、多谱均值、结构排除，以及无效早期谱和有效谱数量限制。结果见 outputs/tests_20261023.log。

独立复审确认上述问题已关闭；紧凑数组构建稀疏矩阵没有显示新的科学语义问题。完整数据运行与产物仍由主代理另行验收。

## 明确的适用边界

- 0.02 Da 固定分箱是粗检索近似，邻箱的近质量峰可能无法匹配；跨加合物的中性丢失证据也可能较弱。
- 每个规范结构/adduct 只保留文件顺序前 8 条有效参考谱，可能偏向先出现的来源和能量；不等同于全库所有谱检索。
- 已知结构验证仅检查谱库检索能力。结构不相交场景将真值移出谱库，其零召回是该方法限制，不能解释为已建立未知结构预测能力。
- 按分子记录最终质量过滤后的真值覆盖情况，避免把候选缺失混同为排序错误。
- 可下载测试是训练样例，不能用其命中率声称隐藏测试准确率。

## 阶段一独立科学复审与记录纠正（2026-10-23）

GPT-6 Astra high 基于执行代理提供的关键源码原文、固定清单/候选哈希、12 项有界测试、64 分子逐结构名次及运行审计复审；其执行视图不能访问 Z:，没有声称亲自 SSH 或重跑数据。复审确认：分子规范键与官方顺序一致；旧 40 个结构排除、70 个 sealed 不评分；A/B 使用相同参考与质量池，B 限于 A top200 并按同结构参考谱最大、查询多谱均值聚合；COCONUT 候选在真值评价前独立冻结。复审发现 B 贪心匹配二分搜索的 ppm 上边界曾漏掉极窄的 r>q 情形；已用精确上下界修复，并以双向内外边界测试核验。修后同一 64 结构重跑 `_06`，A/B 数值与修前 `_04` 相同，最终解释以 `_06` 为准。审查结果只覆盖给定源码/测试/输出证据，不代表隐藏测试或模型性能验收。

`outputs/phase1_20261023_04/metrics.json` 旧字段 `S2_spectral_mrr25=0` 表述错误：真结构从参考谱删除后，纯光谱检索对该真值的理论召回为 0，本轮没有单独运行 S2 光谱性能。历史文件原样保留；`_06` 改为 `S2_reference_audit_only=true`，记录剔除 1,892 条谱、查询键交集 0。S3 只在冻结候选上临时移除 64 个查询键，25,186→25,122、交集 0，见 `_08/s3_isolation_audit.json`；不提供生成模型性能。

S1 只能称“有参考谱支持、按存储内容精确去重的开发集检索”。训练集没有 spectrum/measurement ID，内容签名对峰顺序与浮点调整不鲁棒，同测量改写副本可能残留，不能称测量独立。S2 的 COCONUT mass-only 是另一参考条件；全 64 在本次宽窗与最终质量池覆盖，但 CSV 窗外行没有逐条规范化，无法推断全库或隐藏测试覆盖。MIST/DreaMS 未下载权重、未运行模型；训练数据重叠及独立权重许可尚未核定。下一步不从真值构造模型母体式。

## 阶段二独立科学复审（2026-10-23）

GPT-6 Astra high 基于执行代理提供的输入/排序关键源码、冻结哈希、20 谱预处理及推理形状、逐结构名次、资源日志和 2 项针对性测试审查，没有 SSH 或自行重跑。复审确认：候选唯一中性式来自先冻结的 COCONUT 池，模型标签无真值 SMILES/分子式；20 条 H+ 谱全部有 35–60 个处理峰和 23–57 个匹配子式；预测谱名及 9/64 qid 与固定清单相等，55 个 fallback 的 top25 与 mass_only 逐项完全一致，预测哈希先于加载真值评分冻结。9 个同池结构的 MIST MRR 0.87037 对照 mass_only 0.10159，全 64 hybrid 0.27716 对照 0.16905；9 例名次均改善。审查批准将 MIST 保留为单公式 H+ 外部候选重排的开发分支，不能据此称隐藏集提升、未见结构泛化、稳定模型或生成新结构。S1 A 参考谱检索属于不同场景，仍保留默认。权重独立条款和预训练数据重叠尚不明；详见 [PHASE2.md](PHASE2.md)。前文“模型未运行”是阶段一历史状态，不是当前状态。

## 阶段三独立科学复审（2026-10-23）

GPT-6 Astra high 依据执行代理提供的固定 76 成员/协议、候选与模型输入 SHA、rank/evaluate 关键源码、139/139 跨 RDKit 逐位指纹、17 谱有效输入和推理输出、预测先冻结后真值评价证据及 4 项直接相关测试复审通过；未 SSH、未自行运行模型。确认 q064–q139 分母和预定 6 个可用结构不因效果调整，70 个回退 top25 逐项不变，模型无失败，70 sealed 未评分。全 76 MRR@25 质量排序 0.19697615→hybrid 0.23653418，预定 6 例 0.12856125→0.62962963。固定方法技术复现成功；6<预写 8，效果只作描述性观察，不声称预设效果门槛通过、隐藏集或预训练未见结构泛化。宽窗索引覆盖仅限本批查询，不能等同完整 COCONUT 覆盖；见 [PHASE3.md](PHASE3.md)。

## 阶段四独立科学复审及资源口径（2026-10-23）

GPT-6 Astra high 基于执行代理提供的冻结协议/759 完整配对、7/52 公式门槛、116/116 逐位指纹、18 谱官方子式与模型输出、F/FM 排序与冻结后评价关键源码、逐结构名次及小文件哈希链复审通过；审查代理未 SSH 或自行运行模型。7 个 gate qid 的 F/FM 均在完整候选池上只改变所选公式内部/公式优先顺序，69 个非 gate 两臂 top25 逐列表等于旧 hybrid，70 sealed 未评分。全 76 的旧/F/FM MRR@25 为 0.23653418/0.23911651/0.28760231，Top25 均 53/76；gate7 后验真值公式覆盖 7/7 只属已曝光开发诊断。审查决定保留 FM 为开发候选、F 为分解对照，不能推断隐藏集、预训练未见结构或校准公式概率；详见 [PHASE4.md](PHASE4.md)。

阶段三资源文字只列出了各步骤的 RSS，其中推理为 657,184 KiB；其原始谱输入导出没有独立 RSS 日志，因此不应称推理数值为整阶段内存峰。阶段四全 166 谱导出实测 3,305,200 KiB，是阶段四已记录步骤中最大值；这不能反推为阶段三测量。两个阶段都未突破 12 GiB 限制，原始记录保留。

## 阶段五独立科学复审与阻断（2026-10-23）

Astra 基于执行代理提交的官方源码片段、同谱 58→61 单变量诊断、masked 位置实际注意力、固定 61 两谱不变性、66 向量和 140 预测哈希、旧 9/76 逐列表比较作独立判断，没有亲自 SSH。官方 attention 现行路径把 bool mask 加到 logits，补零键因此取得非零权重；固定 61 只固定上下文，未修补网络。原 76 FM 76/76 回归，旧 9 单式仅 4/9，q006/q033/q044/q045/q059 变化，故原统一封装验收失败。阶段五未作新真值效果评价，历史指标仍只对应其历史批条件，70 sealed 未评分；完整证据见 [PHASE5.md](PHASE5.md)。固定 61 是否成为新开发基线需用户后续明确决定。

[作者仓库公开 issue #17](https://github.com/samgoldman97/mist/issues/17) 对固定 commit 同一 mask 行有第三方报告，该旧仓库 issue 仍 Open、未见作者在其中确认或合并；作者新仓库 coleygroup/MIST 固定 commit 48e339be6535546c87ba4f6b81a0f787c2e4ac40 已在 scaled/additive attention 使用 new_attn_mask，但其模型/权重/运行栈不同，本项目未验证。fixed61 已在用户授权后成为旧权重的开发评价口径，只固定补位上下文，不是上游修复。

## 固定 61 新开发基线复审（2026-10-23）

用户明确授权新口径后，Astra 依据执行代理提交的 src/phase5_evaluate.py 评分关键源码、预测/成员/61 协议 SHA、先冻结后评价证据、三政策各 140 列表的官方 canonical key 有效性及无重复检查、后 76 历史指标等价、逐 qid 排名和资源记录独立复审通过，未亲自 SSH 或重跑。全 140 FM MRR@25 0.29773598，对照 mass_only 0.18420817 和 single_formula_hybrid 0.25510398；FM 对 mass 23 胜、115 平、2 负，q030 与 q049 下降。后 76 FM 0.28760231 与旧阶段四相同；前 64 旧 9 单式榜单仍仅 4/9 完全同，原阶段五硬回归 FAIL 不变。新运行 outputs/phase5_20261023_13/ 的协议 SHA 53655e4b5d466bf308669fe27950d25c61bc14dab1bdd51ef924ca5c948a681e、指标 SHA c38b650b8a545f6e73ff5498ab0656ace9644db54b4b46c96c393b2c1d121129。结论仅是已曝光开发比较基线，不表示独立验证、隐藏集泛化或修复官方 mask；70 sealed 未评分。详见 PHASE5.md。

## 全项目只读回顾（2026-10-25）

本节核对 `BASELINE.md`、`PHASE1.md`–`PHASE5.md`、冻结小型指标/政策文件、提交入口和官方 Kaggle 状态；没有重跑模型、读取封存 70 个结构的结果或改变源码。140 个 fresh_dev 来自 250 个有效 NP 结构中剔除已曝光 legacy40 后的固定哈希划分，**先于 COCONUT 候选与效果评价**；后验最终质量池覆盖 64/64+76/76，只能说明这个 NP 来源开发集，不能当隐藏集覆盖或独立泛化。

| 证据 | 已观察到什么 | 解释边界 |
| --- | --- | --- |
| 旧 40、fresh64 的 S1 已知结构谱库检索 | MRR@25 分别 0.961111、A 0.9440104；同 64 的贪心峰重排 B 为 0.9309896。参考库约 355,296 谱/65,553 规范结构，原基线 246.14 秒、峰值 RSS 4,755.40 MiB。 | NP 来源整体不入参考，但其他来源可有同结构谱；仅存储内容精确去重，没有测量 ID，不能证明测量独立。当前提交入口没有调用谱库 A。B 未胜 A。见 `BASELINE.md:25-39`、`PHASE1.md:7-24`、`src/kaggle_submit.py`。 |
| 外部 COCONUT 候选排序 | 固定 61 的全 140 开发集 mass_only / 单式分支 / FM MRR@25 为 0.18420817 / 0.25510398 / 0.29773598；FM Top25 为 101/140，较 mass 排序 23 升、115 平、2 降。 | 与 S1 参考条件不同，不能排列成同一提升曲线；39/140 真值虽在最终池却未进 FM 前 25。140 分支为 15 单式 MIST、12 多式 MIST、1 多式单候选、112 mass 回退。见 `PHASE5.md:23-37`、`outputs/phase5_20261023_13/metrics.json`。 |
| 可用性与公式门槛 | 冻结政策的 112 回退主因：87 个 `formula_ambiguous_explanation`、24 个 `no_supported_hplus`、1 个整组公式不支持或超预算。阶段二预定 9 例的强改善经过可用式/H+筛选；阶段三仅 6 例，低于预写 8 的效果支持门槛。阶段四 F 仅把旧 76 的 MRR 0.23653418 提至 0.23911651；FM 为 0.28760231，Top25 均 53/76。 | 公式 E_I 是解释覆盖，不是正确公式概率；阶段四所选式后验 7/7 含真值只属已曝光开发诊断，未见全 140 所选式真值率归档。没有用真值式作 MIST 输入，也没有真正无参考结构或全新结构生成成绩。见 `PHASE2.md:7-24`、`PHASE3.md:7-29`、`PHASE4.md:5-24`、`outputs/phase5_20261023_07/policy.json`。 |

负结果和修复分属不同层次：原基线先占前 8 参考谱名额再验证的错误已改为只计有效谱（`REVIEW.md:7-11`）；CSR 紧凑向量实现见 `src/baseline.py:116-122`，246 秒/4.76 GiB 是修后运行，**没有 CSR 前后量化收益**。阶段一 B 不增益；原阶段五动态补位使旧 9 个单式列表仅 4/9 相同，固定 61 是经授权另立的新开发口径而非修复上游 attention mask。提交迁移初版 Morgan 位打包顺序错误，修正后 832 候选字节、66 谱向量、140 榜单机械回归通过；Kaggle 大 `.tar` 版本曾 Failed 且被自动展开，不能断言解包就是失败原因；云 trace v1 因预建 producer 输出目录而 FileExistsError，修正的 trace v2 云端 4 查询与本地逐榜/CSV 字节相同。LOTUS、DreaMS 和作者新版 MIST 目前是来源/适用性审计，不是负性能实验。详见 `PHASE5.md` 与 `SUBMISSION.md`。

全量提交风险是**待验证的规模问题，不是已定位的失败根因**。`src/coconut_candidates.py:148-220` 每次校验约 199 MB 原 ZIP 并扫 738,794 CSV 行；对宽窗命中先 RDKit 规范化，再筛每 qid 最终质量池。`src/formula_audit.py:24-36` 对最终池候选再次做去立体/互变异构/规范键核验；单次查询集内按规范键去重，但没有跨运行持久缓存。`src/kaggle_submit.py:255-309` 对每完整公式×H+谱重复峰处理/官方子式枚举，选中式再做一次；上游 `spectra_utils.py:205-225` / `chem_utils.py:213-227` 先完整笛卡尔子式后过滤，并建立峰×子式质量差。现有 2e6/公式、200 配对和 W≤2e7 是逐 qid 限额；截至 2026-10-25 本轮审计开始时尚缺全 400 的实际成本分布；`finalize_test:418-430` 的指纹检查逐 key 线性搜索所有池，也待测是否显著。前 4/16 个 test molecule 是文件顺序前项，不代表昂贵尾部：本地 16 的候选构建 59.48 秒、宽窗 union 7,005、最终池 union 551；云端 4 的 prepare 70.4 秒且完整运行 161.48 秒。截至 2026-10-25 本轮审计开始时，正式 400 尚未本地端到端跑通，旧云端正式 v1 为 ERROR、无日志与 CSV，不能从运行时长推断卡在任何一段；本轮新证据见下节。`tests/` 没有直接覆盖 `src/kaggle_submit.py` 的单测；目前强迁移证据是本地 4/16 与云 4 的有界一致性，而非全量可靠性证明。

## P0+P1 授权执行证据（2026-10-25）

P1 已在既有冻结 140 开发候选/预测上完成后验损失分解，未重跑模型、未触碰 sealed70。结果与分母见 outputs/p0p1_20261025_01/loss_waterfall.json：FM Top25 101/140、miss39；按互斥首要主因，EI 模糊 29、无 H+ 8、所选式未含真键 1、真键所在选中式但式内名次超 25 为 1。多式门控 13 个中选中式含真键 11 个；q030 式内降序、q049 选式错误但最终仍 rank7。87 个 EI 模糊者真式在前二仅 52 个；不把此后验事实变成 oracle 预测或本轮新政策。

P0 先实测再决定优化：冻结的 test-only 输入分层 32⊂128⊂实际可见 400，未按真值筛选。普通环境 32/128 只用于成本辅助；与云资产同类 portable Python3.12/3.10 的全 400 已独立完成，候选准备 683.34 s、模型 18.00 s、整理 28.47 s，每个 job<900 s；模型 FP mismatch0、CSV实际400 ID同集且每行1–25合法结构，2 空窄池依原 wide-nearest。因现有成本足够，本轮不实施质量前筛、缓存或恢复重构，避免改变首 CSV 代表与回退语义。六项定向无效输入/分母/输出目录检查通过。云端分层32禁网CPU实际版本与本地 portable32 完整CSV字节一致（实际443.41 s）；本地 portable400 已通过；唯一新云端正式 full Notebook v1 已推服务端但仍待终态，尚无新比赛提交或分数，详见 SUBMISSION.md。旧正式 ERROR 根因仍未知，不能因本次 bootstrap 标记或云32成功而称已定位或修复。真实 ERROR/超时保留证据后停，不用开发 MRR 冒充比赛分数。

P0+P1 正式闭环：私人禁网 CPU 云端 full `okkojbjj/enveda-casmi26-fixed61-p0-full-private-cpu/1` COMPLETE，实际 2060.27 s、400 个可见 test ID、跨 RDKit 指纹差异 0；云/本地 portable400 的 `submission.csv` SHA256 同为 `991f5dab83aab8ca1fb42b581d24979b2d4e946ad6adc10c9cc741af0b98cf0e`，400 条完整候选列表逐项一致。唯一 competition submission ref `57622747` COMPLETE，公开分数 **0.134**、私榜空；2026-10-25T09:50:04 官方完整榜单 4005 支有分队伍，提交者/队伍/时间唯一对应 TeamId `17183818`，官方 rank **3538/4005**；榜首显示 0.504，按三位显示分数差 0.370。显示同为 0.134 的 17 队不代表精确并列。原始榜单和提交收据见 `outputs/p0p1_20261025_01/leaderboard_20261025_0949/`、`outputs/p0p1_20261025_01/submission_receipt.json`。这证实一次实际公开榜成绩，不能从已曝光140结构开发 MRR 推断隐藏集或私榜表现；旧正式 ERROR 的根因依然未知。本轮没有实施 P2、训练、扩库或解封70。

## P2 同池谱检索与直接融合复审（2026-10-25）

[outputs/p2_20261025_02/protocol.json](outputs/p2_20261025_02/protocol.json) 在效果计算前冻结 140 fresh_dev、两份 COCONUT 候选父快照、逐 qid 完整池与 FM 全顺序；固定的 32 个只做同参考运行验收。参考一次由全 140 查询合法质量联合预筛构建，非 NP 的 59,194 条有效谱/9,159 个结构，先按规范键/adduct 截取前 8 条**有效**谱，再对查询做存储内容精确签名排重（本次移除 0，不能据此称测量独立）。S2 在评分前移除所有 140 个目标规范键对应 3,997 条参考谱，余 55,197 谱/9,019 结构，交集断言 0。A 仍是原 0.02 Da 分箱、碎片/丢失 0.7/0.3、64 峰，按查询谱对同结构参考取 max 后跨谱等权平均；最终仅在冻结候选池与 15 ppm / 0.003 Da 窗内排名。RRF k=60 只给**正 A 证据**的结构加谱名次，无证据时完整保持 FM 顺序。目标键仅用于预定 S2 参考排除及离线分组；排名函数未接目标键/真值 SMILES，指标在 32 和 140 预测全部冻结后读取。

| 场景与全 140 分母 | A-only MRR / Top1 / Top25 | FM MRR / Top1 / Top25 | RRF MRR / Top1 / Top25 | RRF 对 FM |
| --- | --- | --- | --- | --- |
| S1：非 NP 参考 | 0.93146 / 124 / 139 | 0.29774 / 28 / 101 | 0.71322 / 80 / 139 | 109 胜、30 平、1 负；ΔMRR +0.41549 |
| S2：全部 140 真结构排除 | 0 / 0 / 0 | 0.29774 / 28 / 101 | 0.17069 / 8 / 95 | 0 胜、71 平、69 负；ΔMRR −0.12705 |

S1 质量/候选键及模式合格参考真结构覆盖 139/140；任意结构有正 A 证据 140/140、真键正证据 139/140；支持子集 A/FM/RRF MRR 为 0.93816/0.29988/0.71835。S1 唯一 RRF 退步 q084：FM 第 1、A 第 2、RRF 第 2。S2 支持 0、任意 A 正证据 118/140、真键正证据 0；A-only MRR 0 且 RRF 每 qid 真值名次不得改善的断言通过。旧 FM 全 140 MRR 与 Top1/5/25 计数、完整池与原代表、32/140 重叠预测均一致。由此直接等权融合**未通过预设 S2 不退门槛**，不继续搜索权重/阈值。S1 高分是同结构跨来源且已曝光开发集的谱参考条件，和 S2/隐藏集能力不同；预训练重叠仍未知。140 真结构并未用作 A/RRF 排名补库，70 sealed 未作查询或评价。

本次预测 [run_record.json](outputs/p2_20261025_02/run_record.json) 绑定原预测源码 SHA 941d4594…、参考/32/140 预测哈希；预测 51.85 s、最大 RSS 5,650,976 KiB，2 线程/12 GiB/900 s 限额均通过。独立评价首次因新增旧 FM 断言缺少 old_fm 赋值而 NameError，未写评价产物；原失败 [evaluation_failure_01.json](outputs/p2_20261025_02/evaluation_failure_01.json) 保留。仅修评价接线，不改或重跑预测；旧预测源码逐字节副本、评价源码 SHA 0e860929…及 [evaluation_source.diff](outputs/p2_20261025_02/evaluation_source.diff)（SHA 6e6076d5…）分开绑定，复评通过。原阶段性结论和正式 Kaggle 提交 ref 57622747 / 公开分数 0.134 未改；本轮没有云运行或新提交。下一科学建议仅是待授权的场景/置信路由，不能由 S1/S2 已曝光结果反向宣称泛化。

## P3 可靠性路由复审（2026-10-25）

同一 140 fresh_dev 的 S1/S2 冻结 A 正分全序、FM 完整池和目标键用于固定结构分折及离线校准；预测特征仅 A 第一分及第一第二分差。协议 [protocol.json](outputs/p3_20261025_01/protocol.json) SHA256 e3a1b94e…，五折选择 [fold_selections.json](outputs/p3_20261025_01/fold_selections.json) SHA256 2b7b851c…，双场景 OOF [oof_predictions.json](outputs/p3_20261025_01/oof_predictions.json) SHA256 d1e24b9d…，均先于独立 [评价](outputs/p3_20261025_01/oof_evaluation.json) SHA256 8f36155b…冻结。成功预测及评价源码同为 src/p3_router.py SHA256 bc621498…，每折 112 校准/28 留出、同结构两场景同折；旧 FM 四指标及 S2 名次不改善断言通过。原预检因 CONFIG tuple 与 JSON list 不同在效果前失败，失败记录、旧协议/源码和仅两处列表转换的 diff 在同一 run 保留；该失败没有 OOF 结果。

OOF S1：路由/FM MRR@25 0.29416455/0.29773598、Top1 27/28、Top25 均 101，0 胜 139 平 1 负，A 路由仅 1/140。OOF S2：0.29161353/0.29773598、Top1 27/28、Top25 均 101，同为 0 胜 139 平 1 负、路由 1。唯一变化 q099 属 fold4（阈值 .90/.20）：S1 由第 1 降至第 2、S2 由第 1 降至第 7；A 第一键 JENFZXAOWFNHSE，后验真键 ATBZZQPALSPNMF。校准各折 S2 路由数为 0，但留出 fold4 两场景均误激活。事前 S1 增益≥.01、S2 不退、S1 路由≥20 三项皆未达到；无最终全 140 阈值文件、未部署/提交或触碰 sealed70。原 S1 A-only MRR 0.93146 是同结构参考且已曝光开发场景，仍不能与 S2 或隐藏集泛化混同；本次只否定该固定网格与两特征规则。
## P3 后只读输入分布核对（2026-10-25）

只汇总已冻结的小型查询、成本和政策 JSON，不重新读取原始谱、训练大表或隐藏标签，也不生成新预测。开发集取 outputs/phase5_20261023_02/queries.json（SHA256 36552435…）及 outputs/p0p1_20261025_01/loss_waterfall.json（c30d90bf…）；可见 test 取 outputs/p0p1_20261025_01/prepare_all_portable/queries.json（18ea30ca…）、cost_strata.json（6008617c…）与 model_all_portable/policy.json（9f859443…）。两者来源和选择机制不同，下表只是输入/现有政策分布，不是隐藏集效果比较。

| 已有记录 | fresh_dev 140 | 可见 test 400 |
| --- | ---: | ---: |
| 任意正谱 / 任意负谱 / 正负混合的结构数 | 128 / 78 / 66 | 373 / 124 / 97 |
| 正 [M+H]+ 谱数；没有 H+ 的结构数 | 321；24 | 959；34 |
| 无 H+ 中另有其它正 adduct / 纯负 | 12 / 12 | 7 / 27 |
| 查询中性质量 Da，最小/中位/最大 | 133.04 / 348.17 / 1108.60 | 246.10 / 328.19 / 439.10 |
| 最终窄窗候选数，最小/中位/最大 | 2 / 43 / 385 | 0 / 36 / 570 |
| 完整候选池分子式种数，中位 | 4 | 4 |

现有互斥主回退原因在开发 140 为 87 个 EI 歧义、24 个无 H+、1 个整组支持/预算不足，共 112；可见 test 的 289 个 mass 回退为 EI 歧义 247、无 H+ 34、空最终窄窗池 2、单候选 3、低解释强度 2、整组支持/预算不足 1。test 的另外 80 个多式模型、15 个单式模型及 16 个多式单候选分支不计入 289。空窄池 2 按原 wide-nearest 兜底，不代表窄窗覆盖或模型成功。开发无 H+ 的 12 个其它正 adduct 也尚未证明为 MIST-CF checkpoint 所支持；纯负的 12 个更不能由正离子 H+ 方法直接覆盖。可见 test 质量范围较窄属于这 400 个输入的观察，不可推断隐藏输入、真实标签或新比赛分数。
## MIST-CF 公式判别审计（2026-10-25）

作者 [mist-cf](https://github.com/samgoldman97/mist-cf) 固定 commit 71c820b4…、公开 NPLIB1/CANOPUS 单检查点 [Zenodo 8151490](https://zenodo.org/records/8151490)，权重 MD5 bff0db38c4ae1e68bd93bf3e3ec46094、SHA256 b2345d06…；权重独立再分发条款未核实。仅 common/__init__.py 的 Ray 调参/绘图两处缺依赖可选导入作严格 guard，原件和 [diff](outputs/mistcf_20261025_01/optional_import_guard.diff) 保留；严格 CPU 权重加载完整，max_subpeak=20、cls_mass_diff=False。所有 250 条谱原 timsTOF 均按作者合法字面 Unknown (LCMS) 映射为 15ppm，不猜仪器类型。原生谱处理取 top100、模型最多 20 个已分配子式；零匹配公式仍为合法 CLS-only 输入。固定 q007/q085 完整 22 配对在单独、batch2/倒序及正式 batch8 下全公式排序一致、分数在事前 atol=1e-6/rtol=1e-5 联合容差内。预测先于真式评价冻结，[预测](outputs/mistcf_20261025_01/predictions.json) SHA256 66d47538…，[评价](outputs/mistcf_20261025_01/formula_evaluation.json) SHA256 8c13e076…，预测/评价源码分别 8237fb05…/aaebde77…。

完整 87 分母的**公式 full-rank MRR** 为 MIST-CF 0.92598978、质量误差 0.88122605、冻结 EI 0.57961138；Top1 77/67/30，Top2 81/85/52。相对质量误差赢 17、平 61、输 9；质量误差 Top2 命中却被模型降至 >2 有 5 例（q005、q020、q021、q134、q135），模型从 >2 挽回 1 例（q114），净少 4。87 中仅一条 H+ 谱有 40 例，5 个损失中 4 例属此类；只是后验背景，未证明原因。模型技术有效 87/87、公式 MRR 两项增益通过，但 Top2 不低于任一对照的事前门槛失败，因此不接结构候选排序；本轮没有新的比赛提交或分数。训练结构重叠未核实，开发集已曝光；公式 MRR 不等于比赛结构 MRR@25，亦无隐藏集性能证据。[运行记录](outputs/mistcf_20261025_01/run_record.json) 记载 2 线程、12GiB、每作业≤900s、新下载 2,910,777B；已记录计算作业约 19.3s，最高步骤 RSS 1,097,300KiB。上游 PredDataset 的 tuple 布尔赋值对本次 cls_mass_diff=False 权重不生效，未改模型计算。
## 公开强基线源码与资产只读审计（2026-10-27）

本节只核公开页面、公开 GitHub 小型源码/说明及本项目现有资产清单；没有拉取 Kaggle 大资产、运行模型、上传或提交。Kaggle 历史版本 GetKernel 曾返回 403，本轮没有重试或绕过。[prvsiyan V43 页面](https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline) 显示公榜 0.456、T4×2 约 27m25s，但正常网页在本环境未提供完整源码和全部输入。公开的 [第三方 V43 审计](https://github.com/cubres/enveda-casmi-2026/blob/main/research/v43-pubchem-growth-source-audit/2026-10-20/README.md) 只记 scriptVersionId 360497306、35 cells 的声称 SHA256 f254dfbc…；它没有发布原 Notebook 字节，所附 Dataset 版本未锁。因此不能称 V43 已可复现。

目前可定向审计的强基线候选是 [leesirleesirlee-star 的 V44 Notebook](https://github.com/leesirleesirlee-star/Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra/blob/aac4dd8e2148d49a7de703387d2f26e0cf50eeb4/notebooks/v44_base/notebook.ipynb)：仓库 commit aac4dd8e2148d49a7de703387d2f26e0cf50eeb4，Notebook Git blob 471adf74ee35cc9dc3c98ed571ce402529adc888，148,461 B，33 cells/21 code cells。其 [README 的归属说明](https://github.com/leesirleesirlee-star/Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra/blob/aac4dd8e2148d49a7de703387d2f26e0cf50eeb4/README.md) 把原作者记为 imranarif536/casmi26-v44-pairtail-locked-top1，xiaoyuzhoux120 是该团队 fork；仓库自述 V44 提交 ref 56839982 与自行 V45 ctl ref 56854098 均为 0.417、代码 0 cells 差。这是其自述与可读代码，不是本项目独立重评分；也不是 prvsiyan V43/0.456 的原版源码。

执行代码与说明文存在需保留的区别：cell 5 当前 IS_RERUN=True，仅设为 False 才将可见 test 截为 SMOKE_N=12；谱 ID 的 MD5 被打印，但当前不用于选择分支。cell 3 ICE_BUDGET=300 秒，cell 17 GL_BUDGET=4000 秒；markdown 提到 5400/6000，不能代替执行值。cell 17 显式使用 CUDA，而仓库存的 Notebook metadata 为 Python 3.12.13、dockerImageVersionId=28755、dataSources=[]、GPU 标志 false；这份 metadata 不足以证明评分版真实挂载、Docker 与 GPU 配置。PubChem、ICEBERG、GLACIER 阶段均有捕获异常后继续基础榜单的降级路径。[仓库复盘](https://github.com/leesirleesirlee-star/Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra/blob/aac4dd8e2148d49a7de703387d2f26e0cf50eeb4/README.md) 报告历史 ICE 300 秒只评分 71/400。若日后复现，必须核实际完整可见输入、各通道完成数和评分版本配置；12 分支仅作运行性验收。

该仓库 [外部资源登记](https://github.com/leesirleesirlee-star/Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra/blob/aac4dd8e2148d49a7de703387d2f26e0cf50eeb4/docs/EXTERNAL_RESOURCES.md) 自述下列主线资产。体积和许可为该仓库 2026-10-05 的登记，本轮没有逐项用 Kaggle 核许可或精确 Dataset 版本；Notebook 用 /kaggle/input 文件模式寻找它们，版本均未锁定。MB 为仓库登记值。

| Dataset slug | 登记大小 / 许可 | 作用或缺口 |
| --- | --- | --- |
| ahmedberatozer/casmi26-v4b-models | 1,682.6 MB / Other | 主引擎、64 文件 MANIFEST |
| ahmedberatozer/casmi26-v3-models | 411.9 MB / Other | PubChem FPNet |
| ahmedberatozer/casmi26-fpnet-full1 | 198.8 MB / Other | 主引擎可选 FPNet |
| ahmedberatozer/casmi26-iceberg | 66.9 MB / Other | ICEBERG |
| ahmedberatozer/casmi26-glacier | 61.1 MB / Other；上游 MIT | GLACIER |
| prvsiyan/casmi26-fp-models-v2 | 288.2 MB / CC0 | 第二引擎 FPNet |
| metric/rdkit-2026-3-3-wheel | 148.9 MB / Unknown | 离线 wheel；软件 BSD-3 不等于 Dataset 许可 |
| ahmedberatozer/casmi26-v2-pool | 1,684.6 MB / Other | 约 71–77.5 万结构候选池 |
| ahmedberatozer/casmi26-pubchem-tier | 7,214.1 MB / Other | PubChem 候选层 |
| prvsiyan/coconut-casmi26-candidates | 423.3 MB / CC BY 4.0 | 指纹化 COCONUT |
| prvsiyan/chebi-lipidmaps-casmi26 | 60.4 MB / CC BY-NC-SA 4.0 | ChEBI / LIPID MAPS |
| megayak/casmi26-simulated-ranker-rows | 366.0 MB / CC0 | 模拟 ranker 行 |
| prvsiyan/casmi26-ranker-features | 9.6 MB / CC0 | ranker 特征 |
| dmitriigluzdov/casmi26-pubchem-popularity-prior | 2,754.0 MB / 许可未取得 | 可降级的流行度先验 |

本项目 [已冻结私人资产清单](outputs/kaggle_submission_20261023_01/private_asset/asset_manifest.json) 只有原始 COCONUT CSV-lite ZIP 199,096,787 B、旧 MIST fp checkpoint 60,750,958 B、原推理源码/便携解释器/wheels；与上表没有按 slug 和版本证实相同的预计算模型或候选 Dataset。原始 COCONUT 与 RDKit 软件类别虽相近，不能直接代替 V44 的指纹化池、ranker、FPNet、PubChem tier 和 ICE/GL 权重。仓库根 [MIT LICENSE](https://github.com/leesirleesirlee-star/Enveda_CASMI-2026_Molecule-ID-From-Mass-Spectra/blob/main/LICENSE) 不覆盖外部资产，也不自动解决所转载原作者代码的归属与许可。进入实质复现前，须锁原作者到该 fork 的源码对应关系、实际挂载 Dataset 版本和许可、Docker/GPU 配置，以及这些多 GB 资产的下载/运行预算。本轮未获大资产下载或 GPU 运行授权，未验证其 0.417。


### 同名 V465 衍生 Notebook 的定向核验（2026-10-27）

用户放入 outputs/analog-propagation-casmi-2026-baseline.ipynb 的文件为 315,431 B，SHA256 3fde6f5a7b9de6f7934292cd4acf93274897086ed944d8d68878555401ddd366，35 cells（32 code）。其开头明示基于 Hongbin Gu 的 V465 v15（source scriptVersionId 359783615、原作者公开显示 0.456）；该数字**不是此文件自己的比赛成绩**。第三方将 prvsiyan V43 记为 35-cell source snapshot SHA f254dfbc…，但未公开该哈希的序列化算法或字节范围；它不能与本文件的整 JSON 字节 SHA 直接比较。V465 来源署名及附加实验层也可能被 V43 继承，故当前仅能说本地字节尚未与用户提供的 prvsiyan V43 页面版本绑定，不能据此判定下载错版、本地旧版或必非 V43。源作者页面已在文件中：[V465 v15](https://www.kaggle.com/code/guhongbin/casmi26-v465-v15-pop015?scriptVersionId=359783615)。

Notebook metadata 有 16 项 source：competition 143318 加 15 个 datasetVersion ID（19733458、19722987、21006744、19888840、19822292、20127474、20090231、20016725、19916385、19845497、19915656、20021505、20441000、19692718、19518628），dockerImageVersionId 31430，T4 accelerator。cell 1 另列 15 个 Dataset slug 的 2026-10-20 “当时最新版本/许可”快照，但没有在文件中把数值 sourceId 与 slug 显式配对，也明确不能代表原 V465 评分运行的历史挂载版本；权重/候选 Dataset 的许可不能由 Notebook 的 Apache-2.0 来源说明覆盖。该快照与另一 V44 仓库较早的许可登记对部分 Dataset 存在差异，本轮未把任何一份当成已核的历史许可结论。

cell 10 的实际条件为 KAGGLE_IS_COMPETITION_RERUN 环境标志为真，或当前 spectrum_id 内容签名不同于可见公开签名时全量处理；否则仅处理排序前 12 个 molecule_id，其余仍保留 400 行格式。文件保存的 Papermill 运行记录为 1629.97 秒，cell 10 输出 RERUN False/公开签名匹配，证明这份文件执行过可见 12 活跃分子的运行性检查，**不是隐藏评分**。cell 29 的 metric-identity 重排先备份再覆写 submission.csv（该次 changed_molecules=0）；cell 33 的 E006 再从既有预测/控制榜重排并覆写（该次 changed_rows=1、Top1 未变）；cell 34 的 rank2 从第二引擎名单把首候选移到第 2 位，必要时原子替换最终 submission.csv（该次 eligible=12、changed_rows=0；隐藏运行失败尝试恢复 E006 字节）。因此这些实验层可改变最终提交列表，不能把整个文件称为未改动的 V465 原件。所核尾部输入是测试谱、已有候选/模型输出、sample_submission 和控制 CSV，未见尾部直接读取隐藏真值；这不是对全 Notebook 数据泄漏的证明。

此文件 metadata 没有可将本地字节绑定到 prvsiyan V43 的 Kaggle kernel ID 或 scriptVersionId；内文的版本号指向 V465 来源，也没有这份字节对应的官方 submission ref/publicScore。它目前是“源码及一次公开输入运行记录可审、V43 字节与评分身份均未绑定”。用户已提供 prvsiyan V43 精确页面链接；要闭合来源，仍需该版本官方导出的原始源码字节或可复算的同算法哈希及下载记录，不能只凭同名、署名或不同定义的哈希判定。此轮未运行 Notebook、下载依赖或新增提交。

为后续私人 Copy & Edit 草稿核对，本地文件另冻结 **cell-source SHA256 af820406e83e1fb7a73609d58620678c80a220536d145ceedf2a7264577ed3ec**（序列化后 247,359 UTF-8 字节）。可重算算法：用标准 JSON 解析 ipynb；按原 cells 顺序取二维数组，每项仅为 [cell_type, source_text]，source 若为行数组则原样无分隔符拼接，否则保留字符串；用 Python json.dumps(cells, ensure_ascii=False, separators=(",", ":")) 序列化，再将 UTF-8 字节做 SHA-256。不修剪空白、不统一换行；排除每个 cell 的 outputs、execution_count、metadata 及 Notebook metadata。35 个 cell 均计入（32 code、3 markdown）。该值仅用于与之后草稿按**同一算法**比较 cell 类型与正文是否保持一致，不能证明本地文件即官方 prvsiyan V43。


许可快照的具体阻断分布：15 项中 8 项标 Other（含主 v4b/v3 模型、v2 pool、PubChem tier、ICE/GL 与流行度先验）、3 项标 CC BY-NC-SA 4.0（prvsiyan ranker-features V6、fp-models-v2 V1、coconut-casmi26-candidates V2）、1 项 RDKit wheel V1 标 Unknown；其余 2 项 CC0、1 项 CC BY 4.0。这些只是候选 Notebook 写入的 2026-10-20 最新版快照，尚不能作为 15 个历史 sourceId 的逐项许可证明，NC/Other/Unknown 在下载或复现前均需版本级核定。


### 同名 Notebook 的实际输入与复现门槛（2026-10-27）

以下只对 SHA256 3fde6f5a… 的本地文件做静态审计，**不是运行或成绩复现**。cell 7 的 find(pattern) 在 /kaggle/input/** 递归取按路径长度排序的首个命中；这没有按 Dataset slug 固定版本，同名文件多挂载时也可能选错来源。metadata 是 1 个 competition sourceId 143318 加 15 个数字 datasetVersion sourceId；代码/内文没有数字 ID→slug 对照。官方公开列表只返回 Dataset 当前版本、大小、许可字段，历史版本列表为空；下表 slug 按文件名及候选内文推定，**均未与 15 个历史 sourceId 逐项绑定**。

| cell / 实际读取与用途 | 必要性 | 候选 slug / 历史 ID | 组件 | 公开许可证据 | 当前公开体积 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 7、17、19：v4b 的 casmi/engine.py、MANIFEST.json，加载 FPNet/特征族/LightGBM；16 从比赛 train.parquet 建谱缓存 | 主引擎必需，manifest 全文件 SHA 校验 | [v4b-models](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models)；历史 ID 未配对 | 已训练模型、代码、ranker | 当前卡片 Other，历史细则/授权未核 | 当前 V4 1,234,525,593 B | 待补证；Notebook Apache-2.0 不覆盖权重 |
| 7、14：v3 的 ranker_0.pkl、fpnet_*.pt | 启动定位必需；PubChem 通道异常可降为 base lists | [v3-models](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v3-models)；历史 ID 未配对 | 旧 FPNet/排序器 | 当前 Other，历史细则未核 | 当前 V4 374,959,744 B | 待补证；还须证明通道未静默失效 |
| 7、16、17：pool_meta.parquet、pool_fp.npy 等候选池 | 主引擎必需 | 候选 [v2-pool](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v2-pool)；仅文件名推断，未核真实 slug/ID | 结构池/预计算指纹 | 候选卡片当前 Other | 候选卡片当前 V1 704,631,733 B | 待补证；先核真实挂载路径 |
| 7、14、19：pc_lsid.npy/pc_lpmid.npy 流行度先验 | pc_lsid.npy 启动必需 | 候选 [popularity-prior](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior)；历史 ID 未配对 | PubChem 来源先验 | 当前 Other；内文 10 月20日 V9，现列表 V10 | 当前 V10 2,653,105,770 B | 待补证；版本已有可见漂移 |
| 7、14：pc_smiles.npy 和质量数组 | pc_smiles.npy 启动必需，通道评分可异常回退 | 候选 [pubchem-tier](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-pubchem-tier)；历史 ID 未配对 | PubChem 候选库 | 当前 Other，原数据条款/历史版未核 | 当前 V1 1,203,988,750 B | 待补证 |
| 7、21：ice_runner.py、模拟器 RDKit 2025.3.6 wheel；21 的 gl_runner.py | ICE 包及兼容 wheel 在 cell 7 先检查；ICE/GL 评分失败可回退 | [ICEBERG](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-iceberg)、[GLACIER](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-glacier)；历史 ID 未配对 | 光谱模拟器 | 两卡片当前均 Other，历史授权未核 | 当前 V1 各 64,195,893 / 56,264,835 B | 待补证；需记录真实通道健康/覆盖 |
| 7：rdkit-2026.3.3 wheel | 启动必需，安装后精确版本断言 | 候选 [RDKit wheel](https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel)；历史 ID 未配对 | 软件 wheel | 内文当前卡片 Unknown；需分开核软件和 Dataset 许可 | 未核字节 | 待补证 |
| 附载的 prvsiyan ranker-features、fp-models-v2、coconut-casmi26-candidates | 静态主路径未证明直接读取，不能把挂载等同必需 | 三个历史 ID 均未配对 | 旧特征/模型/候选 | 内文快照与当前 ranker 卡片为 CC BY-NC-SA 4.0；若证实必需，另核适用授权 | 不计入必需体积 | 使用状态待核，不能直接称本方案使用了这些 NC 资产 |

前五项候选卡片**当前**体积相加为 6,171,211,590 B；这是公开卡片大小线索，不是历史下载量或最小运行包。比赛数据另受比赛条款约束，不能与外部 Dataset 许可混同。Other/Unknown 本身不等于违法，但不足以确认原样参赛所需的使用与再分发权；若实际必需外部组件仅有不兼容的非商业或禁止再分发许可且无额外授权，须停止原样竞赛复现。主办方[公开讨论](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741359)的获奖方案外部组件要求应在版本级确认。

cell 10 用环境标志或测试谱签名决定公开路径与全量路径；保存的公开运行仅 12 个活跃分子、1629.97 秒，不能据此保证 hidden 全量在 9 小时内完成。metadata 同时记录 T4 accelerator 与 isGpuEnabled=false，代码实际选择可用 CUDA，否则 CPU，不能仅凭字段断言 CPU 等价。主执行中 build_spec_cache(train.parquet) 是建参考谱缓存；静态外层未见模型 fit/反向传播，实际加载 FPNet 与 LightGBM 既有权重。ICE、GL、PubChem 和部分融合异常可被捕获后继续输出，格式成功不等于全部科学通道生效。最终 submission.csv 还经过上节的 metric-identity、E006、rank2 覆写；隐藏分支与这些模型/许可/运行时均未在本项目复现。

### 公开 V1 fork 运行与提交前许可闸门（2026-10-27）

用户账号的 [Notebook V1](https://www.kaggle.com/code/okkojbjj/analog-propagation-casmi-2026-baseline) 在 12:44 UTC 显示 COMPLETE，未提交比赛。仅下载小产物至 [outputs/v43_fork_source_audit_20261027](outputs/v43_fork_source_audit_20261027)：`submission.csv` SHA256 `dd7df508e109805f21d06d009f8e6e0d89f53bd59c7545edc80adf6ee6ecdace`、`run_manifest.json` SHA256 `82ca738652c0f816002a2e324490803ee9ba26bd31d898e2b32eaa5c4dde0a4e`、日志 SHA256 `dac456649537088873482ce453ce313c63a9b3c1c26faafb2b44e7e74fa2f8bf`。CSV 与官方 `sample_submission.csv` 的 400 个 ID 及顺序一致，每行 1–25 个候选、共 688 个；用本项目 RDKit 2026 `canonical_key` 检查，候选全有效且行内无规范键重复。公开运行仅 12 个活跃查询，其余保留 400 行输出格式；不能据此声称隐藏全量或公榜 0.456 复现。日志的 ICE/GL 子步骤各报告 11 个查询 `status=ok`，E006 改 1 行、最终 rank2 覆写改 0 行；单独 manifest 中较早的 `ICE NOT RUN`/`GL NOT RUN` 不应覆盖后续日志事实。

先前表格把 `prvsiyan/casmi26-fp-models-v2` 列为“未证实使用”；本次云日志明确 `[377s] fp models: 1 single + 1 merged on cuda`，Notebook cell 12 将该目录下 `fp_single_s2.pt` 与 `fp_merged_m1.pt` 传给主引擎 `E.main`。此项**实际使用**，覆盖先前静态判断。公开 [Dataset V1](https://www.kaggle.com/datasets/prvsiyan/casmi26-fp-models-v2/versions/1) 的官方小元数据（本地 `dataset-metadata.json` SHA256 `27869624c10a62f13ff626b0bd1e407fcdaa381999e0c904d587c059855240af`）标记 `CC-BY-NC-SA-4.0`，说明保留这一许可；两权重各 144,110,255 B，卡片给出各自 SHA。元数据中未见允许比赛要求的商业使用与再分发的额外授权；尚未读取云挂载的数字 datasetVersion ID 或文件内额外许可。根据[主办方许可说明](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741359)，原样正式提交因此**暂停**，不能以运行成功、Notebook 自身 Apache-2.0 或“只求分数不领奖”代替这项版本级授权。后续若取得适用额外授权再重新审提交闸门；本轮没有提交、成绩或源码字节与 prvsiyan V43 的新绑定。


### P5 固定质量前两式结构排序：开发集结果与适用边界（2026-10-27）

唯一[协议](outputs/p5_mass2_20261027_01/protocol.json)绑定原 140 fresh_dev、COCONUT final pools、完整 H+ 原峰、旧 fixed61 checkpoint/完整 FM 键序；使用旧 MIST 20ppm 子式、top60+CLS、fixed61 和 Morgan4096 big-pack，不混入 MIST-CF。EI 在此指 fragment explained intensity（片段峰强度解释率），不是电子电离；旧 formula_ambiguous_explanation 由完整候选式与 H+ 谱的平均 E_I 首二间隔小于 0.05、且首位 E_I 不低于 0.50 产生，无真式/真键选组。87 条两式都按冻结全部合法 adduct 中性质量中位数的绝对误差及 formula 字符串选出；逐式 Cartesian≤2,000,000、原完整式 pair≤200/W≤20,000,000，未加新回退阈值。q113/q085 无标签批次测试通过，500/500 所需式×谱及 5850/5850 跨版指纹有效，0 技术回退。首个模型环境调用因未把既有官方源码加入 PYTHONPATH 报 ModuleNotFoundError，未产出指纹；补入原路径后同源码成功，原失败保留在本 run。

[预测](outputs/p5_mass2_20261027_01/predictions.json) SHA256 9f1c10f7… 先冻结，[评价](outputs/p5_mass2_20261027_01/evaluation.json) SHA256 30c430d2… 后独立读取目标键；[另一实现的冻结复算](outputs/p5_mass2_20261027_01/independent_verification.json)重新核全 87 两式/500 谱标签、结构 FP 与模型向量余弦、交替和余序、53 条旧榜单、140 条官方规范键首 Top25 名次，全部一致。全 140 旧 FM→FM_mass2：MRR@25 0.29773598→0.54573536，Top1 28→59、Top5 58→97、Top25 101→124；66 升、64 平、10 降。F_mass2 为 0.28754340、Top25 97，低于旧 FM；FM_mass2 相对 F_mass2 MRR 多 0.25819196。87 个目标中新增 25 次 Top25 命中，丢失 q052/q120 两次；q069 1→17、q099 1→3 表明固定交替亦可伤害强预测。其余 53 条旧 FM 逐列表未变。后验两式真键覆盖 85/87，不是公式正确概率或独立验证。已曝光开发结构、预训练重叠未知，不能外推到 hidden/public；唯一真实比赛分数仍为原 fixed61 提交的 0.134。旧权重可用于本地研究，详细再分发权利仍未闭合；当前不上传新权重、不提交、不触 70 sealed。
