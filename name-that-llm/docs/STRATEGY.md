# 提交与优化策略（2026-10-21修订）

## 今天的教训

公开成绩：0.308 → 0.361 → 0.398 → 0.421；最后去中心候选57482280已COMPLETE，公开0.447。前三次改进方向有效，但提交节奏不合理：基础特征、浅层表示、双模型本可在本地统一比较后只提交较强候选。每日5次是上限，不是目标。

已发生的提交不能收回；今天不再启动实验/提交，只监控最后评分。用户新要求优先于此前“把剩余4次用完”的指示。

## 下一轮先做什么

1. 等最后正式分数，确定当前最佳公开版本。保留v5=0.421作已验证对照，不因v6本地0.53267就宣布正式最佳。
2. 固定原始150个prompt的折归属、输入、随机种子和指标。v4跨软件版本划分差异已记录，不混作同折改进。
3. 对去中心分支做消融：相同冻结特征下，只改变普通/去中心权重；分组方法也单独对照，不同时加入外部数据、微调和多个解码改动。
4. 重点补不完整prompt组的验证：每组只见1、2、3、6条时，比较前后版本的Balanced Accuracy、各类召回、单条回退与误分组影响。已有全组OOF不等同于隐藏测试结构。
5. 保留失败实验和退步类别。现有验证数据已被反复使用，存在选择偏差，新增划分也不能称为从未见过的盲测。强候选冻结后，用预先约定的第二组prompt划分复核方向。
6. 只有现有表示/去中心局限清楚后，再考虑外部公开样本或LoRA。先审核来源、许可、标签可靠性和与官方prompt/文本的重合；GPU仍受单次2h、总先限8h授权约束，不因有配额就使用。

## 哪些候选值得占用一次提交

提交前都应有：
- 相同数据和固定prompt折下的前后对照；
- 改动、假设、实际有效配置和代码标识；
- 不完整组/单条场景及薄弱类别的检查；
- 输出ID/标签/有限值/parquet回读和私有运行验收；
- 说明这次正式提交要验证哪一个尚未解决的问题。

操作门槛：优先考虑OOF提高至少1–2个百分点、至少2/3折改善，并在预设复核划分中方向一致的候选。此为节约提交的经验门槛，不是统计显著性或隐藏成绩保证。更小改进先留在本地；只有明确解决失效模式或重大分布适配问题时才说明例外依据。

不能为了“版本号增加”或“剩余次数”提交。运行成功只需首次打通流程和重要接口变更后验收，不能把反复确认格式当作消耗排行榜额度的理由。

## 每日额度使用

- 默认0–2次正式提交；没有合格候选可以0次。
- 第1次：当天本地批量筛选后的最强候选，验证一个主要假设。
- 第2次：根据第1次提供的新证据，提交有独立依据的修正或明显更强方案。
- 至少保留2次给真实格式/环境故障、重要新方案或临近截止的需求；备用不是必须用完。
- 不因已到晚上而补交，不自动跨日消耗新额度。
- 同时最多1个待评分候选；它未出分时继续做本地分析，不立即上传另一套大方案并提交。
- 若公开榜退步，先定位验证分布、分组和校准；不围绕公开分数反复试权重。

## 每轮给用户的报告

1. 上一版暴露的具体问题；
2. 本轮只改了什么，哪些尝试无效；
3. 相同验证与重要场景的前后数据；
4. 是否达到提交门槛，以及提交能补充什么证据；
5. 若提交，给出真实公开分数、提交ID、剩余额度和当前最佳版本。

今天最终分数出来后，先报告再确定后续实验优先级。未经新的任务推进，不自动启动下一轮GPU或提交。

## 最后评分完成（2026-10-21 北京时间21:05核查）

提交57482280（v6内容去中心），官方状态COMPLETE，公开榜0.447。
相对v5=0.421提高0.026（2.6个百分点）；相对起点0.308提高0.139（13.9个百分点）。当前已验证公开最佳为v6。
当日总提交5次，剩余0。本轮额外4次均有正式结果：0.361、0.398、0.421、0.447。
本地OOF0.53267与公开0.447相差约0.08567；不能将本地9.89个百分点提升外推为公开提升，实际仅2.6个百分点。下一轮应优先校验不完整prompt组、误分组和反复使用开发验证的选择偏差，不继续叠加模型追分。
本次查询GPU quota累计used=0.52h；官方配额可能有更新延迟，该观察不是逐作业计费明细。
只完成记录与通知，停止评分监控，不启动新实验、上传或提交；后续遵循STRATEGY.md，每日默认0–2次，不以用完额度为目标。

## 2026-10-23 新系列预注册（用户授权长任务，总GPU累计<=8h）

A：outputs/style_local_mask_20261023_31，仅四个style视图（等长前/中/后各floor(n/3)字符、全正文固定功能词/标点mask），完整prompt隔离OOF后精确切取两套既有447行mixed样本。原style概率直接复用；固定原TFIDF/LinearSVC参数及row-softmax，不调参、不融合。比较每折BA、对full-style的paired-prompt bootstrap2000/seed20261023、位置对照、相对v8的救/害行与prompt数；mask仅报覆盖代理。

B（以下原预注册，现已完成；实际仅拆顺序调度见末段）：ModernBERT原checkpoint重新加载，matched grid CE vs CE+.1SupCon(temp=.1)，每micro=4作者x2不同train prompts的真实完整网格8行；相同预生成batch计划、初始化/RNG、2micro累积、更新窗口与3epoch，不复制回答，不用validation prompt，CLS末层L2归一无projection/detach。固定512/eager/no compile/FP16 autocast/AdamW2e-5/wd.01/warmup.1；单Kaggle job顺序6fits<=3600秒，保守预算不足停止，不删除CE对照，micro8 OOM停止不换网格。记录曝光/unique覆盖/loss/AMP skips/实际更新数；两arm实际步数因AMP跳过可能不同。两套exact mixed比较机制(SupCon vs matched CE)与实用(each vs v8)门槛各自两套delta>=.01且各>=2折改善，类别下降>.10不得直接晋级；区间仅conditional reused OOF。v14普通随机batch是历史，不是单因果control。

C仅A/B出现一致且实用机制后预注册一次固定整合；否则停止。不得自动选final/排行榜提交/安装依赖/服务器GPU/换server。

B execution scheduling update: v15 stopped at the unchanged budget gate before any fold (SupCon probe passed). Run the same fixed six fits as two sequential private jobs, CE3fold then SupCon3fold, each independently probed and gated at3600sec. No grid/loss/hyperparameter/threshold change; no interim CE result may alter SupCon. Pin the v15 official docker image, compare model/config/version/plan hashes, and judge only when both arms finish. Preserve original v15 technical failure.

本系列最终状态：A四视图均低于全文style；B两套SupCon-CE分别-.004520/-.018643，1/0折改善，CE/SupCon均明显低于exact v8。机制与实用gate全fail，因此C不启动，不融合/晋级/排行榜提交。v15预算技术停止已保留，CE v16与SupCon v17均3fold完成且matched完整性PASS。官方GPU累计观察2.11h，本系列约+.36h；系列到此停止，不因8h还有余额扩展。完整证据outputs/style_local_mask_20261023_31与outputs/modernbert_grid_contrast_20261023_32各REPORT.md/summary；局部删词也删作者措辞，不能将A退步解释为题材作弊或纯语法无效，B训练loss优化也不等于泛化改善。

## 新轮inner-prompt / dropout诊断（2026-10-23，完成）

先完成训练审计outputs/training_audit_20261023_33：历史三组OOF均输出18类且NLL优于uniform，4dropout为0，缺同checkpoint train/逐epoch inner。固定补充诊断_34仅普通随机CE：control0 vs embedding/mlp/classifier=.1、attention0；原ModernBERT/CLS/LR2e-5wd.01/512/FP16eager/batch8acc2不变，固定4epoch。每outer训练100prompt内按SHA256('inner-20261023:'+prompt_id)前20%为inner，其余fit；50outer完全不评价。两臂完整shuffle共享，每epoch同checkpoint eval()完整fit/inner后恢复train，primary4不选best。

v18 probe成功但原三折预算bound4090.55>3300在任何fold前技术停止；保留原失败。独立审查后仅技术调度变为六顺序单fold（controlv19/20/21→regimev22/23/24），科学设置/gate未变。用户明确授权两文件到现有私有ref/id139969019后正规审批执行；每job先成功查quota容纳下一3600s，临时.1 probe AMP128成功后freshreset主训练AMP65536，每job独立budget gate，全fit+inner评价纳入。实际24Dropout与22attention0均验收。

六job全部COMPLETE+CPU小产物验收PASS，严格合并每臂fold0/1/2各一次并核sourcecore/data/model/env/recipe/head/分组与shuffle，比较exit0。epoch4 innerBA control .252778/.241667/.219444（均值.237963），regime .127778/.119444/.155556（均值.134259）；均值Δ−.103704、0折改善，固定诊断门槛（>=+.01且>=2折改善）失败，不晋级、不outer评价、不榜交。

control四轮fitBA .1675/.3393/.5187/.6551，innerBA .12685/.20556/.24444/.23796，末期fit继续改善而inner退步；.1组四轮fitBA .0749/.1020/.1522/.1677、innerBA .06852/.08333/.12593/.13426，拟合也明显更低。不能用.1组较小gap称更好泛化，不能把epoch3替代固定epoch4，不能单归因机制或保证其他dropout/训练长度效果。inner集合重叠且开发资料复用，非3独立盲测。

最终官方GPU2.688339/8h，六job观察增量.556401h/native合计1710.856秒，v24COMPLETE且无活动job。每折planned360；actual control354/356/353、regime353/354/354，AMPskip分记。41CPUtests通过后source e650177e…保持，v8 profile/metadata byte恢复。详细曲线CE/BA/accuracy、成本、失败和身份见outputs/modernbert_inner_dropout_20261023_34/REPORT.md、summary.json、comparison与final_verification.json。当前不扩展新训练，先独立审阅结果。
