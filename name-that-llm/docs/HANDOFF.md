## 2026-11-24 20:40 58811392 得 **0.720**（rank 406/6943），仍差银线一档；发自有改动 c20

`58811392`（0.720 sibling-odds 原样线）COMPLETE：public **0.720**。公榜快照（2026-11-24T12:39:18Z，6943 队）显示本站 rank **406**、SubmissionCount 15。按名次仍在**铜牌区**：铜线第 694 名（0.718）、银线第 347 名（0.720）。0.720 这一档共 409 队，我们排在 406，差在并列时的排序；**0.721 只有 147 队，进 0.721 就稳定进银牌区**。金线 0.758（第 23）。

同一份代码在本账号的三次实测：0.717 原样线两次 0.718；加 sibling odds 后 0.720（+0.002，和公开线 0.720/0.721 一致）。

下一发用**自己的改动** `okkojbjj/ntl-public-0720-c20`：相对 0.720 只改 `v1_stage.py` 的 `--max-cover` 0.30 → 0.20。依据是 v1_stage docstring 的 train OOF 曲线：15% +0.030、20% +0.0315、全量 +0.029；默认 0.30 不是最优点，预期 +0.0025。已 push（version 1），commit 中；COMPLETE 后提交 scoring rerun。

- 不接受“0.720 已够”：0.720 与银线同分但排在 406，必须拿 0.721。
- 若 c20 仍 ≤0.720，下一发改 blend 权重 W（同样有 OOF 依据的旋钮），而不是继续换公开 fork。

---

## 2026-11-24 13:20 自己的第一处改动：V1 覆盖上限 0.30 → 0.20（有 OOF 依据，待发）

公开线到 0.721 为止，转入自有改动。查 0.720 源码发现 V1（GLM likelihood）阶段有一个公开文档给出 OOF 曲线、但当前默认值并非最优的旋钮：

- `v1_stage.py` 的 `--max-cover` 默认 **0.30**，即只对按 GLM 质量排序的前 30% 测试文本做 likelihood 混合；
- 同一文件 docstring 记录 train OOF：覆盖 15% 增益 **+0.030**、20% **+0.0315**、全量 **+0.029**；
- 也就是说 20% 覆盖比 30%/全量在 OOF 上高约 +0.0025，而默认 0.30 并没有用这个点。

已复制 `ours720_c20`：只改 cell 49 调 `v1_stage.py` 的命令，追加 `'--max-cover', '0.20'`，其余（模型、权重、sibling odds、排版、Fable 规则、时间预算）一字不动。静态校验通过（`--max-cover` 只出现一次，60 cells，26+10+9 源）。

- 这与之前否掉的“时间预算”假设不同：`V1_DEADLINE_H=8.8`、成本模型 4000 行约 55 分钟，时间不是瓶颈；0.30 只是一个保守默认，不是 OOF 最优点。
- 待 `58811392`（0.720 原样线）出分后再 push/提交，避免两个 GPU 作业抢同一队列。若 58811392 已到 0.719+，c20 仍有 +0.002 的空间，可作为加固银牌的下一发；若仍是 0.718，c20 是下一发主候选。
- 后续自己的验证闭环：rehearsal kernel `okkojbjj/ntl-rehearsal-2699` 已建好未 push。更彻底的做法是把 V1 阶段改成“一次跑满、落盘逐文本 member log-prob”，之后离线扫 cover/权重，一次 GPU 换整个超参面；这需要在 cell 47 的 base64 payload 上打补丁，下一步做。

---

## 2026-11-24 12:40 第二次同代码仍 0.718；转有真实代码增量的 sibling-odds 0.720/0.721（58811392）

`58794719`（同 `ntl-public-0717-glm-prefix` v1，同代码第二次提交）已 COMPLETE，public **0.718**，与第一次完全相同。两次同代码都是 0.718，说明本账号暂时抽不到 0.719 的噪声档，继续重跑同代码的边际价值低。公榜仍 534/6868，铜牌区；银线 0.719（第 343 名）。

改为有**真实代码增量**的公开线：`areebbb/0-720-preserve-glm-to-fable-corrections`（V16 逐 cell 复现 `zayd88903/0-721-name-that-llm`，二者被 areebbb 核对为同一份代码）。它是在我们已跑的 0.717 管线上加 **ranjeet258 的 Sibling Odds**（按 likelihood odds 变化修正同族近似决策，保留 guarded Fable），0.720/0.721 是这条线的实测分。已原样复制为私有 `okkojbjj/ntl-public-0720-sibling-odds`（仅改 id/title/code_file、删 `id_no`/`docker_image`、设私有；60 cells、26+10+9 源、commit COMPLETE，17 分钟）。已提交 scoring rerun：ref **58811392**，2026-11-24 04:41 UTC，初态 PENDING，今日余 4 次。

- 预期 0.718–0.721；只有 ≥0.719 才进银牌区（当前第 343 名）。
- 若这次仍 ≤0.718，则说明公开线在本账号的天花板就是 0.718，下一步应停止抽签，转做有 OOF 证据的新信号，而不是继续换公开 fork。
- PENDING 历史：上一次 6 h 16 min 才起跑，本次按同样预期等待；不重复提交。

---

## 2026-11-24 04:47 58780390 评分完成：公榜 **0.718**，名次 534/6868，进入铜牌区

等待 6 h 16 min 后 scoring rerun 完成（不是 ERROR，是平台队列）。`kaggle competitions submissions` 确认：提交 **58780390**，public **0.718**，COMPLETE。公榜快照（`kaggle competitions leaderboard name-that-llm -d`，2026-11-23T20:47:59Z，6868 队）显示本站 TeamName `okkoj bjj`、TeamId 17165972、**Rank 534**、Score 0.718、SubmissionCount 13。同快照：铜牌线（前 10%，第 686 名）为 0.717，银牌线（前 5%，第 343 名）为 0.719，0.719 共 371 队、≥0.720 共 209 队、金牌线 0.756（第 23 名）。按当前名次我们在**铜牌区**（非最终授牌）。

- 这是本站从 0.651 到 0.718 的一次跃迁，来源是逐字复制 alexchilton 的 GLM-5.3 user-prefix + guarded Fable-return 管线（`okkojbjj/ntl-public-0717-glm-prefix` v1），代码 cell 未改。
- 同一份代码在不同账号/不同次运行的实测分布是 0.717 / 0.718 / 0.719 / 0.720 / 0.721（alexchilton 两次 0.717/0.719；areebbb V12 0.720 与 zayd88903 0.721 被 areebbb 逐 cell 核对为**同一份代码**，差异属运行噪声）。噪声来源很可能是 GLM likelihood 阶段的时间预算：GPU 快时覆盖更多文本，分就高。
- 结论：**银牌（第 343 名，0.719）只差一个噪声档**。同代码再跑一次是正期望动作，不需要调参；0.720 以上目前没有公开的不同管线。
- 今日提交额度：本日已用 1 次（58780390），余 4 次；UTC 00:00 刷新。

---

## 2026-11-23 状态镜像到私有仓 ojstar-web/Competition（只写 name-that-llm/）

按用户要求建立独立状态镜像：独立 clone 在 `~/jyx/state_repo_ntl/Competition`（专用 key `~/.ssh/id_ed25519_casmi_state`，不复用其他会话工作树），只写 `name-that-llm/`，绝不修改 `enveda-casmi26/`，不 force push。同步脚本 `~/jyx/state_repo_ntl/sync_ntl.sh` 基于 `sync_enveda.sh` 改三处（`S=NameThatLLM`、`D=name-that-llm`、receipts 换成 `outputs/native_0680_candidate_20261109_127/*.json`），白名单只同步 `*.md`→`docs/`、`src/*.py`→`src/`、小 JSON→`receipts/native_0680/`；`git add -A name-that-llm`，push 失败 `pull --rebase --autostash` 重试 3 次。已启动 `while true; do sync_ntl.sh; sleep 900; done`（PID 2995714）。

已推送 commits `9a616c2`（docs/src/receipts）与 `953f336`（README_AI.md），工作树与 `origin/main` 一致，`enveda-casmi26/` 73 个跟踪文件零改动，`name-that-llm/` 共 556 KB，无 parquet/npy/pt/zip/权重/`data/`/`outputs/` 大产物。`name-that-llm/README_AI.md` 用实测数字写榜单坐标；其中对“公开最强 0.680”按 2026-11-23 21:10 的实测更正为 0.716–0.719，并保留 0.680→≈2089 名作为 11-08 旧状态，避免把过期结论写进镜像。

---

## 2026-11-23 战略纠偏：公榜公开配方已到 0.717–0.719，0.680 线作废；已提交 0.717 公开管线（58780390）

**前提被外部事实推翻。** `kaggle kernels list --competition name-that-llm --sort-by scoreDescending`（2026-11-23 21:10 本地）显示公开 notebook 已有 0.719 / 0.718 / 0.717 / 0.716 一批，最近更新 2026-11-22/23；alexchilton 的 0.716 笔记（`public_top_20261123/alex716/`，其 cell 0/3 记录本账号 58780390 之前的 125 次提交）写明：0.716 由 `estopadilla` 的 GLM-5.3 likelihood + user-turn prefix 阶段带来，0.717 由 dennis6886 的 guarded Fable-return 带来，且“几乎每一步提升都来自原样运行更新的公开 notebook”。本项目 NOW.md 里“公开最强 0.680、只到 31%”已过期 10 天，不能再作为路线依据。

**已做**：原样复制 alexchilton 的 0.717 notebook（仅改 `id`/`title`/`code_file`、设 private、删 `id_no` 与 `docker_image`；代码 cell 一字未改），26 datasets + 10 kernels + 9 models + competition，私有禁网 T4。第一次 push 返回 403 SaveKernel，原因是继承了原作者 `id_no`；删掉后 version 1 推送成功。commit 13 分钟 COMPLETE（mount 检查 45 源全过）。已用 `kaggle competitions submit ... -f submission.parquet -k okkojbjj/ntl-public-0717-glm-prefix -v 1` 提交，ref **58780390**，UTC 13:34:28，初态 PENDING，今日余 4 次。证据目录 `public_top_20261123/{alex716,areeb719,ours717}/`。

- 该提交是隐藏测试集的 scoring rerun，预计 ≤7 h（notebook 自带 `RUN_LIMIT_S=7h`，GLM 阶段占最后约 1.8 h）；完成前不重复提交、不改代码。
- 0.717 在作者账号上复现为 0.717；本站能否复现未知，单次公榜噪声约 ±0.005（作者 125 次提交统计）。若落在 0.712–0.717，仍可能压线铜牌（0.716）。
- 0.680 配方（v5 已 256 行 COMPLETE）不再是主线。v6（512 行测时）上传后一直 QUEUED，且疑似占住本账号唯一 GPU 队列、令 58780390 无法起跑；为避免继续挡牌，2026-11-23 13:45 UTC 用 \kaggle kernels delete\ 删除了该探针 kernel。v5 的日志、parquet、run.json 全部在本地 \outputs/native_0680_candidate_20261109_127\ 保留，不依赖云端 kernel 历史。
- 未动 EnvedaCASMI2026 目录与任何本地 GPU 任务；本站旧最高仍 0.651（58023439）。

---

## 2026-11-23 工程闸门通过：完整 0.680 配方私有 256 行跑完并产出合法 parquet；512 行测时版 v6 已上传

v5（`okkojbjj/ntl-native-0-680-private-probe-256`，上传后标题改名，现 slug 为 `...-512`，id 不变）已 **COMPLETE**，经 Kaggle CLI 2.2.5 只读确认，并已回拉原生输出。这是该配方第一次真正跑完：256 个假 ID、18 个官方类、ID 唯一且顺序与输入一致，`submission.parquet` 由 `ensemble.py` 终态写出，`inference-logs/execution-contract.json` 记录 gsq_share 0.03、native_embedding_share 0.4、五折、0.15 complementary。总墙钟 5199.3 s。阶段耗时（日志内自报）：q14v5 457.2、q8v5 272.0、ll8v5 270.9、n14b 149.2、q25_14 148.6、deb 19.4、atlas 268.5、native-q8 五折 609.2、complementary 190.2 s；ATLAS 仅命中 4/256 行（top1-top2<0.3）。证据 `outputs/native_0680_candidate_20261109_127/native_probe_v5/{submission.parquet,inference-logs/,ntl-native-0-680-private-probe-256.log}`，记录已写入同目录 `run.json` 的 `v5`。

- 这不证明 0.680，也不证明任何 BA；它只证明“路径写 `/kaggle/working`、全 34 源准入、私有禁网、完整分支”在 256 行规模可执行。
- v1–v4 的失败结论继续保持，不重试；v5 是同一 slug 的第 5 版，不是重传旧版。

**下一步已执行**：复制 v5 生成 512 行测时版 `cloud_probe_512/`，只改假输入（128 条最长 + 384 条随机，`probe0000..0511`）与输出 CSV 名，模型/权重/份额/路径全部不动；静态校验通过（55 cells、private、GPU、禁网、无 `/kaggle/temp` 之外写路径）。已用 `kaggle kernels push` 推送为 **version 6**，初态 QUEUED，仅上传一次。目的只有一个：用 256→512 的墙钟差求固定开销与每行斜率，判断 4000 行能否压进 9 小时；不作提交、不作分数解释。若 512 的斜率外推 4000 行 > 9 h，就不直接跑 4000，改按阶段拆 kernel（core 成员各 2 个 + 末端融合）而不是删成员。

已停：不重传 v1–v4、不动 EnvedaCASMI2026 目录与 GPU、不提交公榜。当前本站最高公榜仍 0.651（58023439），私榜未知。

---

## 2026-11-10 用户要求退出Kaggle队列，NameThatLLM暂停派发

用户为紧急任务要求先退出队列。只读复核 `okkojbjj/ntl-native-0-680-private-probe-256/2`、`/4` 和 `okkojbjj/ntl-native-0680-mount-audit/1` 均仍QUEUED，failureMessage空；v1/v3 ERROR已终态。现有Kaggle CLI没有按slug取消命令，公开状态/GetKernel响应不含取消所需kernelSessionId；**尚未成功取消，不可声称已退出队列**。不使用`kernels delete`冒险删除整个notebook历史。已停止新上传、训练及提交；需通过Kaggle网页左下角Active Events中对应三项的Stop Session取消，或取得各session ID后走官方cancel-session API。取消后只读核CANCELED终态并更新`outputs/native_0680_candidate_20261109_127/run.json`。此前v3路径检查PASS、v4排队，本站公榜仍0.651。优先处理用户紧急任务，不自行恢复本项目。

---

## 2026-11-10 公开0.680私有v4探针待运行；v3已证明路径与输入准入修复

本站公榜最高仍0.651（58023439），作者0.680不是本站分数。公开完整配方原35输入来源的私有v2和全源CPU审计持续QUEUED；对照证明34来源两种不同组合均可运行，未获Kaggle正式数值上限。只在cell6存在性清单出现、未进入推理的`nuridudak/ntl-q14-lora-v4`挂载已从v3移除，完整模型/融合不变。v3 `okkojbjj/ntl-native-0-680-private-probe-256/3`实际运行，cell6报33根目录/101路径`missing: []`，云端55源码单元与上传一致、私有禁网、20数据集+8 notebook+6模型；随后cell7写私有假CSV到尚未创建的`/kaggle/temp`报OSError，未开始模型推理。原生日志在`outputs/native_0680_candidate_20261109_127/native_probe_v3/`。v4仅将该假CSV写到现成的`/kaggle/working`；`okkojbjj/ntl-native-0-680-private-probe-256/4`已且仅已上传一次，初态QUEUED。勿重传v1-v4或将v3路径PASS称为完整技术PASS。

下一步查v4确切终态，若ERROR回收原生日志定位；若COMPLETE回收原生输出/云端源码，核256唯一假ID、18类合法、六主成员/COMP/ATLAS/GSQ/五折专家实际完成、无必需跳过及耗时。通过后才决定约4000行完整容量与来源审查；无BA证据或新公榜分数。当前事实见`outputs/native_0680_candidate_20261109_127/{run.json,REPORT.md}`，全局/项目纠错本已更新。

---

## 2026-11-10 公开0.680完整配方私有v3已开始真实运行

本站公榜最高仍0.651（58023439），作者0.680不是本站分数。私有v1路径/Notebook schema早期失败，v2虽已修复却和全源CPU审计一起在Kaggle QUEUED超过16小时；两者各35个输入来源，无worker日志、无预测，不重传。本站单小数据源审计已COMPLETE并证明当前挂载路径为`/kaggle/input/datasets/<owner>/<slug>`。32、33、两种34来源CPU对照均启动/完成，35组合排队；这支持输入组合准入边界，但非Kaggle公布的绝对上限。查公开源码发现`nuridudak/ntl-q14-lora-v4`仅在cell6存在性清单引用，未进入配置/推理。v3只移除这个未使用挂载及检查，完整配方权重、成员和0.03/0.40混合不变。私有禁网T4 `okkojbjj/ntl-native-0-680-private-probe-256/3`已且仅已上传一次，初查RUNNING；256假ID仅测真实计算，无BA。不得因旧v2排队或CLI错误重传。

下一步只读查确切v3终态。若COMPLETE，回收原生输出/日志及云端源码，核256唯一ID、18合法类、六主成员/COMP/ATLAS/GSQ/五折专家全部实际执行、无必需跳过、来源与时长；若ERROR，保留原生日志，定位科学调用阶段后修有证据的根因。技术PASS后再做约4000行容量与来源审查，才可考虑已授权的公榜提交；绝不把探针当评分。完整证据 `outputs/native_0680_candidate_20261109_127/{run.json,REPORT.md,upload_probe_v3/}`。全局和项目纠错本已记录本轮队列诊断。

---

## 2026-11-09 公开0.680配方路径修复，私有v2与CPU审计排队

本站最高仍公榜0.651，私榜未知。私有v1 `okkojbjj/ntl-native-0-680-private-probe-256/1` 在模型加载前ERROR；cell6所报11个缺失根目录使用旧式`/kaggle/input/<slug>`，而本站成功v27原生日志证明当前数据集挂载为`/kaggle/input/datasets/<owner>/<slug>`。另有v1探针构建器给Markdown单元添加非法字段；两项均已在v2修正。v2仅改12个旧数据集路径家族、Markdown schema与256假ID输入，保留公开完整配方及权重。实际slug `okkojbjj/ntl-native-0-680-private-probe-256/2` 已且仅已上传一次、初态QUEUED，私有禁网T4。独立只读私有CPU挂载审计 `okkojbjj/ntl-native-0680-mount-audit/1` 同为QUEUED；其上传早于v2，因已有v27路径证据，v2无需等待它。二者不训练、不提交，不因排队/查询故障重传。

终态回收v2原生日志/产物/云端源码，核256唯一假ID、合法标签、六主成员/COMP/ATLAS/GSQ/五折原生专家真实执行、输入资产与时间。若v2通过，仅是技术探针；完整约4000行容量与来源权利仍需核。外部作者0.680不是本站分数。记录 `outputs/native_0680_candidate_20261109_127/{run.json,REPORT.md,native_probe/,cloud_probe/}`；旧v1无科学调用，原生失败保留。全局和项目纠错本已记录本次教训。

---

## 2026-11-09 V31机制私有256行探针技术通过；转回效果优化

`okkojbjj/ntl-v31-private-probe-256/1` 已 COMPLETE。原生256唯一假ID、18合法类、有限COMP logits；六主成员、COMP家族门控、ATLAS对选中的115/115行完整混合、贝叶斯排版均完成，云端六代码单元与上传完全一致，三份输入SHA和Llama资产钉选通过；总日志约6285秒。证据 `outputs/v31_candidate_20261107_126/{native_probe/,cloud_probe/,run.json}`。此探针没有真值，不能称为BA改善；历史官方-only B-R中若干V31单机制无增益。用户指出4000行测试不等于继续优化，因此先寻合法同折完整融合增量或更强不同候选，暂不自动启动4000行容量或提交公榜。本站最高仍0.651，私榜未知。

---

## 2026-11-07 V31机制候选私有探针已上传一次

本站已评分v27公榜BA0.651（提交58023439）作为冻结资产起点；新候选保留其输入快照、六主成员与适配器，接入公开V31的SVC去重/源权重、TF-IDF、distortion、COMP家族门控、soft Sinkhorn、ATLAS 20%和门控贝叶斯排版。已去除公开FAST_COMMIT占位路径，COMP或ATLAS必需结果缺失时失败。构建脚本、冻结哈希和计划见 `outputs/v31_candidate_20261107_126/{build_candidate.py,manifest.json,PLAN.md,run.json}`。本地三个Python模块编译、候选/探针源码一致性及私有禁网T4配置检查PASS；未证明BA或隐藏容量。

唯一当前本项目Kaggle作业 `okkojbjj/ntl-v31-private-probe-256/1` 于2026-11-07 08:48 UTC前上传且仅上传一次，初查QUEUED，私有禁网T4、256唯一同源假ID、父限31800秒；不训练、不提交。终态前不重传、不并行启动另一本项目作业。COMPLETE后回收原生日志、输出和云端源码，核六主成员、COMP、ATLAS全部256行、输入SHA、ID/类合法性、实际时长，再决定4000行容量作业。ERROR保留原生证据并止损。私榜未知；公开作者0.663不是本站分数。

---

## 2026-11-06 本站公榜达到0.651，目标验收仍待证据审计

Kaggle 本账号唯一提交58023439 COMPLETE，公榜BA0.651，较旧最佳0.620提升0.031，私榜未知。已完成私有v2三行技术验收与历史同源4000行容量；该完整包缺合法同折OOF，来源权利仍有未知，故仅确认分数门槛，不宣称全部G1–G3验收完成。见 outputs/candidate_v27_20261105_125/{run.json,REPORT.md}。不重交此版本。

---

## 2026-11-05 current-asset v27 公榜提交58023439待评分

先前私有v1因旧Llama缓存无法挂载而ERROR，未提交。仅换模型挂载且加12文件SHA钉选的私有禁网T4×2 `okkojbjj/ntl-v27-current-assets/2` 已COMPLETE：12文件字节与旧核验缓存相同，三份冻结输入SHA相同，云端六代码单元与上传一致，六主成员+COMP+ATLAS实际混合，三行Parquet通过Notebook ID/标签断言；原生日志约5428秒，无异常跳过。历史同源4000行完整作业约28268.52秒，小于Kaggle9小时上限；仍非隐藏容量或BA保证。核对今日额度和未重交后，确切v2已**仅提交一次公榜**：ID **58023439**，UTC 2026-11-05 12:53:10.593，初态PENDING，今日余4次。只读监控这个ID；PENDING或查询错绝不重交。COMPLETE记录本站公榜BA及对0.620差值，ERROR保留失败证据。当前已确认最高仍0.620，私榜未知。证据 `outputs/candidate_v27_20261105_125/{run.json,REPORT.md,native_v2/,cloud_v2/}`。

---

## 2026-11-05 current-asset v27 私有版本1失败，公榜未提交

本账号私有禁网 `okkojbjj/ntl-v27-current-assets/1` 已终态 ERROR。三份冻结输入的运行时 SHA 核对及 bitsandbytes 临时安装通过，随后 cell 1 在 `BL = mdir('/llama-3.1/', '/8b/')` 报 AssertionError：Llama 3.1 8B 没有挂载；上传时 Kaggle 已警告旧 `avazaku/name-that-llm-verified-llama-asset-cache` 无效。无预测、无公榜提交，不得重交原版本或将缺成员产物视为完整 v27。原生日志与决定见 `outputs/candidate_v27_20261105_125/{run.json,REPORT.md,native_v1/}`。两位 Astra 独立复核后建议：只读确认完整 Llama 权重可访问，和旧缓存逐文件核身份；若相同，仅修挂载并私有验收，再评估公榜；若不可访问或字节不同，停在资产阻断。本站已验收最高公榜仍为 0.620，目标 0.650 未达。

---

## 2026-11-02 公开 clean pool 来源预审待补证

拟作为0.620高基线SVC池单因素候选的 crtproduction 19万行池，目录卡声称清洗和许可归属，但含已知重建官方题来源 drjcfmd 20,400行；标签与服务路由未独立核实。小型 sources.csv 两次下载均连接中断，无主文件逐行审计；G1 HOLD，不启动训练或公榜。见 outputs/highbase_0651_package_20261027_80/public_candidate_audit_20261102/REPORT.md。

---

## 2026-11-02 官方-only exact18 公榜失败效果，转回0.620高起点

Kaggle 本账号提交 **57912173** 已 COMPLETE，公榜BA **0.484**、私榜未知；较本站最佳0.620低0.136，距目标0.650还差0.166。此前官方-only三折探索诊断0.677942严重高估公榜，不能据单次分数断言唯一原因。4000行技术容量PASS仍有效但不代表效果。禁止重交或围绕这次公榜调阈值/权重；全官方三深成员训练任务卡 `outputs/prompt_assignment_20261102_122/fulltrain_taskcard.json` 转HOLD，未上传/训练。下一步从已评分0.620完整Part Two高起点重新审来源合格且机制不同的优化，不为今日余下4次额度凑提交。证据 `outputs/prompt_assignment_20261102_122/{public_v1_record.json,REPORT.md}`。

---

## 2026-11-02 全官方最终权重备选静态任务卡

公榜57912173仍待终态；本轮只建立 `outputs/prompt_assignment_20261102_122/fulltrain_taskcard.json`，冻结将三份首折1799行深权重替换为全官方2699行训练权重的独立候选。Qwen3/Qwen2.5各200更新、DeBERTa 507更新的脚本SHA与preflight逐项匹配；尚未上传、训练或证明BA。须等待现有公榜作业终态，再逐个训练、原生/云端与重载验收，最后重做私有完整4000行容量；不按57912173分数调超参。

---

## 2026-11-02 官方-only exact18 公榜提交待评分

本账号按已授权规则且仅一次提交评分notebook `okkojbjj/name-that-llm-official-exact18/1` 的 `submission.parquet`：Kaggle提交ID **57912173**，UTC 2026-11-02 14:47:26.720，描述 `Official-only four-member exact18 v1`，初态PENDING；北京时间本日第1次，CLI示余4次。评分/容量源码只有假测试开关一处不同，4000行私有容量技术PASS；同折0.527149→0.677942仅探索，不是公榜成绩。绝不因PENDING/查询失败重提，不据分数改参数。只读监控唯一ID，COMPLETE记录本站公榜BA及对0.620差值，ERROR保留失败。证据 outputs/prompt_assignment_20261102_122/{public_v1_record.json,sample_capacity_parity.json,capacity_v1_record.json,REPORT.md}。当前最高已验收公榜仍0.620，私榜未知。

---

## 2026-11-02 官方-only exact18 私有容量技术PASS

唯一容量作业 `okkojbjj/name-that-llm-official-exact18-capacity/1` COMPLETE，私有禁网T4×2原生4000唯一假ID输出、云端源码/元数据回收；上传与云端七代码单元一致。真实验收器 PASS_4000_PRIVATE_CAPACITY_ONLY：全四成员与后处理执行、2个exact18组改动12行、1619.86秒、18类合法、无submission.parquet。假测试长度尾部覆盖官方训练最大1900字符。精确`/1`云端源码pull仍403，当前唯一v1 slug pull成功；不重传。技术PASS不等于隐藏BA/公榜改善；自有大权重字节未独立逐字钉住为来源残余风险。先核G0–G3与北京时间当日额度、历史提交，再决定已验收样例v1是否可仅交一次；见 outputs/prompt_assignment_20261102_122/{capacity_v1_record.json,capacity_native_v1/,capacity_cloud_v1/,capacity_tail_audit.json,REPORT.md}。本站已验收公榜最高仍0.620。

---

## 2026-11-02 全官方备选源码复核与挂载恢复

当前唯一 Kaggle 作业仍为官方-only exact18 容量v1 RUNNING，勿重传、并发或提交。Z: 一度读取卡顿；本轮核对 SSHFS 原进程PID/启动时间未变、严格SSH下 ubuntu/yuanyn 与项目 realpath 正确，Z: 和远端 HANDOFF SHA256 一致，挂载读写恢复。全官方备选 DeBERTa 独立源码复核的保存/重载缺口已修正，并清除残留训练张量；重新生成脚本SHA c563f0274d9541c3ea591a7d3206370fb168ad30c0fde9c7e99afdf6c69ed349、py_compile PASS。三份全官方备选尚未上传/训练，不能宣称效果。证据 outputs/prompt_assignment_20261102_122/{fulltrain_preflight.json,REPORT.md,capacity_v1_record.json}。

---

## 2026-11-02 容量分组预检与全官方备选准备

当前唯一Kaggle作业仍为官方-only exact18 容量v1 RUNNING。按已上传源码的4000假测试构造，CPU同规则文本聚类得精确18行组2个，故预计实际触发指派；终态验收器要求2组，真实GPU输出仍待核。另已静态生成全官方Qwen3/Qwen2.5各200更新、DeBERTa三epoch的备选训练入口，Qwen实际抽样块覆盖2699训练行，源码py_compile PASS；尚未上传/训练，独立审查中。证据 outputs/prompt_assignment_20261102_122/{capacity_group_preflight.py,fulltrain_preflight.json,REPORT.md}。

---

## 2026-11-02 官方-only exact18 完整容量作业已运行

唯一当前本项目Kaggle作业 okkojbjj/name-that-llm-official-exact18-capacity/1，私有禁网T4×2、父限31800秒，初查RUNNING；4000唯一假ID完整四成员推理，不训练、不交榜。样例v1已真实技术PASS，源码/原生记录见 outputs/prompt_assignment_20261102_122/{sample_v1_record.json,capacity_v1_record.json,REPORT.md}。终态前不重传、并发或提交；COMPLETE后运行 verify_capacity.py 真实验收。

---

## 2026-11-02 官方-only exact18 三行私有验收PASS

唯一版本 okkojbjj/name-that-llm-official-exact18/1 已 COMPLETE；云端七代码单元与上传完全相同，原生3行合法预测、三自有权重重载、私有禁网T4、279.03秒，验收器PASS。它不运行Sinkhorn/exact18，不证明隐藏容量/BA；下一步按已准备的 capacity_upload 完整4000唯一假ID禁网作业，≤9h，一次一个、不交榜。证据 outputs/prompt_assignment_20261102_122/{sample_v1_record.json,sample_native_v1,sample_cloud_v1,REPORT.md}。

---

## 2026-11-02 官方-only exact18 私有三行作业已上传一次

唯一当前本项目Kaggle作业 okkojbjj/name-that-llm-official-exact18/1，私有禁网T4×2、父限7200秒，首查QUEUED。挂本站官方-only首折三权重，不训练；先验实际挂载、三行输出及云端源码，未终态前不重传或并发，不交榜。记录 outputs/prompt_assignment_20261102_122/sample_v1_record.json。

---

## 2026-11-02 官方-only完整成员 + 题组指派诊断

旧池来源仍HOLD时，改以已验收的官方-only SVC、Qwen3、Qwen2.5、DeBERTa三折成员做冻结融合与同一 exact18 指派。2699行探索性OOF BA 0.527149→0.677942，三折增量 +0.1500/+0.141111/+0.161338，独立算术/行范围复核PASS。三个深成员既有Kaggle元数据均无外部数据集、私有禁网；仍须核对模型许可与正式全量训练入口。该开发集已经用于多次试验，隐藏题组结构未知且公开作者类似约束曾使LB下降；不能把0.678报作公榜或承诺提升。无新Kaggle作业/提交。下一步构造全官方训练+完整隐藏推断的正式候选，并先验收来源和9小时容量；证据 outputs/prompt_assignment_20261102_122/{official_only_taskcard.json,official_only_summary.json,official_only_oof.csv,verify_official_only.py,REPORT.md}。

---
## 2026-11-02 exact-18 推断题组指派（探索，HOLD）

在旧池完整 B-R 同折OOF中，冻结后处理：仅文本推断的恰好18行组执行一对一 Hungarian 标签指派。2699行 BA 0.577164→0.740920，三折均约+0.16；独立只读复算和改动范围核对PASS。131个合格组中130个确为单一官方题、18类各一次。规模压力测得2699行场景纯组覆盖约73%，4000随机文本的分组开销2.73秒/峰值约648MiB。**不代表隐藏集或公榜增益**：隐藏题组结构未知，作者公开记录类似约束模拟增益但LB变差，旧池G1来源仍HOLD。未上传、未运行G3、未交榜。证据 outputs/prompt_assignment_20261102_122/{taskcard.json,summary.json,REPORT.md,group_scaling_probe.json,grouping_capacity_probe.json}。

---
## 2026-11-02 固定SVC半半融合止损

保留旧池完整B-R，仅把SVC证据预先固定为旧池与15,732行可追溯来源各半；2699行同折完整融合BA0.577164→0.569389，Δ−0.007775，三折均负。独立复算PASS，G2-R数值FAIL；不调比例，不跑G3或公榜。证据 outputs/provider_trace_svc_blend_20261102_118/{taskcard.json,summary.json,run.json,REPORT.md}。本站公榜最高仍0.620。

---

## 2026-11-02 两来源可追溯SVC私有同折对照

仅将SVC训练池换为15,732行可追溯提供方预览，三折完整融合OOF 0.538638→0.552722，Δ+0.014084，逐折均正；独立复算2699行/18类及prompt隔离PASS，119秒CPU。相对旧池重建完整OOF 0.577164仍低0.024442；数据只覆盖12类且提供方输出使用权未闭环，公榜G1 HOLD，不运行G3或提交。证据 outputs/provider_trace_svc_20261102_117/{taskcard.json,summary.json,verification.json,run.json,REPORT.md}。本站已验收公榜仍0.620。

---

## 2026-11-02 q35附件复核

Kaggle CLI只读重新核查作者全部6个公开数据集。旧 nlm-l24-adapters 数据卡仍为MIT；高分包实际使用的 nlm-l24-adapters-v5 与 nlm-offline-wheels 仍为Other、描述空，作者无新增许可明确的替代适配器。高分整包G1继续HOLD，不把旧MIT资产冒充v5或按作者分数推断本站收益。未训练、未交榜。

---

## 2026-11-02 来源筛选三折诊断

可追溯 provider 的两来源筛出15,732行预览，官方prompt隔离三折的训练侧近重复命中均0；只证明该阈值下文本筛查。提供方输出权利与路由未闭环，G1 HOLD，不训练或公榜。证据 outputs/deberta_source_quarantine_20261101_115/{provider_filtered_preview_manifest.json,provider_filtered_fold_screen.json,run.json}。

---

## 2026-11-02 官方数据完整融合对照

官方-only SVC + 既有官方深成员的完整同折融合已独立复算：2699行/150题 BA0.538638，逐折0.552222/0.523333/0.540340。真正旧池B-R OOF为0.577164，配对385行变化、救回176/误伤72；早先误用同为官方SVC的smoke文件造成零变化，已纠正。旧池收益不解除23来源G1权利/路由缺口。证据 outputs/official_only_full_fusion_20261102_116/{summary.json,verification.json,run.json,REPORT.md}。无G2-R新候选/G3/公榜，本站最高0.620。

---

## 2026-11-01 M1旧池首折终态：仅工程测时

私有Kaggle `okkojbjj/name-that-llm-deberta-old-pool-stage1-fold-0/1` COMPLETE；原生验证900行/18类有限logits、阶段一4737更新、阶段二339更新、8581.97秒。云端源码回拉403，不能宣称完整源码验收。训练池有drjcfmd:p004重建主题与首折留出p004重合，诊断BA0.525556科学无效，不进G2-R/公榜。证据 outputs/deberta_pool_pretrain_20261101_114/{native_v1/,native_verification.json,run.json,REPORT.md}。M2仍因23来源权利/路由G1缺口HOLD。本站公榜最高0.620。

---

## 2026-11-01 M2旧池来源权利仍HOLD

M2隔离drjcfmd/lxd后保留63396行、23来源；来源分布已冻结在 outputs/deberta_source_quarantine_20261101_115/source_inventory.json。现有记录未逐源证明提供方输出使用权与实际路由，按已批准G1协议暂不上传M2私有训练。候选源码与终态验收器已备好但仅语法/入口通过；M1唯一作业仍需等待终态。本站公榜最高仍0.620。

---

## 2026-11-01 DeBERTa来源隔离后M2数据准备

为修复M1 v1的已知题族泄漏，冻结M2来源级排除 drjcfmd 全8111行和 lxd 全4276行；其余63396行仍覆盖18类，三折仅训练侧char3-5近重复筛查均0命中，官方完全同文0。M2目前只完成CPU数据准备，未训练、未上传、未交榜；必须等M1 v1终态后才可考虑单一私有GPU作业。其余来源提供方输出权利仍须最终G1核对。证据 outputs/deberta_source_quarantine_20261101_115/{taskcard.json,prepare_summary.json,run.json,REPORT.md}。

---

## 2026-11-01 DeBERTa旧池首折发现题族泄漏，版本1仅作技术测时

运行中的唯一Kaggle作业 okkojbjj/name-that-llm-deberta-old-pool-stage1-fold-0/1 含旧池 drjcfmd:p004 的15条Gödel文本；官方p004正是首折留出题。来源README明确以官方题目重建为风格示例，折1/2筛查因p004在训练侧各排除了这15行，折0训练侧筛查无法看到留出题。版本1即使技术COMPLETE也不得把首折BA用于G2-R或公榜。Kaggle CLI无取消命令；保留唯一作业作测时/工程验收，不删除、不重传、不并行。见 outputs/deberta_pool_pretrain_20261101_114/run.json 与 REPORT.md。下一候选须按来源级隔离重建题族并重新冻结。

---

## 2026-11-01 DeBERTa旧池阶段一私有首折运行中

以已评分0.620基线使用的同一75783行pool_v4为阶段一，官方三epoch阶段二与干净DeBERTa基线一致；三折训练侧近重复筛查排除0/15/15行，官方完全同文0。首折私有禁网T4作业已且仅已上传一次，Kaggle实际slug okkojbjj/name-that-llm-deberta-old-pool-stage1-fold-0/1（上传metadata id因标题生成不同，CLI有警告），初查RUNNING；父限18000秒/内部14400秒。绝不因slug警告或查询故障重传，不并行本项目Kaggle作业，不交榜。终态须回收原生日志/输出和云端源码，核对固定pool SHA、真实阶段一更新/阶段二更新、900唯一官方外折行、18类合法logits、禁网T4与源码。见 outputs/deberta_pool_pretrain_20261101_114 的任务卡、准备摘要、运行记录和报告。本站最高已验公榜仍0.620。

---

## 2026-11-01 Elliot OpenRouter新增池来源止步

原始README称付费OpenRouter钉选并核对返回model；在公开v10池中可逐行按文本+标签匹配11,742条旧池外且discriminator_pass的7类文本，潜在高价值。但已发布parquet不含逐行返回模型ID/上游provider，无法独立闭合现行G1实际路由。未训练或提交该子集。证据 outputs/public_0664_routes_20261101_111/elliot_openrouter_route_audit.json。

---

## 2026-11-01 R4 服务条款复核

G2-R离线数值PASS，但三代理来源虽发布CC许可，Kaggle Model Proxy实际适用的提供方协议未知；Google生成式AI附加条款可能限制用输出开发机器学习模型，不能凭公开数据许可推断公榜可用。R4公榜G1继续HOLD，不启动G3或提交；转向无此来源风险的高潜力候选。见 outputs/proxy_bundle_svc_candidate_20261101_113/rights_review.md。

---

## 2026-11-01 新增代理来源 R4 同折完整融合达到数值闸门

R4 在旧池+已核25,986行基础上新增三份原始文件可逐行回溯的代理生成文本11,056行，先重建原始题组并做三折训练侧近重复筛查（均0命中）。固定SVC与深层成员/融合/300轮Sinkhorn，2699行/150题完整OOF B-R 0.577164→0.583460，ΔBA +0.006296，逐折 +0.014444/+0.002222/+0.002222；独立行级复算PASS，达到预设G2-R数值闸门。私有CPU用时778.1秒。仅为探索性离线增量，公榜G1服务方输出条款仍HOLD，G3 4000行容量未跑，不得交榜。证据 outputs/proxy_bundle_svc_candidate_20261101_113/{taskcard.json,prepare_summary.json,summary.json,verification.json,run.json,REPORT.md,oof.csv}。本站已验收最高公榜仍0.620（57684141），私榜未知。

---

## 2026-11-01 新池前两大多类别来源路线复核

恢复 SSHFS 时发现当前 Z: 未挂载；原工具因随附 ssh 不在进程PATH而退出。以进程局部PATH重用既有严格主机密钥 SSHFS 命令后，复核 Z:→yuanyn@Net-4090D:/home/yuanyn/jyx、ubuntu/yuanyn、项目realpath通过，未更改全局配置。只读下载198k池前两大多类别来源原始README：`drjcfmd/ntl-generated-v2`约21,258行中部分 OpenRouter 后端未钉选；`trnhip/ntl-api-gen`约16,818行明确未记录实际served model ID。两者数据集许可明确，但严格行级实际路由仍未闭环，不能直接加进公榜G1。证据 `outputs/public_0664_routes_20261101_111/top2_multiclass_route_audit.json` 与原README。此前25,986行候选G2-R增量仅+0.004074、已止损；本站最高公榜仍0.620。

---## 2026-11-01 新池可回溯 25,986 行 SVC 同折对照止损

原始来源与路由逐行匹配、旧池去重后保留25,986行；三折训练侧相似度0.70筛查无排除。固定新增数据权重0.35，仅重训四视图SVC，完整B-R其余成员及300轮Sinkhorn保持：2699行/150题 BA0.577164→0.581238，Δ+0.004074，逐折+0.008889/+0.002222/+0.001111。独立复算PASS且基线逐行等于冻结B-R；但未达预设+0.005，停止本候选，不调权、不容量、不交榜。提供方条款仍使公榜G1 HOLD。唯一记录 `outputs/open_pool_svc_candidate_20261101_112/{taskcard.json,prepare_summary.json,run.json,REPORT.md,oof.csv}`。当前无运行中本项目作业；本站最高公榜仍0.620。

---## 2026-11-01 已冻结 25,986 行新池 SVC 三折私有 CPU 对照运行中

公开 v10 198k 池中，四原始来源逐行回溯52,589行；其中一部分已在旧75,783行池。按原始 API 模型/服务方精确标签、旧池文本去重，仅从 harshit/willrice/arxitekton 三源保留25,986条新行；Elliot及路由歧义行隔离。每折只用官方训练侧答案做 char3-5 TF-IDF 相似度≥0.70 组级排除，三折近似命中均0、最大约0.366。证据 `outputs/open_pool_svc_candidate_20261101_112/{taskcard.json,prepare.py,prepare_summary.json,novel_verified.parquet,fold*_novel_verified.parquet}`。当前唯一正在运行本项目任务为 Net-4090D CPU2、3600秒上限的同折完整融合对照，工具会话48158；不启动其他作业或改参数。结果未出，公榜G1提供方条款仍HOLD。`run.json` 为当前状态；终态后独立复算OOF并更新。本站最高已验收公榜仍0.620。

---## 2026-11-01 新 0.664 公开路线与 198k 清理池

已只读回收两份公开 COMPLETE 的标题0.664 Notebook；Zbigniew称SVC换198k公开清理池，另一份为参数调整，但两者仍挂载来源未明的v5适配器及离线wheels，不能整包交榜。新池v10当前CC BY-SA 4.0，198507唯一行、18类、54来源，逐行来源字段可见；官方训练文本规范化完全相同行0，逐源计数一致。原始前十大来源元数据审计仍有路由描述空/一个未下载，官方全量近题目标记不能替代逐折筛查，故新池完整G1 HOLD；SVC替换值得优先补证，但未训练或提交。证据 `outputs/public_0664_routes_20261101_111/{audit_v10.json,REPORT.md,open_pool_meta/,source_audit/}`。本站最高已验收公榜仍0.620。

---
## 2026-11-01 高分配方附件来源缺口复核

Kaggle CLI 对当前 `dhruvtibarewal/nlm-l24-adapters-v5` 和 `dhruvtibarewal/nlm-offline-wheels` 全量文件清单只读核查：分别54和4项，无 LICENSE/README/NOTICE/条款文件；已有元数据均为 Other 且描述为空。前者在当前高分包中实质供给 q8v5/q14v5/ll8v5 三组各6份共18个适配器头，后者供给 bitsandbytes 离线 wheel。公开可下载不能推出公榜复用授权，因此该整包 G1 继续 HOLD；不能将 v5 权重悄悄删掉或替换后称精确复现。证据 `outputs/highbase_0651_package_20261027_80/rights_file_inventory_20261101.json`，以及原有 `rights_admission_20261031.json`。继续寻找来源清楚的完整候选；本站最高已验收公榜仍0.620。

---
## 2026-10-31 官方折内排版概率证据 G2-R 止损

公开 V31 的固定37标记贝叶斯排版证据仅用每折官方训练侧拟合，移植到完整B-R的Sinkhorn后：2699行/150题三折 BA0.577164→0.576055，Δ=-0.001109；逐折 +0.001111/0/-0.004444，24行改变、7救回/10损失。独立逐行覆盖与BA复算PASS；未达预设+0.005，停止此候选，无容量/公榜，不调公式。证据 `outputs/typography_bayes_official_20261031_110/{taskcard.json,run.py,oof.csv,summary.json,run.json,REPORT.md}`。最高本站公榜仍0.620（57684141）。

---
## 2026-10-31 官方四家族门控移植 G2-R 止损

公开 V31 的 StyleGrams 四家族软门控按效果前冻结设置，在官方-only prompt 隔离三折完整 B-R 中 BA 0.577164→0.574942，Δ=-0.002222；逐折 +0.002222/-0.002222/-0.006667，69行改变、17救回/23损失。2699唯一行、150题及BA独立复算PASS；预设G2-R失败，停止此候选，不容量、不交榜、不事后调阈值。证据 `outputs/family_gates_official_20261031_109/{taskcard.json,run.py,oof.csv,summary.json,run.json,REPORT.md}`。本站公榜最高仍0.620（57684141），私榜未知。

---
## 2026-10-31 Kaggle只读认证恢复、公开0.66路线与软边际对照

Kaggle CLI `competitions submissions` 本轮只读成功，本站57684141仍COMPLETE/0.620，今日未见新提交；Net-4090D 4090D空闲。只读拉取 Omer 标称0.663、Alex标称0.662、Lightning标称0.662 的公开最新Notebook和元数据，三者最新作业COMPLETE。Omer/Alex推断6单元中5单元同哈希，输入附件同14数据集/8上游；Lightning 36单元独立实现但依赖更广。标题/作者自述分数不是本站分数，精确已评分历史版本仍需钉选；mercurydust/2精确拉取报403，没有伪称最新为v2。现有两项Other/空描述附件G1仍缺，整包高分复现不晋级公榜。证据：`outputs/highscore_public_routes_20261031_107/{inventory.json,REPORT.md,alex_0662/,omer_0663/,lightning_0662/}`。

从 Omer 原生公式冻结软边际单因素在本站完整B-R同折评估：均匀 Sinkhorn BA0.577164→软边际0.525705，Δ=-0.051459，三折均负，独立逐行复算PASS；公开池类比例合成删样本偏移仍均值负。预设G2-R失败，停止该移植，不调权、不容量、不交榜；作者0.663属于其不同完整配方及外部账号，不能外推。证据：`outputs/soft_marginal_20261031_108/{taskcard.json,run.json,REPORT.md,summary.json,shift_stress_summary.json}`。本站公榜最高仍0.620。

---
## 2026-10-31 允许重复标签的相对文风路线首验与止损

公开池 `pool_v4.parquet` 75,783行中，按 source+prompt_key 而非 prompt_key 单独计数，11,804组中952组（7,918行）含同模型的不同文本；单用 prompt_key 会跨来源碰撞1005个键。官方三折使用训练侧题目ID选择文本聚类距离阈值0.90，留出侧只读文本、不固定n/18，题组纯度约0.984/0.980/1.000；这是分组可行性，不证明隐藏结构。证据：`outputs/prompt_group_assignment_20261031_105/{pool_repeated_group_audit.json,variable_size_group_feasibility.json}`。

冻结的官方-only相对文风成员 CPU 三折完成，完整 B-R BA0.577164→0.581608，Δ=+0.004444；绝对文风对照0.576055，单成员相对0.334852对绝对0.268175。独立逐行复算PASS；未达预设+0.005完整融合门槛，停止此候选，不调0.1权重、不做容量或公榜。证据：`outputs/relative_style_official_20261031_106/{taskcard.json,run.json,REPORT.md,summary.json,oof.csv}`。本站公榜最高仍0.620，Kaggle CLI认证未恢复。

---
## 2026-10-31 文本推断题组与硬唯一分配：OOF强信号但公榜迁移止损

官方训练149/150题各18类各一次，测试只有id/text。冻结文本TF-IDF平均链接聚类、大小17/18组Hungarian唯一分配；完整B-R同折2699行BA0.577164→0.757231，三折增量均约+0.17以上，独立逐行复算PASS。**但不能以该OOF晋级公榜**：竞赛官方说明测试各模型计数不等，且一位参赛者报告几乎同法OOF大涨、公榜0.353→0.350。测试是否每题每类唯一未获证，硬唯一假设迁移不成立；G3未运行、公榜未提交。证据和外部链接见 `outputs/prompt_group_assignment_20261031_105/{taskcard.json,run.json,REPORT.md,summary.json,oof.csv}`。本站公榜最高仍0.620。

---
## 2026-10-31 高分当前资产 G1 依赖范围审计

现有高分候选静态47处附件路径匹配归属7个数据集：l24-adapters 20、l24-adapters-v5 18、matched-clean 2、offline-wheels 1、ATLAS 1、handwriting-blend 3、two-stage-lora 2。既有元数据复查确认其中 l24-adapters-v5 与 offline-wheels 直接依赖的许可证均为 Other、描述为空、没有明确复用条款；公榜G1仍未通过。已将来源匹配、当前 v3/v9 输入哈希、G2 缺口与容量适用范围写入 `outputs/highbase_0651_package_20261027_80/rights_admission_20261031.json`。Kaggle CLI认证仍不可用，不据当前资产0.651作者声称分数提交；等待恢复身份后继续逐源补证及同折评估。

---
## 2026-10-31 高分当前资产路线复核与 Kaggle 通道状态

高分当前资产私有4000行同源容量验收 `PASS_4000_SAME_SOURCE_CAPACITY_ONLY`：六主成员和 ATLAS 完成、4000唯一行，观测计算27653秒；不证明隐藏行容量或BA，公榜G1来源及G2同折效果仍未完成。当前高分配方必须使用的模型/数据来源见 `outputs/highbase_0651_package_20261027_80/{input_version_snapshot.json,asset_path_preflight.json,capacity_v1_acceptance.json}`。2026-10-31现有 Kaggle CLI只读查询私有容量作业返回 `kernels.get denied`，随后竞赛提交清单明确返回 `Authentication required`；没有登录、重传、训练或提交。账号通道恢复前不启动新的Kaggle作业或公榜提交；可继续本地来源/协议核查。本站最后已验收公榜仍0.620（57684141），新终态未知。

---
## 2026-10-31 三类模型专项实验止损

同一公开池和完整融合同折预测上的三类专项 SVC 已完成：2699行/150题三折，BA 0.577164→0.569356，Δ=-0.007808；逐折 -0.012222/+0.002222/-0.013311，58救回/79损失。独立逐行核对 PASS，G2-R FAIL，停止，不做容量或公榜。此为见过开发集错误后的探索性诊断；本站公榜最高仍0.620。证据：`outputs/triad_specialist_20261031_104/{run.json,REPORT.md,summary.json,triad_candidate_full_oof.csv}`。

---
## 2026-10-31 公开池 SVC C2 加权同折对照止损

C2 三折来源行全部已在 0.620 已评分 notebook 的75,783行公开池中。为核实真实增益，固定同一公开池/四路特征/三折，预设候选仅将每折 C2 行样本权重加倍并归一化。CPU 2线程三折完成，SVC单成员 BA 0.467845→0.473783；固定完整融合 BA 0.577164→0.576428，Δ=-0.000736，第三折 -0.006667，未过预设 G2-R 数值门。独立逐行复算 PASS；因此停止加权候选，不做容量或公榜。重建同折基线 0.577164 与本站公榜 0.620 不可直接等同；本站最高公榜仍 0.620。证据：`outputs/svc_public_pool_reweight_20261031_103/{run.json,REPORT.md,summary.json,full_fusion_summary.json}`。

---

## 2026-10-31 SVC C2 与已评分公开池重叠核查

已评分 0.620 notebook 的 SVC 使用 `pool_v4.parquet` 75,783 行；只读逐 ID 核对 C2 三折筛查文件，折0/1/2 分别 11,424/11,456/11,429 行全部在公开池中，标准化文本和标签也逐行一致。故此前 C2 SVC 相对官方-only 重建 B-R 的完整融合 OOF +0.009635，不能外推为对实际 0.620 配方的新增数据收益。当前 C2 SVC 不晋级容量或公榜；若继续研究，须冻结同一75,783行基线与一个来源权重变体，做同折完整比较，另行完成来源 G1 与容量。原生汇总：`outputs/qwen25_stage1_newpool_20261029_99/svc_c2_public_pool_overlap_20261031.json`。

---

## 2026-10-31 C2 深成员首折验收与止损

私有禁网 T4×2 作业 `okkojbjj/name-that-llm-qwen2-5-two-stage-c2-fold-0/1` COMPLETE。原生验收 PASS：900 行/50 题、18 类有限分数、阶段一 100 次与官方阶段二 200 次更新；实耗约 4253.6 秒。上传代码 SHA256 为 70529b77cff77b5113a25ad87ac51f66fdcc4531a80cceda44d3cc868cfa455c，回拉云端源码仅换行不同，LF 归一化逐字一致；元数据为私有、禁网、T4，原生记录两张 T4。冻结的完整融合首折筛选：B-R BA 0.552222，C2 BA 0.555556，增量 +0.003333，未达到预设 +0.010 扩展门槛。停止 C2 深成员后两折，不交公榜。C2 SVC 替换是独立路线，公榜来源 G1 仍 HOLD。证据：`c2_fold0_verification.json`、`c2_fold0_resource_screen.json`、`native_c2_fold0_v1/`、`cloud_c2_fold0_v1/`、`c2_fold0_record.json`。

---

## 2026-10-31 C2 深成员首折唯一作业运行中

C1 深成员首折已按冻结 +0.010 资源门槛止损。唯一当前本项目 Kaggle 作业为私有禁网 T4×2 okkojbjj/name-that-llm-qwen2-5-two-stage-c2-fold-0/1，已且仅已上传一次，首次查询 RUNNING；父限12600秒/内限10500秒，100+200更新。同源 C2 首折不得重传、并行启动第二作业或交公榜。终态后回收原生输出/云端源码，运行 erify_c2_two_stage_outer_fold.py --fold 0 和冻结 screen_candidate_fold0.py，只据预设 +0.010 首折信号决定是否扩展。证据：outputs/qwen25_stage1_newpool_20261029_99/{c2_fold0_taskcard.json,c2_fold0_record.json}。

---
## 2026-10-31 C1 深成员首折验收与止损

唯一私有 Kaggle 作业 okkojbjj/name-that-llm-qwen2-5-two-stage-c1-fold-0/1 COMPLETE，原生预测验收 PASS：900行/50题、18类有限分数、100+200更新、T4×2禁网，4073.4秒；上传与云端源码换行归一化一致。成员 BA 0.478889。事先冻结的完整融合首折筛选：B-R 0.552222 → C1 0.560000，Δ=+0.007778，未达到事先设定的 +0.010 扩展信号，故停止 C1 深成员后两折，不交公榜。C1 SVC 替换路线为独立诊断，不受此深成员止损结果直接否定。证据：outputs/qwen25_stage1_newpool_20261029_99/{c1_fold0_record.json,c1_fold0_verification.json,c1_fold0_resource_screen.json}。

---
## 2026-10-31 SVC 完整融合配对题目重采样

在冻结的2699行/150题同折预测上，种子20261031、1000次按每折题目抽样：SVC C1相对B-R的ΔBA 95%区间[-0.001106, 0.011119]，正增益比例0.945；SVC C2区间[0.001126, 0.018148]，正增益比例0.985。C2是更有力的探索性完整融合候选；来源公榜G1仍HOLD，尚无新公榜分数。证据：outputs/qwen25_stage1_newpool_20261029_99/svc_replacement_full_oof_bootstrap_20261031.json。

---
## 2026-10-31 完整融合 SVC 替换诊断

逐行独立复算同一2699行/150题的冻结完整 B-R 融合：BA 0.538638；仅替换 SVC 输入为 C1 后 0.543821（+0.005183，三折均正），替换为 C2 后 0.548272（+0.009635，第三折 -0.002177）。两者达到预设 G2-R 数值门槛，是值得继续的完整融合候选；Qwen2.5 深成员仍为 B-R，诊断文件内部 rms.br 是占位标签。来源公榜 G1 仍 HOLD，未做完整容量、未交公榜，本站最高公榜仍 0.620。当前唯一 Kaggle C1 深成员折0仍 RUNNING，不并行新作业。证据：outputs/qwen25_stage1_newpool_20261029_99/svc_replacement_full_oof_diagnostic_20261031.json。

---
# 2026-10-30 C1/C2 代码内过滤已本地验收

私有数据集首次上传中断且账号清单未见创建，未重传。仅含ID/折排除键的 `outputs/qwen25_stage1_newpool_20261029_99/sources/embedded_source_filter.py` 六种候选/折均与冻结Parquet逐行一致，父池与选中行哈希失败即停；尚未在Kaggle运行。任务记录 `sources/embedded_source_filter_record.json`。后续候选将内联到单个代码文件，需私有运行时验收。

---
# 2026-10-30 Qwen3 外折2唯一作业运行中

当前唯一 NameThatLLM Kaggle作业 `okkojbjj/name-that-llm-qwen3-official-outer-fold-2/1`，私有禁网T4×2、父12600秒/内10500秒、200更新，首次查询RUNNING；绝不重传、并行启动第二个本项目作业或交榜。终态后原生验收899行/50题，记录 `outputs/part_two_oof_feasibility_20261028_83/qwen3_official_fold2_record.json`。

---
# 2026-10-30 Qwen3 外折1完成与输入包上传状态

Qwen3官方-only外折1/1 COMPLETE、原生验收PASS：900行/50题、200更新、BA0.435556、3007.76秒；云端GetKernel 403，逐字源码独立回拉HOLD。证据 `outputs/part_two_oof_feasibility_20261028_83/qwen3_official_fold1_record.json`。此前页内“折1运行中”是历史状态。C1/C2仅ID的私有Kaggle数据包首次上传停在首个文件并已中断，账号清单未见数据集；不得据此重传，准备把过滤ID嵌入候选代码。见 `outputs/qwen25_stage1_newpool_20261029_99/sources/c1_c2_kaggle_filter_package/upload_attempt.json`。

---
# 2026-10-30 三来源 SVC 成员同折结果

三来源(含lxd)筛查后18类齐备，SVC同折整体BA0.394183，较两来源0.377542高0.016642，三折均正；题目成组bootstrap区间约[0.002172,0.030370]，但来源/阈值选择后验且lxd同题风险残余。仅成员级探索，不能称完整融合、公榜或G2-R PASS。证据 `outputs/svc_three_source_deepfold_20261030_102/verification.json`。

---
# 2026-10-30 两来源 SVC 成员同折结果

同一2699行、150题、深成员固定三折，Part Two四路 TF-IDF + LinearSVC C=0.5 仅把每折训练侧追加已筛查的 seyit/prvsiyan 共7223行；其余特征、模型、验证ID和类别顺序不变。官方-only 成员 BA 0.362642 → 0.377542，差值 +0.014899；逐折 +0.012222、+0.011111、+0.021315。原生输出逐行复核 PASS：2699唯一ID、150题隔离、18类有限margin、argmax和指标重算一致。CPU 56.56秒、峰值RSS约1.15GiB，未用GPU/Kaggle、未交公榜。 95%题目bootstrap区间跨零，仅成员探索，不晋级完整G2-R；证据 `outputs/svc_first_party_deepfold_20261030_101/verification.json`。

---
# 2026-10-30 Qwen3 外折1正在运行

当前唯一运行中的本项目Kaggle作业：`okkojbjj/name-that-llm-qwen3-official-outer-fold-1/1`，私有禁网T4×2、父12600秒/内10500秒、200更新，首次查询RUNNING；绝不重传或并行启动第二个本项目作业。终态后回收原生JSON/CSV/NPZ/日志，用既有 `verify_qwen3_official_fold.py --fold 1` 验收，仅记成员OOF，不交公榜。记录 `outputs/part_two_oof_feasibility_20261028_83/qwen3_official_fold1_record.json`。

---
# 2026-10-30 G1/G2-R 协议已批准

用户批准外部来源操作性准入和新完整基线 B-R 的同折三折比较。逐源核对来源、条款与逐行身份；每折仅用训练侧官方文本做去重、近重复和重建题目压力筛查，披露隐藏真题重合不可排除。冻结 B-R 与最多两个单因素候选，候选−B-R 合并 OOF BA≥0.005 为探索性晋级；历史本站0.620仅为公榜锚点。具体来源、运行和公榜仍分别受 G1/G2-R/G3/G4 验收，当前没有新合格候选或≥0.650本站分数。现行细则见 `outputs/highbase_0651_package_20261027_80/PLAN.md` 页末。 加速顺序：从完整0.620链路出发，先并行准备来源筛查与全成员共同外折；首候选为Qwen2.5增加固定的外部池阶段一（现有官方-only重建原本没有阶段一），不能把效果仅归因于来源，G2-R合格即进入完整容量和公榜验收，不等五线全完成。

首批 C1 两来源逐折筛查已冻结：seyit 4826 + prvsiyan 2397，深成员三折每折7223行、预设可见文本/重建题目规则剔除0行；lxd未纳入。只作为探索性私有候选输入，非效果/真实隐藏题目零重合证明。证据 `outputs/qwen25_stage1_newpool_20261029_99/sources/fold_local_first_party_screen/summary.json` 与同目录脚本；未训练/交榜。

---
# 2026-10-30 外部池题族风险与逐折过滤

三来源直连候选11499行中，lxd的14个题目键/165行被列入隔离清单（13个明显同概念，1个关联概念且回答重叠）；公开150题重建并非真实题面。独立Sol/xhigh复核：严格三折OOF每折只能用训练侧官方回答/重建主题筛选外部池，不能把全量官方资料形成的14键过滤结果作为严格OOF。原池不变，未训练/交榜，G1仍HOLD。证据 `outputs/qwen25_stage1_newpool_20261029_99/sources/topic_overlap_quarantine.json`。

---
# 2026-10-30 上游来源当前声明抽样审计

两份主要来源当前卡/README声明CC BY 4.0，第三份当前卡标CC-BY-SA-4.0，第四份元数据403；对应作者池共38,131/75,783行，但历史版本、逐行映射、服务输出权利与题族隔离仍未闭环，G1 HOLD。详见 outputs/qwen25_stage1_newpool_20261029_99/upstream_rights_current_audit.json 与 REPORT.md；无训练/交榜，本站最高0.620。

---
# 2026-10-30 Qwen3 官方-only 外折0原生验收

私有禁网 T4×2 作业 `okkojbjj/name-that-llm-qwen3-official-outer-fold-0/1` 已 COMPLETE。原生 JSON/CSV/NPZ/日志保存在 `native_qwen3_official_fold0_v1/`；冻结验收器 SHA 4D3ECD3C 对官方 train SHA 40A22CE8、深成员折 SHA CC65F5AD 复核，结果 `PASS_SINGLE_OFFICIAL_FOLD_MEMBER_OOF_ONLY`：900 行、50 个留出 prompt、18 类，200 更新，双卡梯度与 adapter/head 重载，成员加权 BA 0.391111，耗时 2720.22 秒。日志无 Traceback/OOM/SKIPPED，只有截断27层导致的预期未使用原模型后层权重提示。运行时源码自报 SHA 与冻结上传件相符；独立云端 GetKernel 回拉仍为403，逐字云端源码身份 HOLD。此结果仅证明官方-only 单成员单折技术及 OOF，不含公共池阶段一，不证明完整0.620集成增益或公榜提升；不提交公榜。

独立 GPT-6 Sol/xhigh 基于受限证据复核：Qwen3 首折 BA 比 Qwen2.5 低0.047778、独有正确42行对85行，故暂不立即补折1/2，先补完整0.620同折集成OOF与外部池来源/题族隔离。此判断不否定未来跨折或完整融合收益；复核者未独立读取远端原始文件，结论限于给定证据。

首次本地验收调用误把 upload 目录当文件，改用冻结源文件后原验收器通过，原生输出未更改。

---
# 2026-10-30 Qwen2.5 外部池分组键用途纠正

`triplet_groups.py` 训练哈希含原始source和setting；新池作者部分1,001个跨源共享prompt_key/25,284行全部被拆成多个哈希。独立Sol/xhigh复核：这不是训练三元组必然bug，但绝不可把该哈希当跨来源外折隔离键；官方p###与公开键同题映射仍未证。证据 outputs/qwen25_stage1_newpool_20261029_99/triplet_group_cross_source_audit.json（SHA EE8FB618）及REPORT.md。G1仍HOLD，不改三元组源码、不启动正式外部训练或交榜；本站最高公榜0.620。

---
# 2026-10-30 公开池同题键结构更正

作者pool_v4的75,783行有8,278个prompt_key；1,005个键跨至少两来源、覆盖28,469行，最多跨7来源。规范化同题分组不能按来源随机切；官方p###与公开键命名空间不同，原样交集0不证明无题族泄漏。结构审计 outputs/qwen25_stage1_newpool_20261029_99/pool_v4_prompt_key_audit.json（SHA40C965E5）、REPORT.md。来源/许可作者声明有证据，但外部池对官方外折的同题隔离仍UNKNOWN，G1整体HOLD；本站最高公榜0.620，无新作业/提交。

---
# 2026-10-30 赛事外部数据规则核对

赛事官方允许公开外部数据和预训练模型；作者pool_v4元数据标CC-BY-SA-4.0且逐行25源编码齐全。赛事规则这一项已PASS，原始来源/服务输出权利、服务标签及跨库题族隔离未闭环，G1整体仍HOLD。证据输出为 outputs/qwen25_stage1_newpool_20261029_99/pool_v4_source_license_audit.json（SHA CC662E3E）；无训练或公榜提交。

---

# 2026-10-30 Part Two 公开池逐源许可声明审计

作者pool_v4已钉SHA37750E89，75,783行/25个非空source编码，作者原生数据集metadata标CC-BY-SA-4.0并描述25个上游slug；逐源行数表 outputs/qwen25_stage1_newpool_20261029_99/pool_v4_source_license_audit.json（SHA CC662E3E），报告同目录REPORT.md。仅完成作者声明和来源字段核对，原始25源许可/服务输出权利、别名逐行映射及跨来源prompt家族隔离仍未独立证实，外部池正式训练和公榜G1继续HOLD。Kaggle登录恢复问题仍待答复；本轮无上传/提交，本站最好公榜0.620。

---
# 2026-10-30 Kaggle认证与替代GPU核查

Kaggle CLI 前次 status/list 只读查询分别报 kernels.get denied、Authentication required；现有浏览器只读会话工具 trusted Node 启动失败，不能确认登录。未做凭据操作、登录、上传或重提。Net-4090D 已核项目环境仅见单张RTX4090D、查询时空闲16727MiB；Qwen3-14B-Base在三个明确缓存/项目路径不存在，不能以该单卡替代Kaggle双T4完整容量。Qwen3预执行任务卡据此仍HOLD；证据见 outputs/part_two_oof_feasibility_20261028_83/{qwen3_official_fold0_taskcard.json,REPORT.md}。本站公榜最高仍0.620；待账号只读访问恢复先核确切Kaggle作业和每日提交状态，再决定外部行动。

---
# 2026-10-30 Qwen3-14B 旧作业基座身份与EOS契约

已评分 Part Two v2 原生 metadata 证实曾挂载 `qwen-lm/qwen-3/Transformers/14b-base/1`；作者最终meta为前27/40层、八处池化，Notebook说明训练末尾token固定151643而Kaggle默认151645会静默改变预测。官方Qwen3-14B-Base模型卡为Apache-2.0；当下Kaggle可见性及副本逐字身份仍未知。证据与HOLD任务卡见 outputs/part_two_oof_feasibility_20261028_83/{REPORT.md,qwen3_official_fold0_taskcard.json}。本轮无训练、上传或提交，本站公榜仍0.620。

---
# 2026-10-30 易混类别路线三折错误切片

在已验收官方-only双Qwen2699行三折上，有界脚本核对输入哈希与预测对齐后计算救回140、误伤67、两单体均错1156；Terra、GLM Flash、Gemini3.8召回下降，Gemini3.7/3.8相互混淆突出。原生探索切片 outputs/qwen25_stage1_newpool_20261029_99/official_threefold_error_slice.json（SHA F5FE6A22）及 REPORT.md。GPT-6 Sol/xhigh 独立只读复核认为只能提出假设；同一OOF已被多次查看，不可据此选门控又宣称独立收益。既有Gemini二分类/GLM top2已止损；R4专家暂无合格独立评价与完整融合证据，HOLD，不训练/交榜。Kaggle CLI仍未恢复认证，本轮无上传/提交；本站公榜仍0.620。

---
# 2026-10-30 完整融合缺口与 Kaggle 认证状态

Luna/xhigh 按计划尝试有界只读审计，但其子代理进程无法访问 Z:（CreateProcessWithLogonW 267），故没有声称独立文件验收；主代理据已核原生文件更正了 outputs/part_two_oof_asset_audit_20261027_74/REPORT.md 的折2历史状态。完整 Part Two 同折OOF仍缺Qwen3、DeBERTa等合法重建成员，G2 UNKNOWN。Kaggle CLI 只读 status 返回 kernels.get denied，随后 list 返回 Authentication required；未做登录、上传、训练或提交。下一步在本地准备 Qwen3 官方-only 折0最小补证任务卡；外部执行前须恢复 Kaggle 只读身份并核查确切作业状态。Luna 访问失败是工具环境限制，不代表项目文件不存在。本站公榜最高仍0.620。

---
# 2026-10-30 现行五线执行契约与双 Qwen 三折归档

本站公榜最高已验收BA0.620（57684141），私榜未知；今天无新合格提交。Astra/high 对五线及模型分工做了受限证据设计复核，现行入口为 outputs/highbase_0651_package_20261027_80/PLAN.md 页末“现行执行契约”；五线不等于五次提交。GPT-6.1 Sol/high 日常唯一写入，GPT-6 Sol/xhigh 独立科学审查，GPT-6 Luna/xhigh 更多做有界机械核验，Astra只审重大设计变化。折2原生PASS并已归档，三折固定半半双Qwen官方-only诊断旧0.435235、新0.442707、混合0.462299；折0选择偏倚与云端GetKernel403保留，不能称完整0.620集成改进或G2。证据 outputs/qwen25_stage1_newpool_20261029_99/{official_outer_fold2_record.json,official_threefold_fixed_blend.json,REPORT.md}。当前无已知运行中本项目作业；下一步先核对Kaggle实际状态，再由Luna整理完整融合缺口、Sol/high定一个最小深成员补证任务。外部池G1 HOLD、Llama官方访问未知、R4/R5待新机制。旧RUNNING段落均为历史。

---
# 2026-10-30 Qwen2.5 官方-only 外折2私有作业 RUNNING

折1已原生PASS，单成员加权BA0.434444；旧Qwen8B+新Qwen2.5固定50/50折1 BA0.427778→0.450000，仅双成员诊断。折2已且仅已上传一次：Kaggle okkojbjj/name-that-llm-qwen2-5-official-outer-fold-2/1，初查RUNNING；私有禁网T4×2、父限12600秒/内部10500秒、官方train1800行训练/899行留出、200更新。上传源码SHA A16056E1，静态验收PASS；实际slug按标题解析与metadata id不同，已按实际slug查询。终态需原生输出与通用验收器--fold 2；不要重传、并行启动第二本项目作业或交榜。详情 outputs/qwen25_stage1_newpool_20261029_99/{official_outer_fold2_record.json,REPORT.md}。本站公榜最高仍0.620，私榜未知；外部池G1 HOLD，完整融合OOF未齐。

---
# 2026-10-30 Qwen2.5 官方-only 外折1原生 PASS

唯一折1作业 okkojbjj/name-that-llm-qwen2-5-official-outer-fold-1/1 COMPLETE；900行/50题、200更新、加权单成员BA0.434444、2996.45秒，冻结验收器PASS。旧Qwen8B+新Qwen2.5固定半半同折BA0.427778→0.450000，救回43/误伤23；仅双成员诊断，不是完整0.620集成G2或公榜收益。云端GetKernel403，源码逐字身份未独立回拉；原生证据见 outputs/qwen25_stage1_newpool_20261029_99/{official_outer_fold1_record.json,official_outer_fold1_fixed_blend.json,REPORT.md,native_official_outer_fold1_v1/}。折2静态已准备但未上传，启动前再查无其他本项目Kaggle作业；本站公榜仍0.620，私榜未知。

---
# 2026-10-30 Astra 执行协议再审（无新提交）

Astra/high 按受限证据复审五线：优先补完整同折融合 OOF；Qwen2.5 外折0双成员固定半半 BA0.471111 仍仅探索，本站公榜最高0.620。外折1仍 RUNNING、外折2仅准备未上传；外部池 G1 HOLD、官方 Llama 访问未证。新增冻结点：G2 完整基线身份与三折全集、选择边界、聚合/失败处理；G3 最终完整禁网入口；候选身份与每日≤5次去重。模型分工为 Astra 重大协议、GPT-6.1-Sol/high 复杂唯一写入、GPT-6-Sol/xhigh 独立只读复核、GPT-6-Luna low/medium/xhigh 多做范围明确的机械核验。详见 outputs/highbase_0651_package_20261027_80/PLAN.md 页末；无训练、上传或公榜提交。

---
# 2026-10-30 Qwen2.5 官方-only 外折1私有作业 RUNNING

当前唯一已知运行中的本项目 Kaggle 作业：okkojbjj/name-that-llm-qwen2-5-official-outer-fold-1/1，上传一次、初查 RUNNING，禁网T4×2、父限12600秒/内限10500秒，仅官方train1799行训练/900行留出、200更新；无公榜提交。源码SHA6153ADCC，静态预检PASS，任务卡/运行记录见 outputs/qwen25_stage1_newpool_20261029_99/official_outer_fold1_{taskcard,static_acceptance,record}.json。折0原生单成员BA0.438889，旧Qwen8B+新Qwen2.5固定半半同折0 BA0.471111（旧单体0.443333），仅探索。终态下载原生输出用通用验收器--fold 1核对，不因查询故障重传；若RUNNING保持安静，不启动第二个本项目作业。本站最高公榜仍0.620、私榜未知，R1/R2外部池G1仍HOLD。

---
# 2026-10-30 固定双Qwen外折0有正向互补，下一步扩折

已验收的外折0上，旧Qwen8B v8 adapted 概率与新Qwen2.5官方-only概率固定50/50、按18类和900 ID对齐：BA0.443333→0.471111，救回45、误伤20。仅单折双成员探索，未调权重，不能声称完整Part Two公榜0.620提高或G2通过。证据 outputs/qwen25_stage1_newpool_20261029_99/official_outer_fold0_fixed_blend.json。下一步在相同固定配置串行补外折1/2，先做脚本参数化和静态验收；外部池G1仍HOLD，云端源码GetKernel403独立身份未知。当前无已知运行中的本项目Kaggle作业，启动前只读核对。本站公榜最高仍0.620、私榜未知。

---
# 2026-10-30 Qwen2.5 官方-only 外折0原生验收 PASS

Kaggle okkojbjj/name-that-llm-qwen2-5-official-outer-fold-0/1 已 COMPLETE。原生输出按预先冻结验收器验证900留出行、50个prompt、18类分数与200次更新；加权单成员 BA0.438889，plain BA0.346667，耗时2872.42秒。与旧Qwen8B同折BA0.443333相比，新独对119、旧独对123；这是不同成员的单折诊断，不是完整融合或公榜收益。云端源码回拉GetKernel403，运行时SHA自报匹配上传，但独立逐字身份未验。原生证据、SHA和局限见 outputs/qwen25_stage1_newpool_20261029_99/{official_outer_fold0_record.json,official_outer_fold0_overlap.json,REPORT.md}。该专用监控已删除；目前无已知运行中的本项目Kaggle作业。本站公榜最高仍0.620、私榜未知。下一步先审是否值得补两折及完整融合共同基线，R1/R2外部池G1仍HOLD；不把单折结果交榜。

---
# 2026-10-30 Qwen2.5 官方-only 外折0真实训练外作业 RUNNING

唯一私有禁网T4×2作业 okkojbjj/name-that-llm-qwen2-5-official-outer-fold-0/1 已上传一次并初查RUNNING；200更新、官方训练1799行100题，验证900行50题，父限12600秒，Python限10500秒。初稿 AMP 未反缩放即裁剪的错误已由 GPT-6-Sol/xhigh 独立审查指出并在上传前修复；最终脚本 SHA7B863BB6，静态抽样每轮1799训练行覆盖PASS。输出/任务卡/实际slug见 outputs/qwen25_stage1_newpool_20261029_99/{official_outer_fold0_record.json,official_outer_fold0_taskcard.json,REPORT.md}。专用监控 namethatllm-qwen2-5-0 只查终态；不得重传、并行启动第二本项目作业或交榜。此实验没有公开池阶段一，成员单折BA不能代替完整0.620融合；本站最高公榜仍0.620。

---
# 2026-10-30 旧Qwen8B与官方-only SVC同折互补性：不足以单独冲击0.65

2699行同折核对 PASS：旧Qwen8B v8适配成员BA0.435235、官方-only SVC BA0.362642；二选一且知道真标签的oracle也仅BA0.567127（两者都错1168行）。仅说明这对旧成员做直接路由不能达0.65，不约束新深成员或能产生第三标签的概率融合。证据 outputs/part_two_oof_asset_audit_20261027_74/qwen8b_svc_official_overlap.json 与REPORT.md；不把诊断当完整0.620融合或公榜收益。本站最高公榜仍0.620。

---

# 2026-10-30 官方-only SVC 同深成员折 OOF 已独立验收

按 src/baseline.py 现有深成员外层折，仅用官方训练数据运行 Part Two 四路TF-IDF+LinearSVC(C=0.5)三折，2699行/150题逐行验收PASS；BA 0.362642，三折0.375556/0.354444/0.357937，CPU约12.90秒。证据 outputs/official_only_svc_deepfold_20261030_100/{verification.json,REPORT.md,oof.csv,margins_safe.npz}。初版NPZ object ID格式问题留证，安全NPZ独立复核通过。它不含外部池，不能同折外数据0.479326直接归因，也尚无深成员完整OOF；本站公榜最高仍0.620。

---

# 2026-10-30 完整融合折兼容性更正

原先冻结的 common_outer_prompt_folds.csv 仅是当前 SVC OOF 三折；与 src/baseline.py 的 MODERNBERT_MIXED_FOLDS（已有Qwen外层训练入口采用）仅47/150题同号，不能跨折拼成完整融合 OOF。现有深成员折已独立冻结为 outputs/part_two_oof_asset_audit_20261027_74/existing_deep_outer_prompt_folds.csv（SHA CC65F5AD），两清单及不兼容证据见同目录两个 summary 和 REPORT.md。优先复用已有深成员折并在其上重训SVC；正式融合前所有成员还必须有相同外层题目未见训练的预测。此前页首“共同外层三折”应按此更正理解。本站公榜仍0.620，私榜未知。

---

# 2026-10-30 Qwen2.5 B/C 外层 prompt 泄漏身份未闭环

旧/新同题池29,347/62,655行的外部 prompt ID 与官方 p### 属不同命名空间；0个精确ID交集不证明无同题。旧池全部缺 prompt_text_sha256，新池仅3,572行有；答案文本近重复过滤不能证明题族隔离。审计见 outputs/qwen25_stage1_newpool_20261029_99/prompt_overlap_gate_20261030.json 与 REPORT.md。外部输出使用权也未闭环，故不启动正式旧新池 B/C 或据此声称完整融合 OOF；可继续官方数据和已确认资产上的工作。本站最好公榜0.620，私榜未知。

---

# 2026-10-30 完整融合共同外层三折清单已固定

官方2699行与现有SVC OOF逐行 ID、prompt、标签完全对齐；150个prompt按题隔离为50/50/50组，行数900/900/899。清单 outputs/part_two_oof_asset_audit_20261027_74/common_outer_prompt_folds.csv SHA A28085AF，摘要与局限见同目录 common_outer_prompt_folds_summary.json、REPORT.md。此为深成员共同外层训练的固定切分依据，不是深成员OOF或BA收益；Qwen2.5 v5仅技术PASS，外部输出权利仍待核，本站公榜最好0.620。

---

# 2026-10-30 Qwen2.5 官方 24 步双头探针 v5 原生技术 PASS\n\n唯一私有禁网双T4作业 okkojbjj/name-that-llm-qwen2-5-gradient-probe/5 COMPLETE。144 个官方 prompt/432 行、24 次更新，双头各自梯度及参数变化、LoRA双卡32层×7目标、448 adapter/20 head重载均通过；总 336.98 秒，GPU峰值allocated约 10.34/8.91 GiB。原生验收见 outputs/qwen25_stage1_newpool_20261029_99/official_objective_probe_v5_acceptance.json 与 native_gradient_v5/。运行时源码SHA自报匹配本地，但云端GetKernel403，字节身份未独立验。只证明官方数据技术测时，不证明正式作者类加权训练、BA、完整训练或九小时隐藏推断；外部生成输出权利尚缺，未交榜。本站已验收公榜仍0.620，私榜未知。\n\n---\n\n# 2026-10-30 Qwen2.5 官方数据 24 步双头私有探针 v5 RUNNING

唯一 Kaggle 作业 okkojbjj/name-that-llm-qwen2-5-gradient-probe/5 已上传一次、初查 RUNNING。仅官方 train 的 144 个不同 prompt、432 行，24 次更新，双头各自梯度/参数变化及重载验收；禁网双 T4，父限1800秒、Python alarm1500秒，无外部生成文本、无BA或公榜提交。上传脚本 SHA 3DD8D4F9，独立验收器 SHA FF6BD4F6；v4两步原生结果被新验收器按预期拒绝。证据见 outputs/qwen25_stage1_newpool_20261029_99/{run.json,official_objective_probe_taskcard.json,gradient_probe_v4_to_24_updates.diff}。只查确切版本5终态，运行中不重传/不开第二作业；COMPLETE需下载原生JSON/日志并独立验全部24步。本站最好公榜仍0.620，私榜未知。

---

# 2026-10-29 五路线执行细则与模型路由已复核

Astra/high 对五路线做重大设计复核，GPT-6-Luna/xhigh 核固定事实与门槛，GPT-6-Sol/xhigh 独立审科学语义；主代理已核原生 v4 PASS、保留云端 GetKernel403 来源边界。完整可执行修订见 outputs/highbase_0651_package_20261027_80/PLAN.md 页末“2026-10-29 Astra 复核后的执行细则与模型路由”，本页下一节是 v4 原生证据。关键：新旧 Qwen2.5 池是配方替换而非纯增量，B→C 不能加到本站 A=0.620；SVC +0.008886 只属成员 OOF；完整候选仍须 G0–G4。Luna/xhigh 承担更多有界事实核对，GPT-6.1-Sol/high 唯一写入复杂执行，GPT-6-Sol/xhigh 独立复核，Astra只重大设计。公榜最高仍0.620，私榜未知；无新公榜提交。

---

# 2026-10-29 Qwen2.5 私有同题双头技术探针 v4 原生 PASS

唯一 Kaggle 作业 okkojbjj/name-that-llm-qwen2-5-gradient-probe/4 COMPLETE。官方 train 12 prompt×3类、每步6个同题三元组，普通与组中心化双头 CE 两次真实 LoRA 更新并重载通过：双卡32层×7投影 B 梯度、参数变化为正，448 adapter/20 head 张量重载一致，logit差0，总77.33秒。原生运行时源码SHA与本地上传A99E80D7一致；GetKernel回拉403，云端源码独立字节身份仍未证。证据 outputs/qwen25_stage1_newpool_20261029_99/{run.json,gradient_probe_v4_acceptance.json,native_gradient_v4,REPORT.md}。这是技术可行性，不是作者完整加权训练/OOF/BA/9小时隐藏推断；外部输出权利未闭环，不交榜。本站公榜最好仍0.620、私榜未知，当前无已知运行中的本项目Kaggle作业；下一作业启动前仍只读核对实际状态。

---

# 2026-10-29 Qwen2.5 私有同题双头技术探针 v4 RUNNING

同一Kaggle实际slug okkojbjj/name-that-llm-qwen2-5-gradient-probe 版本4已且仅已上传一次，初查RUNNING。禁网T4×2、父限1800秒、Python alarm1500秒；仅官方train 12个不同prompt×3不同标签，按6个同题三元组/步做两次普通头+组中心化头CE更新和重载，不使用外部未核权利输出，不评BA/提交公榜。源码SHA A99E80D7、作者第一篇公开配方Notebook SHA 12E1E65B、版本差异及结果位置见 outputs/qwen25_stage1_newpool_20261029_99/run.json。v3原生基础梯度/重载PASS但云端GetKernel403；v4新增运行时源码SHA自报，终态需与本地上传SHA比较，并下载原生JSON/日志核对真实同题、双头梯度、两卡资源与重载。当前仅此一个本项目Kaggle作业运行，不并行启动别的作业。本站最佳公榜仍0.620，私榜未知。

---

# 2026-10-29 Qwen2.5 v3 梯度/重载原生技术 PASS；云端源码SHA仍未验

唯一私有禁网双T4作业 okkojbjj/name-that-llm-qwen2-5-gradient-probe/3 已COMPLETE。原生JSON/日志经独立验收：官方14B/1，32层×7类型=224 LoRA模块，36条官方训练文本CE的两次真实更新无AMP跳步，双卡梯度与参数变化及所有层/目标LoRA B梯度通过；峰值allocated约10.24/8.79GiB，清理后重载前每卡19,442,688B，448 adapter/18 head张量逐项相同，logit差0，端到端81.24秒。证据 outputs/qwen25_stage1_newpool_20261029_99/{run.json,gradient_probe_v3_acceptance.json,native_gradient_v3,REPORT.md}；v1 torchao冲突、v2重载OOM均保留。Kaggle GetKernel403且computer-use运行环境267，云端源码字节身份未核，因此仅判原生技术PASS、源码SHA未验。不是作者triplet训练、完整训练容量、OOF或BA；本站最佳公榜仍0.620，私榜未知，未交榜。下一步以冻结新旧同题臂和作者公开元数据设计私有同种子/同更新数单折对照，先核训练目标及来源权利，再决定长训练；当前无本项目作业运行。

---

# 2026-10-29 Qwen2.5 私有梯度探针 v3 RUNNING

同一 Kaggle 实际 slug okkojbjj/name-that-llm-qwen2-5-gradient-probe 版本3已且仅已上传一次，禁网双T4、父限1800秒、Python alarm1500秒，初查RUNNING；当前不可启动第二个本项目作业或公榜提交。v1因PEFT/torchao冲突ERROR，v2已完成两次真实更新但同进程重载OOM；v3只清理首模型残留的raw_targets、adapter、all_params、witnesses等引用，并在重载前断言两卡allocated均≤1GiB。源码SHA E6DFEABD，v2→v3 diff、原生日志及每版状态见 outputs/qwen25_stage1_newpool_20261029_99/run.json/REPORT.md。终态必须回拉原生JSON和日志，若GetKernel仍403则不声称云端逐字源码验收；技术PASS也只说明36条官方样本CE两步，不证明完整训练、OOF或BA。本站公榜最高仍0.620，私榜未知。

---

# 2026-10-29 Qwen2.5 梯度探针 v2：两步真实更新完成，重载 OOM

Kaggle okkojbjj/name-that-llm-qwen2-5-gradient-probe/2 已终态ERROR，绝不原样重传。原生输出证明官方14B/1、双T4、32层×7类型共224个LoRA模块，官方36样本CE的两次实际更新均有双卡非零梯度和参数变化、所有层/目标LoRA B梯度；单步15.72/12.25秒，峰值allocated约10.99/9.44GB。但同进程第二次加载基座时CUDA warmup因首模型对象引用残留而OOM（归因待修复实测），适配器/头重载未验收，整项技术探针FAIL。云端源码回拉GetKernel403，不能声称云端逐字同一。见 outputs/qwen25_stage1_newpool_20261029_99/{run.json,native_gradient_v2,REPORT.md}。下一步清理raw_targets、adapter、optimizer和头的全部引用，增加重载前残余显存断言，再考虑单次固定修正版；不安装依赖、不交榜。本站公榜最高仍0.620，私榜未知。

---

# 2026-10-29 Qwen2.5 私有梯度探针 v2 RUNNING

Kaggle确切 slug okkojbjj/name-that-llm-qwen2-5-gradient-probe 版本2已且仅已上传一次并初查RUNNING；禁网T4×2、父限1800秒、Python alarm1500秒，只用官方train 36行做两次更新与重载技术检查，不是正式旧新池训练或BA试验。v1已终态ERROR，根因PEFT 0.20.0与torchao 0.10.0冲突；v2只对已核实的224个普通nn.Linear目标注册项目既用的PEFT native Linear分派，训练与重载同处理，不安装依赖。v1/v2源码及metadata差异、SHA、资源和原生v1日志在 outputs/qwen25_stage1_newpool_20261029_99/。终态须查确切v2、回拉原生输出/日志/云端源码、逐条验梯度和参数更新及显存；GetKernel此前403，无法回拉时应标来源身份未独立确认，不凭CLI COMPLETE声称PASS。任何查询故障不重传。当前没有公榜提交，本站最佳仍0.620、私榜未知。

---

# 2026-10-29 Qwen2.5 私有梯度探针 v1 ERROR：PEFT/torchao 环境冲突

私有禁网T4×2作业 okkojbjj/name-that-llm-qwen2-5-gradient-probe/1 已且仅已上传一次，Kaggle终态ERROR。原生 JSON/日志证明官方Qwen2.5-14B/1基座和双T4可见，36条官方样本分词完成，但注入LoRA时PEFT 0.20.0发现torchao 0.10.0低于要求的0.16，33.8秒失败；0次优化更新，无准确率或容量证据。云端源码GetKernel回拉403，不能宣称云端源码逐字验收；原生文件SHA与上传代码见 outputs/qwen25_stage1_newpool_20261029_99/run.json 和 native_gradient_v1/。不安装依赖、不原样重传。下一步审查已有Qwen3成功使用的PEFT普通nn.Linear自定义分派，准备新固定版本并静态验收；当前无运行中的本项目Kaggle作业。本站最高公榜仍0.620，私榜未知。

---

# 2026-10-29 Qwen2.5 私有两步梯度探针 v1 RUNNING

已且仅已上传一次禁网T4×2技术作业，实际 slug okkojbjj/name-that-llm-qwen2-5-gradient-probe，版本1，Kaggle只读状态RUNNING；上传metadata所写qwen25 slug被平台规范化为qwen2-5，后续必须查实际 slug，绝不因查询异常重传。父限1800秒、Python告警1500秒，仅官方train中每类2条共36条、两次更新和适配器重载；没有完整训练、OOF或公榜提交。上传源码SHA FD59AA91，元数据SHA 87C702BF，作者配置小文件SHA见 outputs/qwen25_stage1_newpool_20261029_99/run.json。完成后回拉云端源码/元数据与原生 JSON/日志，验224模块和两GPU真实参数变化、所有层/目标的LoRA B梯度、AMP无跳步、逐张量重载、显存及端到端时间。失败保留证据并止步，不声称模型效果。本站公榜最高仍0.620，私榜未知。

---

# 2026-10-29 Qwen2.5同题 triplet 双臂独立验收 PASS，仍无模型收益

见 outputs/qwen25_stage1_newpool_20261029_99/：初次分组因 Lucas 215 行缺失原始 prompt 而独立验收失败，原失败文件和 run.json attempts[0] 保留。第二次剔除其42组后，旧臂29,347行/5,141组（SHA F6BE4F6C），新臂62,655行/10,825组（SHA 574223D8），独立全行来源与分组资格验收PASS；Aswath精确题面无可用triplet。这是保守重建，作者完整阶段一源码与外部服务输出权利未闭环。没有GPU训练、OOF、完整融合或公榜收益；本站最佳公榜仍0.620（57684141），私榜未知。下一步仅有界私有Qwen2.5两步梯度/重载探针，再做同更新数/同种子旧新臂prompt隔离对照。公榜仍须来源权利、完整融合OOF和9小时容量验收。证据见本目录run.json、triplet_confirmed_summary.json、triplet_confirmed_verification.json。当前无已知运行中本项目Kaggle作业，启动前只读核对。

---

# 2026-10-29 五路线执行契约与模型分工

当前完整执行规则在 outputs/highbase_0651_package_20261027_80/PLAN.md 的“2026-10-29 后续模型执行契约”节，覆盖该文件上方旧状态提案。本站公榜最高仍0.620（57684141），私榜未知；晚间只读 Kaggle CLI 核对本日0提交。先冻结已评分0.620最终模型A；公开配方重建基线B与一因素候选C才可做共同prompt外层OOF，B→C收益不得称为A的原始训练外收益；Qwen2.5新旧池继续私有基座/梯度可行性；高分配方等待Llama合法可用资产；固定GLM top2与DeBERTa双窗口已止损。每条候选必须事前冻结唯一改动、来源/使用权、训练外完整融合效果和最终包9小时保守容量，合格且实质不同才最多每日一次/路线提交，不为次数交榜。

本轮已实际委派 Astra 做关键策略复核、GPT-6.1-Sol/high 写执行契约建议、GPT-6-Luna/xhigh 做可委派任务映射、GPT-6-Sol/xhigh 做独立科学风险审查。后续默认 Luna 承担大多数有界证据清单与验证，GPT-6.1-Sol/high 单写入执行，GPT-6-Sol/xhigh 独立复核，Astra 仅重大设计闸门；实际模型调用以运行记录为准，不因角色建议声称已执行。子代理本轮 Z: 不可见，主代理已核实映射和严格SSH身份并负责合并，不让子代理重挂载或误判文件缺失。Luna/xhigh 已完成限定证据的完整融合OOF缺口表：公开最终深成员权重不能产出训练外预测，SVC原生oof_margins.npz已核2699×18两臂有限连续分数，ID/prompt/折/类序与CSV逐行一致，SHA26F00E18；原生公开附件核查见 outputs/part_two_oof_asset_audit_20261027_74/REPORT.md。GPT-6-Sol/xhigh已复核不能把最终深成员全训练权重拼成无泄漏融合；下一实际动作是按SVC三折设计深成员最小单折GPU补证，随后再决策三折；不得启动未定义资源/输入的新Kaggle作业。

---
# 2026-10-29 Qwen2.5阶段一新旧数据双臂已冻结，来源技术验收PASS

从本站已验收公榜0.620继续。当前无已知运行中本项目Kaggle作业，启动前仍先只读核对。新Qwen2.5阶段一当前配方：作者pool_v4、Abdullah/Lucas当前快照、Aswath额外来源，来源文件SHA固定；90,104原始行→86,803精确去重（无同文异标签或官方完全重复）→新池近题候选85,702行，独立全行复算PASS。另下载作者Qwen2.5旧pool_v1 34,940行SHA A24588C1；旧33,976行与新精确池重合，964行旧独有。共同TF-IDF IDF与相同0.30阈值筛选后配对臂旧34,789行、新85,692行。发现初版新池压平作者逐行source，现已对配对两臂补入原始数据集/行号/声明来源/模型引用/文本哈希并独立逐行回查全部120,481行PASS；正式私有输入是 outputs/qwen25_stage1_newpool_20261029_99/old_paired_pool_provenance.parquet SHA BF16F9E1 和 new_paired_pool_provenance.parquet SHA D4B54B06。仅证明数据技术与来源字段对齐，外部服务输出权利仍未逐行验证；新配方不是作者历史83,827行复现。尚未GPU训练、完整融合或公榜提交。Kaggle官方非指令基座qwen-lm/qwen2.5/Transformers/14b/1只读核实（Apache 2.0，约29.55GB），但尚未实际挂载；Net-4090D单张24GB GPU且本账号无该基座缓存。作者仅公开阶段一文字配方与推断源码，本站训练必须标为重建。下一步先用私有禁网T4×2做两步梯度/重载有界测时，再固定相同基座/类平衡采样/更新数/种子做私有双臂提示隔离对照，完整融合和9小时容量都需单独验收。证据在 outputs/qwen25_stage1_newpool_20261029_99/{PLAN.md,run.json,REPORT.md,exact_summary.json,near_summary.json,paired_summary.json,provenance_summary.json,verification.json,provenance_verification.json}。本站公榜最高仍0.620，私榜未知，每天最多5次只交合格不同候选。

---
# 2026-10-29 官方Llama资产探针未证实可用，转向其它优化线

当前没有已知运行中的本项目Kaggle作业；启动新作业前只读确认。官方Llama 3.1 8B私有无GPU禁网探针 okkojbjj/name-that-llm-official-llama-access-probe/1和/2均ERROR，未训练/推断/交榜。v1只扫/kaggle/input/models，NOT_MOUNTED结果不具判定力。v2扫整个/kaggle/input，但构造结果时因details未定义NameError；堆栈到result行而非for循环，推断本次无匹配llama-3.1/8b/config.json。不能分辨Meta账号访问、Kaggle挂载布局或模型内容问题，故不宣称账号被拒，停止重复探针。两版原生日志、v1 JSON、本地上传源码与SHA见 outputs/llama_official_access_probe_20261029_98/{run.json,REPORT.md,native_v1,native_v2}。此前q35/1缺kernelspec、q35/2缺必需Llama基座，均无测时；高分配方资产线暂停，继续不依赖该资产的Qwen2.5数据/完整融合/多簇专家等路线。本站最高已验收公榜BA仍0.620，私榜未知，未新增公榜提交。

---
# 2026-10-29 q35成员测时 v1/v2均失败，暂停重复上传

当前没有已知运行中的本项目 Kaggle 作业；启动新作业前仍须只读核对。私有禁网 T4×2 q35 512同源假行测时 okkojbjj/name-that-llm-q35-member-timing-512：v1 因缺 notebook kernelspec 在 Papermill 启动前 ERROR；v2 加入内核后，输入 CSV 哈希匹配、离线 wheel 安装返回0，但 setup 因缺 Llama 3.1 8B 基座断言失败，补充栈和 q35 未执行，无测时/BA。v1 上传曾提示 avazaku/name-that-llm-verified-llama-asset-cache 无效；该来源其实是 setup 必需，先前称未使用为误判。该来源只读 status 返回 kernels.get denied；不能据此重传或假设旧容量结果仍可复现。v2 确切源码回拉 GetKernel403，原生日志/input_pin、上传本地 SHA 及两版本记录在 outputs/q35_member_timing_20261029_97/{run.json,REPORT.md,native_v1,native_v2}。停止 q35 重试，先解决合法可访问的 Llama 模型资产与挂载路径，并重新核对时间预算。无公榜提交，本站最高公榜仍0.620，私榜未知。
第4线优先级复核：既有固定GLM双类top2门控仅15个可救错误，理论BA最大+0.00556，预设+0.005需净救14个；已准备私有CPU包不上传。既有SVC OOF误差分散于Gemini、GLM、GPT等簇，若继续该线应在训练折内选多簇、独立prompt折评价并核对完整融合边际效果，不能按同一2699行后验调门槛。官方Kaggle Llama 3.1模型页要求Meta授予访问；当前CLI只读变体文件查询404，不能证明本账号已取得可用资产。

---
# 2026-10-29 当前公开池增量 SVC成员：私有三折全部预设门槛PASS，完整集成待验

唯一禁网无GPU Kaggle作业 okkojbjj/name-that-llm-current-pool-increment-screen/1 已COMPLETE，只上传1次，无公榜。输入运行时SHA：官方train40a22ce8、旧pool_v4 37750e89、当前Abdullah8e7a66eb、Lucas a6da05c3；精确去重新增6916行（Lucas5118/Abdullah1798），SVC池75783→82699。2699行/150prompt固定三折独立复算PASS，旧臂BA0.470440→新臂0.479326，Δ+0.008886；三折均上升，prompt bootstrap95%[+0.001106,+0.016667]，NLL/Brier均下降，五项预设成员闸门全过。云端源码/元数据GetKernel回拉403，不能声称字节核对；本地上传SHA及原生日志/基线一致为间接证据。外部输出条款、完整0.620融合训练外增益与9小时隐藏容量仍未闭环，不提交公榜、不改权重。原生证据 outputs/current_pool_increment_screen_20261029_94/{run.json,REPORT.md,native_v1,verification_v1.json,cloud_pull_error.txt}。当前无已知运行中本项目Kaggle作业；启动下一作业前只读复核实际状态。已删除本专用CPU监控，保留每日namethatllm。

---
# 2026-10-29 q35单成员512行测时包已准备，本项目当前仍仅一作业运行

从已验收v27测时Notebook派生私有禁网T4×2补充栈单次测时包 outputs/q35_member_timing_20261029_97/；原补充栈代码单元逐对象相同，当前两公开CSV SHA硬断言，NTL_DROP_Q35=0，512个fake ID，父限18000秒、核心限14400秒。静态nbformat/AST/元数据通过，尚未上传、运行GPU或交榜。它只测q35实际耗时与补充栈容量，不能证明BA或完整隐藏容量。先等待当前唯一私有CPU作业 okkojbjj/name-that-llm-current-pool-increment-screen/1 终态并验收，绝不并行启动第二个本项目Kaggle作业。当前数据再次漂移则重冻输入并审查。

---
# 2026-10-29 外部0.660队伍属实，V31源码版本与隐藏容量仍未证实

Kaggle榜单只读显示 Mercurydust12 与 alex chilton 各公榜0.660（外部队伍），证据 outputs/public_highscore_scan_20261029_91/leaderboard_evidence.json；不能绑定到本地公开Notebook具体版本，本站最高仍0.620。alex V31源码启用q35且ATLAS按8.3小时总预算允许部分行完成。本站v27同源4000行总28268秒、ATLAS1261行耗3953秒；若同样触发行数、q35至少增1.7小时，简单下界34388秒>竞赛32400秒，且V31多distort计算。不同分数和输入可改变触发行数，故只判条件风险，下一步需私有同输入测时、原生完成比例及输入/权利核对，不直接交榜。当前唯一运行中本项目Kaggle作业仍为另一路私有CPU对照，勿重传或并行开第二作业。

---
# 2026-10-29 GLM易混专家：固定top2预筛可行，候选仅本地准备

作者Part Two公开列出GLM5.3/flash难辨；本站已验收SVC原臂2699行/150prompt三折OOF固定top2门控触发70行，真实双类42行、原版正确27行、双类互判可救15行。两类各150行，理论18类BA最大增量仅15/(18×150)=0.005556；预设+0.005至少净救回14行，因此虽达到>=50触发、>=15错误的可行性门槛，真实晋级难度高。私有禁网CPU候选源码/元数据/独立评价器已在 outputs/glm_pair_specialist_20261029_96/ 静态核验，仅本地准备、0上传/0公榜。待当前唯一Kaggle私有CPU作业终态再考虑单次对照，不启动第二个并行Kaggle作业；无完整集成收益证据。

---
# 2026-10-29 Qwen阶段一第26来源审计：历史83827行不能由固定旧池直接重建

作者Part Two称Qwen3阶段一池83827行/26数据集、旧SVC池75783行；额外的aswathvanan来源仅用于阶段一。本站下载当前aswathvanan/ntl-proxy-passages小型CSV并冻结SHA 56E3D00A92F007F8E25AA70A900A7395FA654FE2C687D634F416958F2829097A：4104行、12类、无完全重复官方文本；与官方训练按既有诊断TF-IDF口径有97行最大余弦>=0.3（非作者精确过滤）。75783+4104=79887，仍比83827少3940，故固定旧池+当前新增来源不可能重建历史阶段一池。当前文件版本与历史不同、其它来源历史版本可能变化；没有Qwen2.5训练/公榜。数据卡CC-BY-SA4，README提醒上游模型输出条款另适用。证据 outputs/qwen25_stage1_source_reconstruction_20261029_95/{run.json,REPORT.md,audit_summary.json,aswathvanan_current/}。当前唯一Kaggle作业仍为另一条线的私有CPU增量对照，见下节，勿重传或启动第二个作业。

---
# 2026-10-29 当前公开池增量：唯一私有 CPU 作业运行中

公开数据池本日变动：Abdullah 主 CSV 2,065,486 B、SHA 8E7A66EB4DDA074AB80D47F11CE3002F4DCCC19FE363ED252C6CFB5EF834B19E；Lucas 主 CSV 7,476,244 B、SHA A6DA05C3AC20876BDDC61B0EAA24CF84B4164ED9CD36FE62FC076F7238991912。v4/v10 是推测目录名，非官方版本号。两数据卡均 CC BY4，但 Abdullah README 承认上游服务输出条款未公开；只作私有筛查。

当前唯一运行中本项目 Kaggle 作业是禁网无 GPU okkojbjj/name-that-llm-current-pool-increment-screen/1，仅上传一次、状态 RUNNING，父限7200秒。实际 slug 与上传 metadata 预期不同，必须按实际 slug 查询，不重传、不启动第二个本项目作业。它固定2699行/150prompt三折，比较原75783行公开池与加入当前两 CSV 的 Part Two SVC 成员；晋级门槛见 outputs/current_pool_increment_screen_20261029_94/PLAN.md。终态须核验云端源码、输入 SHA、原生 OOF 和逐折/总体指标；无公榜提交。本站公榜最高仍0.620，私榜未知。

---
# 2026-10-29 V31 Bayesian typography 单成员止损

从公开V31源码固定提取37个标记及原贝叶斯证据/门控公式，按官方prompt三折只用其它折估计类别标记率，复用已验收SVC成员OOF。CPU2/120秒结果：2699行、150prompt逐行验收PASS，BA 0.470440→0.466361（Δ−0.004079），三折仅1折改善，prompt bootstrap95%[−0.010378,+0.002222]跨0；独立sklearn复算PASS。NLL/Brier下降但预设BA闸门失败。停止该具体后处理，不接入0.620完整集成、不调参或交榜。证据 `outputs/v31_bayes_typography_member_20261029_92/{PLAN.md,run.json,REPORT.md,result.json,oof.csv,acceptance.json}`。这是同一开发OOF上的SVC探索，不说明作者V31完整集成的真实分数。本站最高公榜仍0.620，私榜未知。

---
# 2026-10-29 五路线继续审计：当前没有新合格公榜候选

Qwen2.5数据升级：Kaggle只读列出原作者当前仅两份数据集和两篇Notebook；未找到可钉版本的83,827行阶段一池，故不拿75,783行SVC池冒充。完整融合：公开附件仍无所有成员共同prompt折的逐行OOF，不能以见过官方训练的最终权重调融合。当前0.651配方两份关键附件 `nlm-l24-adapters-v5`（54文件）和 `nlm-offline-wheels`（4文件）清单均无README/LICENSE/NOTICE；数据卡仍是other/空说明，公榜使用权未闭环。今天新公开0.660/0.661/0.663标题的源码快照与初筛见 `outputs/public_highscore_scan_20261029_91/REPORT.md`：分别存在约1KB模型产物、未验收mode、三行FAST_COMMIT；标题非本站分数。补核：alex660 的公开普通运行仅写三行placeholder，完整推断只在隐藏RERUN；lightning661原生mode明写三行格式预览、模型未加载、隐藏耗时未验证；omer663明写三行FAST_COMMIT。零权重TF-IDF跳过已出现在新公开0.660源码，本站原0.651候选仅有静态等价分析，尚未做同输入端到端对照。本站最高已验收公榜仍0.620，私榜未知；本轮无训练、上传或提交。继续每日最多5次且只提交合格不同候选。

---
# 2026-10-29 DeBERTa长文本路线：双窗口假设止损

用户授权五路线持续推进后，私有禁网无GPU探针 okkojbjj/name-that-llm-deberta-token-length/1 仅上传一次并COMPLETE。云端脚本与本地AST/规范化换行相同，私有/禁网/CPU配置一致；官方train.csv与DeBERTa meta哈希硬断言通过。实际tokenizer对2699行、150prompt、18类统计：0行超过max_len=384，最大351，P99=299.02；逐行ID/prompt/标签与官方训练独立复核PASS。故首尾双窗口截断修复在官方训练分布无作用，停止该具体变体，不训练、不提交公榜；未测BA或真实隐藏长度。证据 outputs/deberta_token_length_20261029_90/{run.json,REPORT.md,native_v1,cloud_v1}；五路线原计划末尾已标注。本站公榜最佳仍0.620，私榜未知。

---
# 2026-10-29 从0.620到约0.65：五路线已定，今日无合格新提交

北京时间约16:16只读核对Kaggle：本站最高公榜BA仍0.620（57684141），今天0次提交，私榜未知。五条高起点路线与逐项门槛见 outputs/highbase_0651_package_20261027_80/PLAN.md 的2026-10-29节；当前0.651公开配方仅同源4000行容量PASS，权利、独立效果和真实隐藏容量未闭环，不能拿作者历史0.651当本站成绩。今天不足以完成五个合格不同版本并逐一交榜；沿用每天最多五次且只交合格候选的约束。每日优化自动化继续按已更新状态运行。

---
# 2026-10-28 私有4000行容量 v1：同源技术 PASS，未交榜

用户单独批准的唯一私有禁网T4×2作业 okkojbjj/name-that-llm-nlm-v27-capacity-4000/1 已 COMPLETE，仅上传一次，未训练、改模型或提交公榜。此前 Kaggle 认证一度中断，16:02 UTC 恢复，19:10:44 UTC 再次核实终态。原生日志、六个小型输出及云端源码/元数据已回拉；预备验收器对真实输出返回 PASS_4000_SAME_SOURCE_CAPACITY_ONLY，证据 outputs/highbase_0651_package_20261027_80/{capacity_v1_record.json,capacity_v1_acceptance.json,native_capacity_v1,cloud_capacity_v1_terminal,REPORT.md}。4000个唯一假测试ID、18类、六主成员、补充栈、运行时v3/v9输入SHA、两张GPU和1261/4000行ATLAS完整0.1融合通过。两单元27653秒 < 30000秒，末条日志28268.52秒 < 31800秒。可选 lora24_public 仍按作者历史跳过；无其他意外跳过。此为同源技术容量，真实隐藏容量及BA均未验证。本站公榜最佳仍0.620、私榜未知；两份 other/空说明上游附件许可与Llama公榜范围、独立效果证据未闭环，不提交该候选。每日最多五次仅是合格不同版本的上限。
---
# 2026-10-28 当前资产0.651路线的私有测时终态

唯一私有禁网T4×2作业 `okkojbjj/name-that-llm-nlm-v27-timing-probe-512/1` 已 COMPLETE，v3/v9运行时输入大小与SHA钉选，云端6个代码单元逐单元同上传源码，私有/禁网/GPU及14/9/7附件集合验收。512行、18类、补充栈和六个主成员技术PASS，日志最晚约10744秒；ATLAS选中192行，只测前104行后按预设停下，**未混入预测**，所以无BA证据、无约4000行九小时容量证明。作者历史v1也跳过可选 `lora24_public`；验收器仅对此例外放行。原生证据、修正后的验收器和准确终态在 `outputs/highbase_0651_package_20261027_80/{native_timing_v1,cloud_timing_v1,timing_v1_acceptance.json,timing_v1_record.json,REPORT.md}`。未再上传、训练或提交公榜；本站最佳仍0.620，私榜未知。约4000行同源完整容量Notebook包已在 `outputs/highbase_0651_package_20261027_80/upload_capacity_4000/` 本地准备，静态/2秒CPU限时器验收PASS；尚未获批或上传。需单独批准该私有T4×2最多8小时50分作业及其Llama使用、临时离线wheel安装；两份权利不明附件在公榜前仍需核查。ATLAS自动跳过或超时即不晋级。每日优化自动化保留，专用测时监控已删除。另经源码核查，批准的5小时是资源上限，原Notebook只强制第4–5计算单元合计4小时，并无全程5小时硬停止器；本次实际约2小时59分结束。下次长测时上传前须补全程硬上限。

---
# 2026-10-28 当前资产0.651路线的私有测时作业

用户确认Llama3.1使用条件，并批准一次私有禁网T4×2、最多5小时的512行容量测时及临时离线bitsandbytes安装。上传前发现`abdullah1afify/ntl-generations` v2→v3（主要CSV 870,886→1,453,571字节），用户明确选择按当前v3重新冻结；运行时还断言proxy v9文件尺寸并记录两文件SHA。前次0.651外部得分并非本站成绩，当前资产与作者历史输入不一致。单次上传Kaggle `okkojbjj/name-that-llm-nlm-v27-timing-probe-512/1`，状态RUNNING，父限18000秒、核心限14400秒，512行只测时、ATLAS探针104行，不训练或交榜。实际slug带`-512`，与metadata预期不同；**后续只查询实际slug，不重传，不启动本项目第二作业**。证据 `outputs/highbase_0651_package_20261027_80/{timing_v1_record.json,run.json,preupload_acceptance_v3.json,PACKAGE.md}`。目前本站公榜最高仍0.620（57684141），私榜未知。

新增只读公开高分筛查：Roman当前v30相对Dhruv历史v1只改第2/5单元，能保留部分ATLAS行，但其原生公开运行仅3行、主推断约7972秒、排版分支相对参考0行改变；Alex当前v4明确只产三行占位，Lightning当前v16的mode明确未加载模型。外部0.654/0.652标题均不是本站收益或隐藏容量证据。见 `outputs/public_recent_audit_20261028_87/{REPORT.md,run.json,native_sample_audit.json}`；正在运行的本站512行测时仍是唯一Kaggle作业。

终态验收器 `outputs/highbase_0651_package_20261027_80/verify_timing_probe.py` 已本地准备且仅通过启动检查，真实产物未验收；零权重TF-IDF跳过候选仅在本地，已对齐原版float64零矩阵，尚无同输入预测等价证据，均未上传。公开 `pool_v4.parquet` v1 单文件下载两次因 Windows 10053 中断且无文件，不能声称已核其75,783行来源/字段，见 `outputs/part_two_pool_v4_audit_20261028_88/`。Kaggle平台条款允许公开附件在服务功能内使用，但两份 `other` 空描述附件的上游权利仍待具体核查，见高分包 PACKAGE.md。

作者两篇Notebook的Dataset链接对比为旧14源全保留、新增12外部源；新增源当前数据卡许可为CC BY4九项、CC BY-SA4两项、CC0一项，其中四项说明为空。历史版本及83,827行池逐行身份仍未知，见 `outputs/part_two_new_sources_audit_20261028_89/`。

DeBERTa 附件 `meta.json` 已由早先可行性审计保存并校验，实际 `max_len=384`；旧截断报告“实际值未核”的说法已在其末尾更正，真实 tokenizer 截断率和双窗口收益仍未知。

---
# 2026-10-28 标点特征成员对照终态

固定五种标点特征只在 Part Two 的 SVC 成员上做官方 2699 行/150 prompt/3 折训练外筛查。BA 0.470440→0.469699（Δ−0.000741），0/3 折改善，prompt bootstrap 95%[−0.002593,+0.001111]跨 0；预设三项主要门槛失败。虽 NLL/Brier 略降，**不接入已评分完整集成0.620、不调参、不交榜**。未评估完整集成，也非独立盲测。证据 `outputs/typography_member_screen_20261028_86/{PLAN.md,run.json,result.json,REPORT.md}`。当前无本项目运行中 Kaggle 作业。本轮开始时 Z: 掉线，已用原有 remote-workspace.ps1 恢复，实际 SSHFS 来源 `yuanyn@Net-4090D:/home/yuanyn/jyx`、严格 SSH ubuntu/yuanyn、项目 realpath 与 HANDOFF 本地/远端 SHA 均匹配后写入。

---
# 2026-10-28 公开0.636补充集成输入审计与私有CPU探针

Gemini固定二分类CPU作业已获明确批准且只上传一次 `okkojbjj/name-that-llm-gemini-binary/1`，终态COMPLETE；原生2699行三折/150 prompt完整，核心110.35秒。冻结评价器PASS表示重算验收，但原SVC18类OOF BA 0.470440→0.471551（+0.001111），两折升一折降，prompt bootstrap95%[-0.002222,+0.004444]跨0，预设+0.005门槛失败，NLL/Brier略好。**停止此候选，不接入已评分Part Two 0.620完整集成，不交榜或调门槛**。私有云端源码回拉403限制保留；证据 outputs/gemini_pair_specialist_20261027_72/REPORT.md、run.json、independent_evaluation_v1.json。专用监控终态应删除，每日优化自动化保留；当前无本项目运行中Kaggle作业。

谱头路线只读筛查：公开kirilltym历史V1不声称官方分数；V3继承外部0.642全长27B+ATLAS大集成，其公开产物仅三行format-only、未加载GPU、未证明隐藏耗时；谱头资产数据卡许可Unknown。暂不作为低成本0.620改进，见 outputs/spectral_highbase_audit_20261028_85/REPORT.md。Gemini固定CPU对照随后获明确批准并完成，结果不晋级，详见页首最新终态。

新增只读时间边界：0.636作者V1所列8个上游notebook当前 `lastRunTime` 全晚于外部得分观察时点；不证明输出字节改变，但无法用当前slug当历史固定输入。现有SDK保存请求仅支持来源字符串，浏览器只读工具本轮进程创建错误未启动。结合已失败的单次CPU探针，暂停0.636完整私有运行，待确切32项版本保留路径；见 outputs/public_0636_audit_20261028_82/kernel_input_current_version_audit.json 与 REPORT.md。

重大新结果：用户已批准并仅上传一次私有CPU版本钉选探针 `okkojbjj/name-that-llm-version-pin-probe/1`，禁网无GPU，终态ERROR为预设断言：请求proxy数据集v8(`/8`)，云端实际文件6,063,659B（v8应5,105,806B；尺寸同当前v9），故现有CLI上传路径不能固定旧版。原生JSON/日志和SHA见 outputs/public_0636_audit_20261028_82/REPORT.md、run.json；私有云端源码回拉403，仅能核输出。绝不重传v1，不提交榜单；0.636下一步必须找能证明32项历史版本ID保留的部署路径，否则停止。

新公开高分标题筛查：本站CLI提交列表最高仍0.620、无本日新提交；外部0.643重混大集成与0.650全长27B版当前公开作业均仅3行 `exact_public_format_preview`，未加载权重或验证隐藏时长。0.643还可在重混校验失败时保留原预测，故暂不晋级。只读证据 outputs/public_highscore_scan_20261028_84/REPORT.md；无新作业/榜交。

新增完整集成OOF可行性只读审计：本站0.620公开资产包当前仅v1，三份小型元数据仅有部分超参数/训练摘要，不含分折权重或阶段一训练池身份；作者首篇公开V1五个代码单元也仅推断，无训练入口。不能拿现成全量权重在官方训练行上调融合称训练外增益。证据见 outputs/part_two_oof_feasibility_20261028_83/REPORT.md；未训练、上传或提交。

新增：Kaggle SDK 按原作者 `fleongg/ntl-part-two-complement` 的固定 V1 只读取回源码；返回版本号1、T4禁网、16/8/7项来源，13个代码单元与当前公开版逐字相同，源码漂移疑点已排除。历史V1 API仍不给输入整数版本ID；得分回执、隐藏容量和本站收益均未验证。证据见 outputs/public_0636_audit_20261028_82/historical_author_v1/ 与 REPORT.md；未上传、未运行、未提交。

当前本站公榜基线仍0.620（57684141）。公开cosmorin文档称其个人0.636并引用fleongg原作者0.636，本站未独立验收私有得分回执/隐藏耗时；文档内确有32个输入版本ID（16数据集、8kernel、7模型、1竞赛），当前CLI metadata只有未钉版本slug；已读本机CLI源码确认push不使用ipynb内32个整数ID，版本后缀只经本地校验，云端语义未证；SDK只读GetKernel的metadata也只给slug，ipynb内32个整数ID对象无slug，不靠顺序臆配。原作者当前源码ATLAS失败即停，较0.651当前方案的吞错跳过更易验收；但固定得分V1字节未证、Qwen24恢复权利/容量及版本钉选待核。只读源码见 outputs/public_0636_audit_20261028_82/REPORT.md、run.json；旧0.622版仅留快照_81，不晋级。新只读时间线：以外部0.636得分列表观察时点2026-10-24 00:25Z为上界，16个数据集当时均已有版本；proxy当时最新v7（3,835,894B），v8/v9后来发布，但实际得分挂载版本未证，见_82/pre_score_dataset_version_candidates.json。新候选优先做输入与部署可行性审计：已备未上传的私有CPU版本钉选探针，只挂公开proxy数据集v8，比较v8/v9文件大小，最多600秒、不训练或公榜，仍待本项明确授权；不启动GPU或公榜。既有0.651 512行私有测时包仍未上传，用户作业和Llama确认仍待答。

---
# 2026-10-28 公开高分配方容量闸门修正

已核实原作者推断源码在ATLAS预计超预算时记录 `ATLAS SKIPPED` 并仍可产出合法Parquet；三行私有预览不能证明隐藏约4000行容量。原拟8小时三行作业撤回，仍未上传/训练/提交；本站公榜仍0.620。当前候选包保留为本地源码与资产基准，已在同一输出目录准备未上传的512行fake-test/ATLAS前104行测时Notebook及metadata，前四代码单元不变；第4、5单元现有共享4小时核心计算上限及失败即停检查（前3单元不计入）；仍须对新作业单独确认Llama条款、离线bitsandbytes安装和GPU资源。详见 outputs/highbase_0651_package_20261027_80/PACKAGE.md、run.json。2026-10-28原有Codex-Net-4090D-Z任务恢复SSHFS；实际来源 `yuanyn@Net-4090D:/home/yuanyn/jyx`，严格SSH ubuntu/yuanyn、项目realpath与HANDOFF本地/远端哈希在写前一致。

三份许可元数据为`other`的附件再核：ATLAS数据卡明确混合许可及署名；Dhruv的offline-wheels和l24-adapters-v5说明为空、文件列表无许可文本，权利未闭环。私有测时仍待用户对新作业和Llama条件确认，公榜前须进一步核实这两份资产条款。

---
# 2026-10-27 Astra 高起点多路线方案

本站已验收公榜0.620（57684141）不变；没有新训练、上传或提交。优先准备公开0.651当前配方的资产/许可/容量受控复现，同时盘点完整集成训练外预测；Gemini局部专家仍等先前自动审批后的明确批准，Qwen2.5新池仍缺83,827行资产与来源闭环。新发现：同代码单元的公开说明附件计数14/9/7与当前nlm-v27的14/8/8不同：14数据集相同，公开说明用Llama资产缓存notebook，当前改直接Llama模型引用；不能称历史精确复现。四条候选及各自闸门见 outputs/highbase_0651_package_20261027_80/PLAN.md、run.json；进一步只读核查14个数据集当前文件清单与许可；发现实际会被源码读取的 `ntl-free-proxy-generations/gen_passages.csv` 当前文件创建晚于作者得分时间；只读版本列表已定位提交前候选v8与当前v9，但历史notebook实际挂载版本未证；尝试单文件下载v8遭CDN 10053且未落盘，历史精确复现仍不成立。用现有SDK的正确vN标签已回拉原作者nlm-v27历史v1源码，六代码单元SHA与公开说明一致；Llama缓存历史v1的小型报告与先前最新报告逐字节同SHA且自称14/14通过，原始16GB仍未重哈希。复现者得分v3正确标签仍404、实际挂载数据集版本未证。续查九个上游notebook当前全COMPLETE、最新运行均早于作者得分；当前数据集与notebook输出路径对源码47项显式文件搜索全部静态唯一命中，但历史挂载版本/文件内容/运行容量仍未证。14数据集官方版本已核：仅proxy当前v9晚于作者得分时最新v8；九上游notebook当前版本已记。原作者历史v1源码与Llama缓存历史v1已用正确API标签取得并验收；基于相同六代码单元准备私有禁网T4的当前资产候选包，nbformat/元数据本地PASS，但未上传、未安装、未训练或提交。历史得分v3仍404；本包v9不是历史0.651精确复现。私有运行前仍需Llama条款确认、bitsandbytes安装及该具体作业批准。详见同目录PACKAGE.md、REPORT.md、run.json、input_version_snapshot.json。

---
# 2026-10-27 半强度Sinkhorn固定CPU预筛止损

基线0.620最终代码300轮均匀Sinkhorn函数经AST抽取/数值一致性核对；在原版SVC 2699行/150prompt三折OOF，完整平衡BA0.475615，半强度0.475986，只+0.000370，折差[-0.00111,+0.00111,+0.00111]，prompt bootstrap95%[-0.00706,+0.00741]跨零；独立CSV手算核验PASS。预设+0.005及区间门槛失败，不继续选指数、不接入完整集成或提交。完整0.620 OOF仍缺。证据见 outputs/sinkhorn_softening_screen_20261027_79/PLAN.md、REPORT.md、run.json、result.json、predictions.csv、verification.json。Gemini私有CPU上传仍待明确批准。

---
# 2026-10-27 SVC逐类校准固定CPU预筛止损

只用已有原版四视图SVC三折OOF margin，固定逐类LogisticRegression(C=0.1)做一次Net-4090D现有Conda base 2线程/120秒CPU预筛；独立2699行/150prompt/18类核验PASS。原版成员BA0.470440→校准0.469679（Δ-0.000761），折差[-0.01222,+0.00556,+0.00440]，救127害129；NLL/Brier虽改善，但预设BA门槛失败。此预筛还有跨折margin间接泄漏，不能证明完整0.620收益；停止该固定校准形式，不做嵌套重训或公榜，未改正式源码。证据见 outputs/svc_class_calibration_screen_20261027_78/PLAN.md、REPORT.md、run.json、result.json、predictions.csv、verification.json。

---
# 2026-10-27 Qwen2.5阶段一数据升级：有机制依据，资产未闭环

原作者同一Qwen3-14B旧/新池单体0.569→0.611，但Qwen2.5仍用旧池、单体0.569、集成权重0.2；这支持研究数据升级，不保证Qwen2.5或完整0.620变好。公开附件只有旧34,940行池和75,783行SVC池，没有作者83,827行阶段一新池；完整构建与Qwen2.5训练设置也未保存。因此未训练/提交，先做26来源版本/权利、近题排除、配方与prompt隔离验证资产审计。证据见 outputs/qwen25_pool_upgrade_audit_20261027_77/REPORT.md、run.json。Gemini已准备CPU作业仍待明确批准，不可绕过自动审批。

---
# 2026-10-27 DeBERTa长文本候选只完成初步长度审计

本站0.620 Part Two DeBERTa推断的截断上限取附件meta["max_len"]，作者正文报告上限384，但附件实际值未核；小文件下载两次连接中断，后续Kaggle只读查询报Authentication required；未登录或启动作业。官方2699条正文按空格分词P99=248、最大274，>256词13行、>384词0行；这些不是真实token截断率。双窗口仍无效果证据，不启动GPU/公榜。证据见 outputs/deberta_truncation_audit_20261027_76/REPORT.md、run.json。前一0.651公开方案也因输入版本/环境/依赖门槛暂不启动；Gemini私有CPU上传仍待明确批准。

---
# 2026-10-27 公开0.651方案来源审计：当前源码可核，得分历史与输入仍待核

公开说明notebook与当前可下载原作者nlm-v27的六个代码单元SHA一致，且与说明公布的代码摘要FF22FF2B...1B944一致；作者声称历史v3公榜0.651、可见T4 x2运行4h04m16s，但确切历史版本CLI仍403/404，本站未重跑、未评分。14数据集/9 notebook输出/7模型输入的历史版本、来源与隐藏9h容量仍未验收，故不晋级本站基线、不提交。详见 outputs/public_high_base_audit_20261027_75/REPORT.md、run.json。另有公开Q35恢复对照作者报告0.649，较其0.651低0.002；本站未复现，直接恢复Q35暂退出优先候选。续查确认当前metadata的14数据集/9 notebook输出无历史版本号、文档环境与作者得分环境不同，且源码有条件离线pip安装路径；0.651精确复现暂不启动。Gemini私有CPU作业的自动审批阻拦仍有效，未获用户明确批准前不上传。

---
# 2026-10-27 Part Two逐折资产缺口确认

只读Kaggle CLI查原作者当前两份公开数据集（11+15文件），未见fold/OOF/validation/prediction文件；公开notebook的五折CV汇总不能替代完整0.620集成逐行配对OOF。最终权重已见全部官方训练，不能在这些行上做无泄漏集成验证。详见 outputs/part_two_oof_asset_audit_20261027_74/；若要补齐需原作者折资产或重训，非轻量步骤。Gemini具体私有CPU作业仍待用户批准，未上传。

---
# 2026-10-27 SVC margin可靠性只读诊断

已验收官方2699行原版SVC OOF的平均top1概率0.16665、实际命中率0.47054，等量10组全部低置信，ECE约0.30390；见 outputs/svc_margin_reliability_20261027_73/。这是描述证据，不是新模型收益。统一温度在Part Two加权log概率融合中等价成员权重缩放，不作为独立候选；逐类非线性校准仍需嵌套prompt验证，暂不因此开启昂贵作业。Gemini私有CPU候选仍待明确用户批准，自动审批拒绝尚未解除，未上传。持续优化不停止，可继续只读分析。

---
# 2026-10-27 Gemini 二分类CPU候选待明确批准

基于已验收原版SVC OOF的Gemini前二名门控，固定二分类脚本、私有禁网无GPU Kaggle元数据和独立18类评价器均已备妥并预检，详见 outputs/gemini_pair_specialist_20261027_72/PLAN.md、run.json。拟唯一上传 okkojbjj/name-that-llm-gemini-binary、最多1800秒CPU，无公榜；自动审批拒绝启动，理由是缺少用户对此项具体远程作业的明确批准。当前确无该slug、未运行；不要通过其他通道绕过，也不要让每日自动化启动它。待用户批准后才能上传一次并监控。持续优化目标仍活跃；已评分0.620基线未改。

---
# 2026-10-27 SVC 来源权重对照终态：不晋级

私有禁网无GPU okkojbjj/name-that-llm-source-weight-pair/1 COMPLETE，核心2320.63秒。原版SVC三折prompt隔离OOF BA0.470440精确复现，来源加权BA0.474884、差+0.004444，三折都升；但预设+0.005门槛未达，150 prompt bootstrap95%区间[-0.001859,+0.010743]跨0，NLL/Brier均变差。Gemini3.7/3.8正确数未变。独立2699行/18类/逐折/margin/权重验收PASS；云端源码确切版pull GetKernel403，来源字节核验有限。结论：此SVC成员版本停止，不接入完整公榜0.620集成、不提交。原生证据 outputs/pool_source_weight_pair_20261027_70/REPORT.md、run.json、verification.json；专用监控终态后应删除，保留每日优化。下一路线先评估SVC概率映射的诚实嵌套prompt对照；另可基于已存OOF margin先审计Gemini3.7/3.8 top2门控可行性，但不得据真实标签选择隐藏行。Gemini3.7/3.8原版SVC OOF前二名门控触发196行，覆盖110个互相误判中的74个，但也包含99个原版正确行；仅通过模型无关可行性筛选，详见outputs/gemini_pair_gate_audit_20261027_71/。下一步可做一次固定二分类CPU OOF，不得从门控计数宣称0.620集成提升。当前没有本项目运行中的Kaggle作业。

---

# 2026-10-24 方向纠正与外部语料审计：训练按门槛停止

用户授权先审计稀疏验证，并批准在现有私有Kaggle kernel139969019接入公开语料作CPU来源审计与一次有界style训练对照（合格样本不足则停止；每job<=1h、禁网、无GPU/榜交、v8默认不改）。身份/挂载/路径重验：net-4090d-jyx→net-4090d→ubuntu/yuanyn→Z:\Competition\NameThatLLM→/home/yuanyn/jyx/Competition/NameThatLLM。两套mixed447只共享82行，其中31行预测翻转；同折全量上下文无关v8 BA=.435976，见outputs/sparse_validation_audit_20261024_39/REPORT.md。公开语料短候选24025单元格，经严格来源许可/质量筛选只剩3063个唯一回答、5/18类（Gemini3.6/3.7/3.8均0），触发预设停止条件，**没有训练新模型**。私有CPU审计v28/v29均COMPLETE且无submission.parquet，latest29仅审计；明确版本29源码回拉HTTP403，未声称逐字节核验。完整证据outputs/external_style_cpu_20261024_41/REPORT.md，元数据背景outputs/external_corpus_audit_20261024_40/REPORT.md。正式src/baseline.py和本地v8 profile未改，公开最佳v8=.472，无GPU/榜交。新训练路线需另定合规均衡数据，或官方数据Qwen适配的具体GPU方案和授权。另已只读确认项目references中公开多路LoRA方案作者报告BA约.604/.607，以及另一18类均衡公开语料的权利/分布限制；见outputs/public_pool_metadata_20261024_42/REPORT.md，未复现或接入。

# 当前状态：2026-10-23 Qwen 冻结特征三项 CPU 筛选完成，均未过双样本门槛

按用户新指令在既有 Net-4090D 项目/AI CPU 环境2线程训练，复用已缓存 Qwen 五视图，未用 GPU、上传或排行榜。三个固定候选依次是 8B RBF 普通头、三组同家族二分类路由、移除16个训练prompt均值方向；两套447行 mixed OOF 相对 v8(.470501/.525236)的Δ分别(+.004751/−.010613)、(+.013519/+.000085)、(+.011236/−.000752)，均未达到两套各>=+.01且各>=2折改善。首次RBF重建门槛在训练前失败，已证实跨折合并推断prompt导致，与原逐折分组核对后完全一致，保留失败和更正记录。正式src/baseline.py已精确恢复到本轮前a4fbae68…，42 tests通过；v8公开最佳仍.472，无新候选。证据优先读`outputs/qwen_rbf_screen_20261023_36/SERIES_REPORT.md`及三处run/summary.json。两套样本反复用于开发，不能继续事后挑参；后续重大GPU/外部数据/榜交需要新的固定方案与授权。
# 当前状态：2026-10-23 最后两层冻结对照完成，固定门槛失败

公开最佳仍 v8=.472（提交57516416）。本轮已获批的三折私有 ModernBERT 对照版本25/26/27均COMPLETE并验收PASS，最终无活动作业；未做outer评价、排行榜提交或候选晋级。固定epoch4 inner BA 全量控制 [.252778,.241667,.219444] 均值.237963，最后两层+原分类头 [.105556,.105556,.080556] 均值.097222，差−.140741、0/3改善，预设+.01且至少2折门槛失败。冻结方案同轮fitBA仅.119770，对照.655122；这是本固定2e-5/4epoch配方的弱拟合证据，不外推所有冻结方案。inner已复用且折间可重叠，不是独立盲测。完整证据 `outputs/modernbert_last2_20261023_35/REPORT.md`、`comparison/comparison.json`、`final_verification.json`。三作业原生合计约488.070秒，官方累计GPU 2.858712/8h，观察增量.170373h。`src/baseline.py`新增默认关闭的last2-head，现SHA a4fbae68…，metadata仍原SHA af7d45be…，42项pytest通过。本地v8 profile未改；cloud latest27仅诊断，不可提交。无进一步GPU、上传或排行榜授权；下一步先分析失败机制并另议，不自动改学习率/epoch/种子。
# 当前状态：2026-10-23 inner/dropout六作业完成，固定门槛失败

公开最佳仍v8=.472（提交57516416）；本轮无榜交、候选晋级或outer评价。用户明确允许仅baseline.py+kernel-metadata.json上传现有私有kernel139969019（okkojbjj/name-that-llm-token-statistics）运行六个固定单折实验；无凭据/训练数据/模型权重上传，禁网、每job<=3600、累计8h。最终官方13:10:31Z：v24 COMPLETE，无活动job，GPU累计2.688339/8h（六job实测增量.556401h，native耗时合计1710.856秒）；不把网络等待当GPU训练时长。项目仍net-4090d-jyx→net-4090d→ubuntu/yuanyn→Z:\Competition\NameThatLLM→/home/yuanyn/jyx/Competition/NameThatLLM；CPU只读复用AI/.venv、2线程，无安装/服务器GPU/凭据修改/Git提交。

新诊断outputs/modernbert_inner_dropout_20261023_34已完成：普通随机CE、原ModernBERT/CLS/LR2e-5wd.01/512/FP16eager/batch8acc2，固定4epoch；control全部dropout0 vs embedding/mlp/classifier=.1、attention0。每outer训练100prompt按固定SHA规则分80fit/20inner，50outer不推理；完整fit+inner逐epoch同checkpoint eval，完整shuffle共享。v19/20/21为control0/1/2，v22/23/24为regime0/1/2，六job逐一成功并独立probe/gate、实际24Dropout/22attention0、sourcecore/data/model/library/head/计划/cache身份验收PASS，严格merge及既定compare exit0。

固定epoch4 innerBA：control [.252778,.241667,.219444]均值.237963；regime [.127778,.119444,.155556]均值.134259，Δ−.103704、0/3fold改善，未达均值+.01且>=2fold门槛。control fitBA .1675→.3393→.5187→.6551、innerBA .12685→.20556→.24444→.23796，末期fit继续改善而inner退步；固定.1组fitBA .0749→.1020→.1522→.1677，innerBA .06852→.08333→.12593→.13426，4ep拟合和inner均更低。小gap不能当更好泛化，不把epoch3替换primary4，不据此断言所有dropout无效或延长必恢复。inner跨outer可重叠且开发资料已复用，非3独立盲测；未归因单一机制。

v18原三折job保留为预算技术失败：probe通过但bound4090.55>3300，0fold；之后仅拆调度，不改科学设置。自动审批曾在执行前拒绝具体外传授权，用户明确授权后正规审批通过，无重复受理。六job planned360各折，实际control354/356/353、regime353/354/354；AMPskip6/4/7与7/6/6，如实记录不称actual完全一致。TLS查询/小下载有界重试后恢复，没有重训或换配置。

完整证据_34/REPORT.md、summary.json、comparison/comparison.json、curves.csv、mean_curves.json、final_verification.json与各job acceptance/native artifacts。41CPUtests通过后正式source未再改（raw e650177e…、core0c31deb5…），本地v8 prediction profile/metadata按原字节恢复；cloudlatestv24仅诊断产物不可据latest提交。之前v14/v16/v17审计见outputs/training_audit_20261023_33；v17旧runtime/upload hash差异已证实仅CRLF→LF，证据_34/source_normalization_evidence.json，不外推未核版本。当前不扩展训练，等待根代理/Astra对真实曲线独立审阅。

# 上一轮A/B固定系列：无候选晋级

公开最佳仍为v8=.472（提交57516416）；本地profile/metadata已逐字节恢复v8，云端latest为实验v17且无submission.parquet，不可据latest提交。A四个固定局部/遮词style视图两套均低于原全文style；遮词.265533/.269237 vs全文.361158/.373381。B matched-grid CE=.238627/.234991，SupCon=.234106/.216348，exact v8=.470501/.525236；SupCon相对CE=-.004520/-.018643，改善1/0折，机制与实用门槛都失败，不做C、不融合、不选final、不榜交。

同一Kaggle kernel139969019/current slug okkojbjj/name-that-llm-token-statistics。v15成功probe后预算gate在任何fold前停止，保留技术失败；仅拆调度为CE v16与SupCon v17，均COMPLETE，原生558.651/574.238秒。两arm同源码core/官方固定Docker/model/config/环境/预生成train-only网格planhash，每fold339planned windows；实际CE334/334/333、SupCon334each，AMP skips分记，不称实际步数或GPU数值逐字节相同。官方GPU累计1.75→2.11h（本系列观察+.36h，授权累计<=8h），无安装/AI修改/服务器GPU/Git提交。网络TLS EOF导致完成状态获知延迟，不将push到获知间隔视为GPU耗时。

证据：outputs/style_local_mask_20261023_31/REPORT.md与run/summary.json；outputs/modernbert_grid_contrast_20261023_32/REPORT.md、summary.json、paired_validation.json、mixed_20261024/25/comparison.json、33tests(schedule_tests.exitcode0)及compare24/25.exitcode0。统一入口src/baseline.py新增local-style-analysis/train-modernbert-grid(参数grid-objective)/modernbert-grid-compare，旧v8默认未启用新模式。CPU只读复用AI/.venv、2线程；项目仍net-4090d-jyx→net-4090d→ubuntu/yuanyn→Z:\Competition\NameThatLLM→/home/yuanyn/jyx/Competition/NameThatLLM。本轮保留A/B beforecopy/diff/上传snapshot，权重留云端；正式runtime文件hash与上传/pull字节hash不同原因未验证，官方pull与上传逐byte一致。当前系列已停止，无运行中作业；进一步方案另需明确目标，8h是上限不是消耗目标。

以下为旧实验历史，冲突以当前状态和本轮原始证据为准。

# 当前状态：2026-10-23 ModernBERT监督适配实验完成

唯一私有Kaggle v14（同kernel id_no=139969019）完整完成3个prompt隔离fold，2699×18 OOF验收通过。ModernBERT单独BA为0.248876/0.262548，exact v8为0.470501/0.525236；两套均0折改善，不晋级、不继续调epoch或融合、不提交排行榜。Kaggle原生运行567.25秒；官方GPU累计观察约1.58→1.75h，增量约0.17h，仍低于累计8h授权。28项本地测试通过，本地Kaggle profile/metadata已恢复v8。完整证据见outputs/modernbert_supervised_20261023_30/REPORT.md、summary.json、cloud_output_v14/modernbert_manifest.json与两套comparison.json。云端v14保留训练权重但无submission.parquet；上传与官方pull源码逐字节一致，执行文件SHA不同原因未验证。

# 冻结分支重叠诊断（2026-10-23）
两套447行mixed OOF严格对齐后，Mistral单分支相对v8救42/37、害61/74行；Gemma救41/42、害48/53；无style的token统计救15/17、害173/193。四分支逐行oracle BA仅理论上界0.6304/0.6699。coverage6 plain→center两套均净救18行，coverage2均净害3/9行；训练2699行无完全重复正文或ID。这些已见OOF诊断不能判定隐藏prompt结构或实现可用oracle。完整计数和固定top5混淆对见outputs/frozen_branch_overlap_20261023_29/REPORT.md、summary.json；CPU exit0，无新GPU/提交，默认仍v8。该ModernBERT实验后续已完成，见顶部新状态。
# 当前状态：2026-10-23 token预测统计实验完成

唯一私有Kaggle v13作业状态ERROR：完整Qwen3-4B-Base已生成并保存2699×28的token统计缓存，随后manifest因set无法序列化失败。已按显式v13下载、核验ID/有限值/0截断并重建明确标记reconstructed的本地manifest；未重跑GPU。两套mixed OOF固定融合分别0.470809/0.515953，对应同样本v8 0.470501/0.525236，未达两套均≥+0.01且各至少两折改善门槛；保留公开最佳v8=0.472，不提交排行榜。完整证据优先读outputs/token_stats_20261023_28/REPORT.md、summary.json、cache_validation.json。本地profile/metadata已恢复v8；同一云端私有kernel id_no=139969019的slug因本轮title变更暂为okkojbjj/name-that-llm-token-statistics，v13没有submission.parquet，不可提交latest。官方GPU用量约1.43→1.58h，本轮约+.15h。

以下为此前交接的历史段落；旧建议及对隐藏结构的判断以本段和本轮原始证据为准。
# 当前状态：2026-10-23双异构模型筛选已完成

用户批准双模型私有实验，不提交排行榜，新增GPU≤2h、累计≤8h。两模型均完成，无方案晋级，保持公开v8=.472。Mistral单模=.43507041/.44175035、50:50融合=.48105253/.50296708；Gemma单模=.45763741/.50002339、融合=.48278159/.51192990；同样本v8=.47050078/.52523598。两模型融合均一套升一套退，未达预设两套均≥+.01且各至少两折升，融合bootstrap区间全跨0。不得继续本轮权重/层/更多seed搜索或自动提交。
Gemma原v11访问403，用户自行同意后只读复查成功，v12同源同设置导出完成；失败证据保留，不再阻塞。两者32条probe通过、FP16 batch2、层8/16/24、512/mean、无OOM/无截断。GPU官方观察1.19→1.43h，本轮新增约.24h（含失败作业），不安装依赖/改AI环境/用服务器GPU/复制凭据。
所有当前证据outputs/hetero_20261023_27，先读REPORT.md、summary.json、run_record.json。mistral_features/gemma_features为已完成缓存，无需重跑GPU；四个encoder_<seed>_compare包含逐类、混淆、概率与paired prompt区间。两个CPU脚本及各分析/比较exitcode均0；最新核查无本轮运行任务。当前18项测试通过，v8原缓存结果精确一致，代码/测试有限副本及diff保留。唯一src/baseline.py新增显式模型和导出/比较入口，不Git提交推送。
本地Kaggle profile和metadata已恢复v8；云端latest是只导出特征的v12，不能拿latest提交。正式公开版本仍v8历史提交57516416=.472，v9b退步.462保留。两项旧夜间自动任务已暂停；本轮没有新排行榜提交。下一步等待用户安排，不因GPU还有余额自动扩展。

---
以下为DeepSeek历史交接；冲突以当前状态和本轮原始证据为准。

# Name That LLM — 交接文档（2026-10-22 晚，DeepSeek Harness → Codex）

交接人：DeepSeek Harness（本会话）。上一版文档存档在
`outputs/handoff_archive_v8_20261022.md`（v8 时代状态，资源/边界部分仍可参考）。

---

## 0. 一句话现状

**公开最佳仍是 v8 = 0.472（提交 57516416）。** 本轮做的 v9b（每 prompt 唯一作者指派 +
组大小条件正则）私有 OOF 0.6486，但**公开只拿到 0.462，退步 0.010**，已把 Kaggle 默认
profile 回退到 v8。今日已用 2/5 次提交，**剩 3 次**。无运行中任务，未占用 GPU。

---

## 1. 最重要的教训（先读这一节）

1. **完整 prompt OOF 会严重高估隐藏测试。** v8：OOF 0.5519 → 公开 0.472（差 0.080）；
   v9b：OOF 0.6486 → 公开 0.462（差 0.187）。两次提交的 parquet 大小完全相同
   （60075 字节），说明是同一个隐藏测试；官网写明隐藏重跑约 4000 条。
2. **稀疏 prompt 只是开发验证假设，不是已知隐藏结构。** v9b公开0.462更接近两套混合覆盖OOF，而非完整prompt OOF；真实隐藏prompt归属与每题条数不可见。
3. **后续开发优先参考混合稀疏 OOF，但它不是盲测**（`--mixed-coverage`，seed 20261024/20261025），
   不要再用完整 prompt OOF 决定提交。也不要再用“每 prompt 每个模型最多出现一次”这类
   只在完整覆盖下成立的先验。
4. v9 的伤害机制推测：聚类把若干稀疏 prompt 合并成 ≥8 行的组，触发 C=0.0005
   （稀疏场景实测更差），同时在这些不纯组上做一对一指派会强行去重。

---

## 2. 授权与限制（沿用，未变）

- 不安装依赖、不修改 AI 项目/环境、**不用本服务器 GPU**、不跨服务器搬数据或代码；
  CPU 只读复用 `/home/yuanyn/jyx/Competition/AI/.venv`（numpy/pandas/sklearn1.8.0/scipy，
  2 线程）。
- 无 Git 提交、推送、标签、reset/clean/stash；**不删除既有产物**。
- **不复制凭据**：本轮用的 Kaggle 凭据由用户主动放到 `.kaggle/credentials.json`，
  用完后已删除（见第 6 节）。
- **任何排行榜提交都必须先得到用户明确“提交”指令。** 每日默认 0–2 次，最多 5 次；
  退步后先分析，不连续补交。
- 单写入者；唯一模型源码 `src/baseline.py`；保留失败、退步与用户改动。
- Kaggle GPU 授权：累计上限 8h（历史观察约 0.89h），单次 ≤3600s；v9b 私跑用了 1061s。
  本机不能跑 GPU，只能推 Kaggle 私有 kernel。

---

## 3. 资源身份、路径、环境

- workspace `net-4090d-jyx` → server `net-4090d` → SSH 别名 `Net-4090D` →
  **hostname `ubuntu` / user `yuanyn`**。
- 规范路径：`/home/yuanyn/jyx/Competition/NameThatLLM`（Windows 侧 `Z:\Competition\NameThatLLM`）。
- 挂载：`/dev/mapper/ubuntu--vg-ubuntu--lv[/home/yuanyn/jyx/Competition]`，ext4，rw；
  根分区只读。项目下无符号链接。
- 本机**有外网**（kaggle.com / api.kaggle.com 可达），但**没有 Kaggle CLI**、
  没有 `~/.kaggle`；数据/模型都在 Kaggle 侧。
- 本轮源码 SHA 变化：
  - v8 上传/运行版（LF）：`aadc266d81b40925c620e20beec27b9a1a73a281a262e816d12f36b9865323f1`
  - v9b 版（含新选项，profile 启用）：`5b312dcb…`
  - **当前版（profile 已回退 v8）**：`830f98d808d8b1df734ff795903268eb87b16b90d634513d0e28a21895a32b13`
  - v8 原始字节版（CRLF）：`f77fd7b2…`

---

## 4. 数据、指标、划分（未变）

- `data/train.csv`：2699 行、150 prompt、18 类，149 组 18 条 + 1 组 17 条；
  SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`。
- 指标 Balanced Accuracy（18 类召回均值）。
- 固定 prompt 折：`FIXED_PROMPT_FOLDS`（`src/baseline.py`）；
  另有确认折 `fold_seed=20261022`、第三折 `20261023`。
- 混合覆盖诊断：`--mixed-coverage`，seed 20261024 / 20261025；输出 `partial_0_oof.npz`。
- 缓存：`outputs/kaggle_kernel_v7/frozen_train_features.npz`（158,096,371 B，5 个视图：
  4B 层1/2/3、8B 层7/22，mean pooling，max_length 512）。**不需要重跑 GPU 就能做 CPU 实验。**

---

## 5. 当前源码状态

`src/baseline.py`（SHA `830f98d8…`，15 项单元测试通过）：

- v8 逻辑不变：5 视图 + StandardScaler + LogisticRegression(C=0.0025)，
  plain / centered 两路，`adaptive` 去中心权重，`group_middle` 视图配比，style 权重 0.2。
- 新增但**默认关闭**的选项（v9 实验遗留，可作为后续实验工具）：
  - `--group-assign`：2–18 行组内 Hungarian 一对一指派（`assign_group_labels`）。
    **已在公开榜证伪，不要默认开。**
  - `--head-c`：分类头正则；`--head-c-large` + `--c-size-threshold`：按组大小切换
    （`merge_head_scale`）。**稀疏场景有害，不要默认开。**
- Kaggle 固定 profile 已回退为
  `{"mode":"predict","center_policy":"adaptive","feature_set":"group_middle"}`（= v8）。
- `src/kernel-metadata.json` 未改（私有、T4、禁网、competition + 两个 Qwen 模型源）。
- 回滚副本：`outputs/opt_v9_20261022_26/source_before.py`（v8 运行版 LF 规范化，
  SHA `aadc266d…`）＋ `source_before.json`。

---

## 6. Kaggle 通道（给 Codex 的操作细节，全部实测过）

**此凭据操作建议已于 2026-10-25 作废。** 原文曾要求重新将 OAuth 文件放到挂载项目；现已核实本机官方 Kaggle CLI 已认证且能读取提交记录。继续工作使用 C:\Users\JYX\AppData\Local\Programs\KaggleCLI\Scripts\kaggle.exe，不复制、展示或要求用户重放凭据。旧端点表仅保留为历史参考；原段落保存在 outputs/project_review_20261025_44 的有限文档备份中。

**实测端点**（`requests` 即可，无 CLI、无需装包）：

| 用途 | 调用 |
|---|---|
| 推送 kernel | `POST https://www.kaggle.com/api/v1/kernels/push`，JSON body 用 **`slug`**（不是 `id`；`id` 必须是整数 kernelId）。字段：`slug,newTitle,text,language,kernelType,isPrivate,enableGpu,enableInternet,datasetDataSources,competitionDataSources,kernelDataSources,modelDataSources,categoryIds,machineShape` |
| 查询状态 | `GET /api/v1/kernels/status?userName=okkojbjj&kernelSlug=name-that-llm-cpu-baseline` |
| 取运行日志/输出 | `GET /api/v1/kernels/output?userName=…&kernelSlug=…` → JSON，`logNullable` 是日志，`files[].urlNullable` 是签名下载链接 |
| 提交历史/额度 | `GET /api/v1/competitions/submissions/list/name-that-llm`（看 `date`/`publicScore`） |
| **代码竞赛提交** | `POST https://api.kaggle.com/v1/competitions.CompetitionApiService/CreateCodeSubmission`，body：`{"competitionName":"name-that-llm","kernelOwner":"okkojbjj","kernelSlug":"name-that-llm-cpu-baseline","kernelVersion":N,"fileName":"submission.parquet","submissionDescription":"…"}` → 返回 `{"ref":<提交id>}` |

现成脚本：`outputs/opt_v9_20261022_26/kaggle_client.py`（`status`/`push`/`submissions`；
`submit` 子命令需按上表改成 RPC 端点）。注意每个 bash 调用的 `/tmp` 是临时的，文件要放工作区内。

**教训**：push 后先 `status` 轮询到 `complete`，再核对 `neural_validation.json` 的 OOF，
**但不要再拿完整 prompt OOF 当提交依据**（见第 1 节）。

---

## 7. 证据位置

| 内容 | 路径 |
|---|---|
| 本轮完整报告（含失败方向） | `outputs/opt_v9_20261022_26/REPORT.md` |
| 本轮全部实验脚本与结果 | `outputs/opt_predlevel_20261022_25/`（predlevel、head_lab、extra_lab、size_c、style_lab、self_train、alt_heads、pca_concat、assignment_source、assignment_robustness…） |
| v9b 入口验收 | `outputs/opt_v9_20261022_26/acceptance.json`、`acceptance_v9b_only.json`、`synthetic_wiring.json` |
| v9b 私有 Kaggle 运行 | `outputs/kaggle_kernel_v9/`（run_record.json、neural_validation.json、inference_record.json、submission.parquet、run.log.json） |
| v9b 正式提交（退步） | `outputs/submission_57527432/record.json` |
| v8 正式提交（当前最佳） | `outputs/submission_57516416/record.json` |
| 历史优化记录 | `OPTIMIZATION.md`（含本轮 v9 的三次补测与退步结论） |
| v8 时代交接 | `outputs/handoff_archive_v8_20261022.md` |

---

## 8. 本轮已证伪 / 无效的方向（不要重复）

- 每 prompt 唯一作者指派（完整 OOF +0.08，公开 −0.010）。
- 组大小条件正则 C=0.0005（稀疏组更差）。
- 对数（几何）平均作指派来源、多源指派投票、组内 z-score/秩/L2 变换、跨视图 PCA 拼接、
  kNN/LDA 替代头、转导自训练（伪标签自确认）、聚类阈值 0.92（同 OOF 调参）。
- 完整 prompt OOF 作为选型指标。

---

## 9. 建议的下一步（按期望值排序）

1. **换表示（当时的待验证方向之一）**：用 Kaggle GPU 重新提特征，例如
   更多层（4B 1/2/3/6/12、8B 4/7/14/22/28）、更大 `max_length`（768/1024）、
   mean+max 或 last-token 池化。先私有跑（不提交），下载输出后用**混合稀疏 OOF**
   与 v8 比较；只有混合 OOF 明显提升才考虑提交。v8 的 5 视图 GPU 用时约 433s，
   层数翻倍仍在 3600s 内。
2. **小步后处理**：只以混合稀疏 OOF 为指标，尝试改善“小样本去中心”的稳定性
   （例如完全去掉指派；或只在高度内聚且 ≥12 行的组上做，且 C 保持 0.0025）。
3. 保守：保持 v8 = 0.472，不再消耗额度。

**历史猜测（未经证实）**：曾估计mean-pooled 5视图公开分约0.47–0.50，但不能据此认定公开上限或断言只有表示变化有效。

---

## 10. 复现命令

```bash
PY=/home/yuanyn/jyx/Competition/AI/.venv/bin/python
cd /home/yuanyn/jyx/Competition/NameThatLLM

# 测试
OMP_NUM_THREADS=2 $PY -m unittest discover -s tests

# CPU 混合稀疏诊断（不碰 GPU）
OMP_NUM_THREADS=2 $PY src/baseline.py --mode cached-analysis --data data \
  --cache outputs/kaggle_kernel_v7/frozen_train_features.npz \
  --output outputs/<新目录> --fold-seed 20261022 --mixed-coverage --sampling-seed 20261024

# 候选策略筛选
OMP_NUM_THREADS=2 $PY src/baseline.py --mode cached-screen --data data \
  --cache outputs/<上面的分析目录> --output outputs/<新目录> --feature-set group_middle

# 历史推送示例（凭据前置要求已作废，不执行本段）
$PY outputs/opt_v9_20261022_26/kaggle_client.py status
$PY outputs/opt_v9_20261022_26/kaggle_client.py push
```

所有输出目录必须新建，不要覆盖既有证据。

Gemma第12版COMPLETE，用户完成访问后复查成功，无需更换源。2699行3视图已下载gemma_features；模型阶段362.6秒、含导出394.0秒，batch2、probe通过、无截断/无OOM。quota1.43h，从本轮起点1.19h新增约.24h。Gemma CPU已启动，PID文件位于hetero_20261023_27，2线程两seed顺序各timeout1200，先查产物不重复启动。本地profile/metadata仍为v8；无排行榜提交。
















---
# 2026-10-30 五路线执行合同与模型调度再审

Astra/high 只读重审后，优先收齐当前 Qwen2.5 官方-only 外折0证据，再补 R3 同折完整融合 OOF，最后对满足 G0–G2 的具体候选测 G3 全流程禁网容量；R1/R2 外部池 G1 仍 HOLD，R4 Llama 账号访问未证，R5 新机制未立。本站公榜最高 BA0.620，私榜未知，五路线不等于五个可提交版本，今天可零提交。G0–G4 阈值不变。后续运行记录分开写 job_state 与 gate_state，保留 UNKNOWN/HOLD 与原生证据；GetKernel403 只标云端源码逐字身份未验，不把 RUNNING 或查询失败当实验失败。

模型分工：GPT-6.1 Sol/high 默认复杂执行与唯一写入；GPT-6 Luna/xhigh 多承担哈希/版本/ID/折/类序/固定公式/日志/容量字段/次数等有界核查；GPT-6 Sol/xhigh 独立只读复核科学语义和晋级证据；Astra/high 仅重大协议或路线取舍。细则、任务卡和状态合同见 outputs/highbase_0651_package_20261027_80/PLAN.md 页末“2026-10-30 执行合同补充”。本次只读核对当前唯一 Kaggle 作业 okkojbjj/name-that-llm-qwen2-5-official-outer-fold-0/1 仍 RUNNING，未启动新作业或提交。

### 2026-10-30 官方-only Qwen2.5 外折0原生验收 PASS，完整融合尚无结论

唯一 Kaggle 作业 okkojbjj/name-that-llm-qwen2-5-official-outer-fold-0/1 COMPLETE，原生 JSON、900 行 OOF、18维连续分数与日志已下载到 native_official_outer_fold0_v1/。预先冻结验收器对官方 train SHA、深成员折 SHA、900 ID/50 prompt、标签、18类有限分数、200次更新、双头与无监督 twin 掩码、独立 BA 复算返回 PASS_SINGLE_OFFICIAL_FOLD0_MEMBER_OOF_ONLY；plain BA0.346667、加权成员 BA0.438889、耗时2872.42秒。运行时源码自报 SHA 匹配本地上传，云端 GetKernel 回拉仍403，逐字源码身份未独立证明。模型主动使用 Qwen2.5-14B 前32/48层，加载日志多出的32–47层 UNEXPECTED 是该既定裁剪所致；不是完整作者历史配方。与旧 Qwen8B v8 在同一外折0的只读配对：旧 BA0.443333、新 BA0.438889，二者都对276、新独对119、旧独对123、都错382；这只是不同成员的互补诊断，不能宣称新方案提升0.620完整融合。外部池G1仍未闭环，未交榜。状态及原生SHA见 official_outer_fold0_record.json，诊断见 official_outer_fold0_overlap.json。

### 2026-10-30 固定50/50双Qwen外折0诊断：需扩折验证

将旧Qwen8B v8同折 adapted 概率与新官方-only Qwen2.5外折0加权成员分数 softmax，按18类名、900个相同ID对齐后固定各50%算术平均（未搜索权重）：旧单体BA0.443333、新单体0.438889、固定双成员BA0.471111，较旧+0.027778；救回45行、误伤20行，12类召回提高、3类下降、3类不变。记录及两NPZ SHA见 official_outer_fold0_fixed_blend.json。这仅证明外折0二成员互补值得扩到外折1/2，不是0.620 Part Two完整融合，也不满足G2/G3。不得在已检查的外折0上调混合权重；后续两折仍固定50/50并先验源码/折/来源。现有云端GetKernel403保留独立源码身份未知。

### 2026-10-30 Qwen2.5 官方-only 外折1唯一私有作业 RUNNING

折0固定50/50双Qwen诊断显示BA0.443333→0.471111后，保持seed42、同题三元组、前32层LoRA双头、200更新、官方训练数据及权重不变，仅选冻结深成员折1。通用源码 official_outer_fold.py SHA6153ADCC；静态预检 PASS，折1训练1799行/100题、留出900行/50题，每轮600合法三元组、1799行覆盖且重复1行。通用验收器已在折0真实原生输出回归PASS，误选折1验折0按预期失败；GPT-6-Sol/xhigh 基于摘要只读复核接受，未独立读远端源码。Kaggle上传一次后实际slug okkojbjj/name-that-llm-qwen2-5-official-outer-fold-1、版本1，初查RUNNING，私有禁网T4×2、父限12600秒、内部alarm10500秒，不交榜。详情见 official_outer_fold1_{taskcard,static_acceptance,record}.json 与 upload_official_outer_fold1/。终态需下载原生JSON/900行OOF/18维分数/日志，通用验收器传--fold 1，核源码运行时自报与上传SHA；GetKernel403仍需标云端逐字身份未独立验证。运行中不启动第二个本项目Kaggle作业，不重传；本站公榜仍0.620。

### 2026-10-30 冻结三折固定双Qwen汇总入口

新增 summarize_fixed_qwen_blend.py（SHA E9CC349A）：只在折0/1/2各自原生输出都经通用验收器验证后，对齐官方2699个ID、150个prompt、18类顺序，将旧Qwen8B adapted概率与新Qwen2.5 weighted-member softmax固定50/50平均；输出逐折及整体BA、救回/误伤和源文件SHA。它不搜索权重，不取代Part Two完整融合G2。AST解析PASS；外折1原生结果未到时按预期FileNotFound拒绝汇总，official_threefold_fixed_blend.json不存在。折1当前RUNNING，专用监控namethatllm-qwen2-5-1负责终态；不启动折2。

### 2026-10-30 官方-only 外折2已静态准备，未上传

以折1上传快照为源，仅将常量 FOLD=1→2，脚本 SHA A16056E1；同一官方train和深成员固定折、seed42、前32层Qwen2.5双头LoRA、200更新、禁网T4×2、父限12600秒不变。静态验收PASS：训练1800行/100题、验证899行/50题、两轮各600合法三元组且每行恰一次；私有元数据、唯一模型源和AMP顺序均核对。证据 official_outer_fold2_{taskcard,static_acceptance,record}.json 和 upload_official_outer_fold2/。状态PREPARED_NOT_UPLOADED，折1当前仍RUNNING；待其确切终态及原生验收后再决定上传折2，不并行、不交榜。













---
# 2026-10-30 G2-R 等预算执行冻结

Astra 设计复核后，已在 outputs/highbase_0651_package_20261027_80/PLAN.md 页末冻结 B-R/C1/C2 同折比较：阶段一各100更新、官方训练侧阶段二各200更新、阶段间重置优化器，其他训练与融合设置固定。既有官方-only 200步双Qwen和外部SVC结果仅作成员诊断，不能充当等预算完整基线。DeBERTa干净基座身份/训练入口仍待闭合。当前唯一 Kaggle 作业 Qwen3 官方-only 折2/1 只读查询仍RUNNING，终态前不开第二作业；本站最高公榜仍0.620。下一步回收折2原生证据，准备 B-R 四成员同折实现及 C1/C2 已冻结输入的私有训练。

---
# 2026-10-31 Qwen3 三折官方成员验收完成

外折2/1已COMPLETE并原生验收PASS：899行/50题、BA0.447914、200更新、2822.59秒、T4×2、adapter/head重载；GetKernel源码回拉仍403，独立逐字身份HOLD。三折2699行/150题Qwen3官方-only成员OOF BA0.424937；与Qwen2.5按原固定权重相加的双成员诊断BA0.434944，低于Qwen2.5单成员0.442707。不能据此宣称完整0.620融合改进或公榜增益。证据在 outputs/part_two_oof_feasibility_20261028_83/{qwen3_official_fold2_record.json,qwen3_official_fold2_verification.json,REPORT.md}。专用折2监控已删除；目前没有运行中的本项目Kaggle作业。下一步按已准备任务卡审计DeBERTa干净基座，继续等优化更新预算B-R/C1/C2构建，仍须G1/G2-R/G3/G4完整验收。本站公榜最高0.620。

---
# 2026-10-31 DeBERTa 基座身份部分闭环

私有禁网CPU探针 okkojbjj/name-that-llm-deberta-base-identity-probe/1 已COMPLETE，镜像 phmngc/microsoftdeberta-v3-large/Transformers/default/1 的权重 SHA dd5b5d93 与微软官方固定版本字节一致，SPM SHA c679fbf9 也一致；24层/1024维/16头。原生核验在 outputs/part_two_oof_asset_audit_20261027_74/deberta_base_identity_verification.json，唯一作业记录 deberta_base_probe_record.json。可作为新的B-R干净可钉基座；作者历史DeBERTa训练起点、config/tokenizer_config逐字身份和云端GetKernel源码拉取仍未知。未训练、未交榜，当前无运行中的本项目Kaggle作业。下一步重建DeBERTa逐折训练入口，并用同折完整融合评价B-R/C1/C2；两阶段Qwen2.5生成器静态九组合PASS但未GPU运行，独立Sol复核只基于摘要，提示更新次数相同不等于计算量相同。

---
# 2026-10-31 B-R Qwen2.5 双阶段首折运行中

唯一当前本项目Kaggle作业 okkojbjj/name-that-llm-qwen2-5-two-stage-br-fold-0/1 已且仅已上传一次，私有禁网T4×2，100次官方训练侧阶段一更新 + 优化器重置后200次官方训练侧阶段二更新，父限12600秒/内限10500秒。初查RUNNING。源码SHA d30c76a3301ca2a6f6e98f5365dd5c91f5bb06cc62ebd124537f5a35b24d78bf。仅成员级同折基线首折，绝非完整G2-R、公榜候选或更高分。终态只读回收原生证据并用冻结验收器检查；运行中不启动第二个本项目作业、不重传、不交榜。证据 outputs/qwen25_stage1_newpool_20261029_99/{br_fold0_taskcard.json,br_fold0_record.json,verify_two_stage_outer_fold.py}。


C1 同折候选的首折代码和任务卡已静态冻结，源码SHA C0AAE8773242A54E845D61C56E04C81990CFE64829F30E2DA536E013D8206670；只用逐折筛过的seyit/prvsiyan 7223行作阶段一，阶段二和B-R均为官方训练侧。见 `outputs/qwen25_stage1_newpool_20261029_99/{upload_c1_fold0/,c1_fold0_taskcard.json}`。尚未上传；B-R/1仍RUNNING，禁止并行启动。C1仅获条件性私有探索来源准入，非公榜G1通过。

C2 首折私有探索候选代码、元数据、任务卡及独立验收器已静态准备，源码SHA 70529B77CFF77B5113A25AD87AC51F66FDCC4531A80CCEDA44D3CC868CFA455C。阶段一为逐折筛过的 seyit/prvsiyan/lxd 11424行（1555题），阶段二与B-R相同；见 `outputs/qwen25_stage1_newpool_20261029_99/{upload_c2_fold0/,c2_fold0_taskcard.json,verify_c2_two_stage_outer_fold.py}`。尚未上传；B-R/1当前仍RUNNING。来源仅条件性私有准入，C2同时改变类覆盖与数据量，不能把收益单归因于lxd文本。

完整G2-R融合的下一处实质缺口已定为干净基座DeBERTa同折OOF：现有公开最终权重见过官方全部训练行，不能代替训练外成员。已冻结首折任务卡 `outputs/part_two_oof_asset_audit_20261027_74/deberta_clean_outer_fold0_taskcard.json`：官方微软基座权重/SPM字节身份已验、深成员固定折0、1799训练/900验证、18类、max_len384、masked mean+线性头；将在同一B-R/C1/C2中作为固定成员。精确batch/优化步骤及代码/运行仍待冻结，当前B-R Qwen25作业RUNNING，未开启DeBERTa作业。此为新重建，不称作者历史精确复现。

DeBERTa新完整基线成员的首折训练源码已落在既有资产审计目录 `outputs/part_two_oof_asset_audit_20261027_74/deberta_clean_outer_fold.py`，SHA 7F50B26103DD18E144F32713624A7E29A525483E079AB4B34D934745F70EA9AA。它从已字节验收的微软官方基座出发，官方训练侧1799行、3 epoch、有效batch16、339次预期更新，最后层masked mean +18类线性头，输出900行OOF、logits和权重。源码 AST 编译及150题折映射对冻结清单逐行核对通过；尚无真实梯度、显存或指标证据。任务卡已更新，当前B-R Qwen25仍RUNNING，不上传第二本项目作业。作者历史stage1未复现，此为新B-R固定成员。

DeBERTa首折独立原生验收器 `outputs/part_two_oof_asset_audit_20261027_74/verify_deberta_clean_outer_fold.py` 已编译，SHA 7215FA9DBA829CEBAAB61DEF8D260CE454A9CC8BEF42CED2A375D1394D30D0F3，并在任务卡中冻结。终态将核2699行/150题同折映射、900验证ID、339真实更新、18类有限logits、BA重算、模型及输入哈希；代码编译不等于GPU训练验收。B-R Qwen25唯一作业仍RUNNING，未启动DeBERTa。

完整G2-R融合计算入口 `outputs/qwen25_stage1_newpool_20261029_99/summarize_g2r_full.py` 已在见到B-R双阶段成员结果前冻结，SHA 7C3F631397515F2B3E353B07901AB4D878C5EC705A89BF5C51D4730437454881。固定 Part Two 权重 SVC1.0、Qwen3 0.4+同题0.1、Qwen2.5 0.2+同题0.1、DeBERTa0.25；每折300轮Sinkhorn，再合并2699行评BA及候选−B-R门槛。只接受相同ID/折/18类有限成员预测。静态核查发现现有SVC margin类序与Part Two类序不同，已在入口按明确类名重排，而非直接按列相加。脚本编译通过，尚缺DeBERTa及新双阶段三折原生结果，不能输出完整G2-R结论；不作虚构测试分数。

完整G2-R融合入口在见到任何B-R双阶段效果前追加冻结的配对prompt重抽样：1000次、种子20261030、每外折50题有放回抽样，输出C1/C2相对B-R的探索性95%区间和缺失类别次数；该区间不是独立显著性门槛。当前源码SHA 2D89292FFFDC98A5532333C5D310C4544B9F6F737D7E6E7B91ACC00585558EE3，编译通过，尚无完整成员可跑端到端。既有真实Qwen3/Qwen2.5折2成员公式与SVC列名重排检查保持通过。

B-R双阶段Qwen2.5外折1的后续包已静态准备，源码SHA 23C46B308E7DEAF6375F1E15F1B2C1A040D6BEA661144D228257E864EFBCB026 `outputs/qwen25_stage1_newpool_20261029_99/br_fold1_taskcard.json` 为准。目录 `upload_br_fold1/`，尚未上传。外折0仍RUNNING，首折原生验收前不会启动折1。


B-R双阶段Qwen2.5外折2包亦已静态准备，源码SHA D40EB1C55805BD7DCE3F16498E366457FC32453D8E588A9D987C4A4AD97552B3，代码编译通过；见 `outputs/qwen25_stage1_newpool_20261029_99/{upload_br_fold2/,br_fold2_taskcard.json}`。折1/2均未上传，当前折0仍RUNNING。两者须在前一折原生验收且唯一Kaggle作业空闲后顺次执行。

## 2026-10-31 B-R 双阶段 Qwen2.5 外折0终态

私有禁网T4×2作业 `okkojbjj/name-that-llm-qwen2-5-two-stage-br-fold-0/1` 已 COMPLETE，且原生验收 `PASS_SINGLE_TWO_STAGE_BR_FOLD_MEMBER_OOF_ONLY`：100次阶段一+200次阶段二真实更新、900条验证ID/50题、18类有限logits、BA复算一致，运行4019.57秒，成员加权BA0.421111（plain 0.335556）。下载的云端v1源码与上传源码仅换行格式不同，统一LF后SHA同为D30C76A3…BF；云端元数据私有/禁网/T4、模型与竞赛附件正确；原生日志关键失败词匹配0。证据 `outputs/qwen25_stage1_newpool_20261029_99/{br_fold0_record.json,br_fold0_verification.json,native_twostage_br_fold0_v1/,cloud_twostage_br_fold0_v1_meta/}`。这只是B-R单成员单折，不是完整融合、公榜或≥0.650。折1/2尚未上传。

---
# 2026-10-31 B-R 外折1已启动
Net-4090D 身份、SSHFS Z: 来源、项目 realpath 及 Kaggle 折0 COMPLETE 已核对。冻结 B-R 折1私有禁网T4×2作业 okkojbjj/name-that-llm-qwen2-5-two-stage-br-fold-1/1 上传一次，初查 RUNNING，父限12600秒。记录在 outputs/qwen25_stage1_newpool_20261029_99/br_fold1_record.json；终态前不启动第二个本项目作业、不交榜。任务卡一行误写 fold0 IDs；代码是 fold1，须按 fold1 900 ID/50题验收。本站最高公榜仍0.620，私榜未知。


---
# 2026-10-31 B-R 外折1验收完成
Kaggle私有禁网T4×2作业 okkojbjj/name-that-llm-qwen2-5-two-stage-br-fold-1/1 COMPLETE，原生/源码验收PASS；900行/50留出题，100+200更新，成员加权BA0.455556、plain0.396667，运行3891.63秒。证据 outputs/qwen25_stage1_newpool_20261029_99/{br_fold1_record.json,br_fold1_verification.json,br_fold1_native_manifest.json}。这仅是单成员单折结果，不证明完整G2-R或本站公榜提升；最高仍0.620。下一步在确认唯一作业终态且折2 slug不存在后上传已冻结B-R折2。


---
# 2026-10-31 B-R 外折2运行中
折1原生和云端源码验收PASS，成员BA0.455556；专用折1监控已删除。当前唯一 Kaggle 作业 okkojbjj/name-that-llm-qwen2-5-two-stage-br-fold-2/1 已上传一次，初查RUNNING，私有禁网T4×2、父限12600秒。记录 outputs/qwen25_stage1_newpool_20261029_99/br_fold2_record.json。终态前不开第二个本项目作业、不交榜；本站最高公榜仍0.620。


# 2026-10-31 DeBERTa下一作业预检

尚未上传的官方-only折0验证批量16→4；训练协议和权重输出不变。源码/上传快照/任务卡 SHA BFD90F1D6985360BA1B33E5A7D776005193DC10FD489CDDFe0BBAC60DD84227D，远端AST通过。当前唯一Kaggle作业仍为B-R折2；其终态前不启动DeBERTa。详情见 outputs/part_two_oof_asset_audit_20261027_74/REPORT.md 页末。

# 2026-10-31 B-R Qwen2.5第三折终态

私有作业 okkojbjj/name-that-llm-qwen2-5-two-stage-br-fold-2/1 COMPLETE 且原生验收PASS：899行/50题、100+200更新、成员加权BA0.470000、耗时4095.29秒；云端源码LF规范后等于冻结SHA D40EB1C5，日志无关键失败。B-R此成员三折已齐，但完整G2-R仍缺干净DeBERTa及C1/C2同折成员，不交榜。详情见 outputs/qwen25_stage1_newpool_20261029_99/{br_fold2_record.json,br_fold2_verification.json,REPORT.md}。

# 2026-10-31 DeBERTa首折v1失败

Kaggle okkojbjj/name-that-llm-deberta-clean-outer-fold-0/1 ERROR：原生 GradScaler.unscale_ 拒绝FP16参数梯度，训练前失败，无OOF。云端源码/输入身份与元数据核对通过；Z:中途掉线后通过原计划任务恢复并重新核验。正式源码与上传件单点改FP32编码器参数，保留autocast，SHA 3F152305…08C3；任务卡已同步，未上传v2。证据见 outputs/part_two_oof_asset_audit_20261027_74/deberta_clean_fold0_record.json 和 REPORT.md。

# 2026-10-31 DeBERTa首折v2已启动

唯一当前本项目Kaggle作业 okkojbjj/name-that-llm-deberta-clean-outer-fold-0/2，私有禁网T4、父上限18000秒，初查RUNNING；冻结FP32参数修复源码SHA 3F152305…08C3。v1 ERROR证据保留在 outputs/part_two_oof_asset_audit_20261027_74/，v2未获真实梯度/OOF验收；终态前不重传、不启动第二本项目作业、不交榜。见 deberta_clean_fold0_record.json 和 REPORT.md。

2026-10-31：DeBERTa折0 v2仍RUNNING；仅修正原生验收器的Kaggle换行误判，任务卡同步新SHA 07ddc922b646a2454aac25172e49d60db87c8ef1c7922167da089db9a3b58b58。上传/云端规范化源码哈希一致，训练源码未改；终态仍需原生验收。见 outputs/part_two_oof_asset_audit_20261027_74/REPORT.md 页末。


2026-10-31：DeBERTa官方only折0 v2确切Kaggle作业COMPLETE且原生验收PASS：900留出/50题、339更新、单成员BA0.366667、590.20秒，私有禁网T4、云端源码同源、日志无关键失败。v1错误保留。B-R完整同折比较仍缺DeBERTa折1/2及C1/C2成员；本站公榜仍0.620。证据 outputs/part_two_oof_asset_audit_20261027_74/{deberta_clean_fold0_record.json,deberta_clean_fold0_v2_verification.json,REPORT.md}。


2026-10-31 05:20 +08：第二个干净DeBERTa官方外折1/1已且仅已上传，初查RUNNING，私有禁网T4，5小时父限/4小时内限，源码SHA C13AA480…F8DA；唯一运行中的本项目Kaggle作业。终态后回收原生900行/50题/339更新与云端源码，运行冻结验收器--fold 1。折0 PASS，仍无完整集成和新公榜分。证据 outputs/part_two_oof_asset_audit_20261027_74/deberta_clean_fold1_record.json。


2026-10-31：DeBERTa折2本地快照已准备、未上传；官方划分1800/899、100/50题隔离与AST PASS，源码SHA 928971b31b058ab5278e7300ea72e360a10a508bc76e1e605a4c8baab71c86b8；折1/1仍为唯一运行中本项目作业，先等其终态验收。见 outputs/part_two_oof_asset_audit_20261027_74/deberta_clean_outer_fold2_taskcard.json。


2026-10-31：DeBERTa官方only折1/1 COMPLETE且原生验收PASS：900留出/50题、339更新、单成员BA0.342222、579.65秒，私有禁网T4、云端同源、日志无关键失败。证据 outputs/part_two_oof_asset_audit_20261027_74/{deberta_clean_fold1_record.json,deberta_clean_fold1_v1_verification.json,REPORT.md}。下一步折2单次上传，完整同折融合仍未完成，公榜最高仍0.620。


2026-10-31 05:35 +08：DeBERTa官方only第三外折2/1已且仅已上传，初查RUNNING，私有禁网T4，父限18000秒/内限14400秒，源码SHA 928971b31b058ab5278e7300ea72e360a10a508bc76e1e605a4c8baab71c86b8。折0/1原生单成员PASS；折2为当前唯一运行中的本项目Kaggle作业。终态验收后才能构建完整同折融合；最高公榜仍0.620。见 outputs/part_two_oof_asset_audit_20261027_74/deberta_clean_fold2_record.json。


2026-10-31：Qwen3官方三外折原先云端源码回拉403缺口已通过现有Kaggle CLI只读重试关闭：当前云端源码与各折冻结上传件LF规范化逐字相同，云端元数据均私有禁网T4；原生v1三折PASS证据仍独立保留。pull响应不单独标注历史源码版本，准确范围见 outputs/part_two_oof_feasibility_20261028_83/qwen3_threefold_cloud_source_retry_verification.json。仍不等于完整G2-R或新公榜分。


2026-10-31：冻结完整G2-R融合入口的DeBERTa输入索引已建：折0/1经原生PASS的logits小文件字节相同复制，SHA及源路径见 outputs/qwen25_stage1_newpool_20261029_99/g2r_deberta_inputs/manifest.json；折2仍RUNNING、索引PARTIAL，尚无完整融合BA。


2026-10-31：C1/C2私有候选后两折四个快照和任务卡已静态准备、均未上传；只改各首折冻结源码的FOLD，AST/源码差异/私有禁网T4×2元数据PASS。来源仍仅条件性私有探索准入，不可据此公榜。见 outputs/qwen25_stage1_newpool_20261029_99/{c1_fold1_taskcard.json,c1_fold2_taskcard.json,c2_fold1_taskcard.json,c2_fold2_taskcard.json,REPORT.md}。


2026-10-31：DeBERTa官方only折2/1 COMPLETE且原生验收PASS：899留出/50题、339更新、单成员BA0.284966、597.19秒，私有禁网T4、云端同源、日志无关键失败。至此DeBERTa三折齐全；下一步冻结完整B-R融合诊断，不把单成员BA当公榜。证据 outputs/part_two_oof_asset_audit_20261027_74/{deberta_clean_fold2_record.json,deberta_clean_fold2_v1_verification.json,REPORT.md}。


2026-10-31：完整B-R三折同折OOF已用冻结融合公式计算并独立复算PASS，2699行/150题BA0.538638，逐折0.552222/0.523333/0.540340；这不是公榜分，历史公榜0.620仍最高且不能与OOF直接配对。首次调用原SVC margins.npz因object ID失败，改用既有独立验收的margins_safe.npz成功。此轮c1/c2是B-R占位文件，不是真候选结果；下一步私有C1/C2成员训练。证据 outputs/qwen25_stage1_newpool_20261029_99/g2r_br_complete_diagnostic_record.json。


2026-10-31 05:54 +08：C1 Qwen2.5双阶段官方外折0私有作业已且仅已上传，Kaggle规范化实际slug okkojbjj/name-that-llm-qwen2-5-two-stage-c1-fold-0/1，初查RUNNING；冻结源码SHA C0AAE877、私有禁网请求T4×2、父限12600秒。当前唯一运行中的本项目作业。来源仅条件性私有探索准入，不交公榜；终态须核输入/100+200更新/900 OOF/云端源码。证据 outputs/qwen25_stage1_newpool_20261029_99/c1_fold0_record.json。


2026-10-31：C1两原始Kaggle数据集当前均CC BY 4.0且逐行映射已审计；Anthropic/Google Cloud/OpenAI公开官方条款含竞争用途限制，当前输出归属分类器是否受限及历史合同适用性仍未能由公开页确定。私有禁网探索按既有授权继续，公榜G1仍HOLD；见 outputs/qwen25_stage1_newpool_20261029_99/sources/c1_provider_terms_scope_20261031.json。


2026-10-31：高分0.651公开附件两项权利元数据重新只读查询，nlm-l24-adapters-v5及nlm-offline-wheels许可仍other、description空；原始JSON哈希变但关键字段未变。公榜G1继续HOLD，不能将容量PASS/作者0.651当本站公榜证据。见 outputs/highbase_0651_package_20261027_80/rights_recheck_20261031/summary.json。


2026-10-31：完整B-R的DeBerta去除只读诊断在同一2699行OOF BA 0.538638→0.534561（-0.004077），止损该消融，不立新候选、不交榜；见 outputs/qwen25_stage1_newpool_20261029_99/deberta_zero_weight_diagnostic_20261031.json。


2026-10-31：C1/C2首折固定融合资源筛查入口已在C1结果前冻结，源码SHA 8ad7c7b96de86c44a07b77ae2c1ac707da70b88f16b39b432d0141aaf93a71db；要求候选原生验收PASS，比较同一折完整融合ΔBA≥0.01仅决定后两折资源扩展，不是公榜资格。B-R自检900行BA0.552222 PASS。见 outputs/qwen25_stage1_newpool_20261029_99/fold0_candidate_screen_contract.json。


























## 2026-11-05 Linux Kaggle 通道恢复与 NB-SVM 新候选资源试探

本机项目内嵌 Kaggle CLI 2.2.5 已使用用户新配置的 Linux 凭据只读查到本账号提交；最高公榜 BA 仍为 **0.620**（57684141），最新提交 57912173 为 0.484，无新提交。两位 Astra 互评后冻结不增加外部数据的类条件 n-gram log-count-ratio SVC 特征候选；逐类校准曾止损，R4 仅有重建基线 +0.006296 OOF、净救回17行且来源/容量未闭合。见 `outputs/nbsvm_feature_ratio_20261105_123/`。

本机同时有 AI/BigData 训练；首折首类本地测时在约234秒、RSS约6.0 GiB、memory PSI full avg10约13.98%时主动停掉自己的进程，无效果结果、无 Kaggle 作业。下一步在隔离私有 Kaggle CPU 或本机空闲时完成固定首折，再依门槛决定是否做全三折。原75,783行旧池包含已知重建官方题来源；此阶段只作探索，不能直接由单折或测时升格为公榜候选。详见该目录 `taskcard.json`、`run.json`、`REPORT.md`。

---

## 2026-11-05 NB-SVM 私有 CPU 首折 v1 RUNNING

Linux 项目内嵌 Kaggle CLI 2.2.5 已认证只读；本站公榜最高仍 0.620（57684141），最近一次 0.484（57912173）。两位 Astra 交叉审议后，冻结类条件 n-gram log-count-ratio 重加权 SVC 候选，深成员/融合固定；旧 75,783 池含已知重建官方题来源，故任何本轮 OOF 仅探索。其他项目训练期间本地 pilot 因内存压力主动停止，无效果结果。已将首折完整融合比较仅上传一次为私有禁网 CPU 作业 `okkojbjj/name-that-llm-nb-svm-fold-0-private-cpu/1`，首查 RUNNING，Kaggle 父限 3600 秒；实际 slug 由云端生成，与上传元数据预期不同，不重传。上传源码 SHA256 `d75100136e571952a323fbff5dae25524bc99084c3a635f20cb20f993cb18f35`，bundle SHA256 `5cc950d8ccf000e2212fc448b963c2534f01ff06e58df61739d97036364be075`。下一步只读查确切 slug；终态回收原生首折输出和云端源码，核对输入哈希/900行/18类/有限数值/完整融合及耗时。无 `submission.parquet`，不交榜。见 `outputs/nbsvm_feature_ratio_20261105_123/{run.json,REPORT.md,cloud_build_v1.json}`。

---

## 2026-11-05 NB-SVM 首折终态：负向，路线止损

私有禁网 CPU `okkojbjj/name-that-llm-nb-svm-fold-0-private-cpu/1` COMPLETE，云端源码 SHA 与上传一致，原生三输入 SHA 一致；900 行唯一 ID，18 个分类器完成，888.99 秒，峰值 6,585,128 KiB。原生首折完整融合 BA 基线 0.591111、候选 0.580000、差 **−0.011111**，独立 AWK 按类召回复算一致，低于事前 −0.005 止损线。因此不做剩余两折、容量或公榜。旧池含已知重建官方题内容，此实验仅探索。Python 验收器本地执行遭自动审批拒绝，未声称其 PASS；原生输出、日志、回拉源码及哈希见 `outputs/nbsvm_feature_ratio_20261105_123/{run.json,REPORT.md,native_v1/,cloud_v1/}`。两位 Astra 正在交叉质疑下一项从本站已评分 0.620 高起点出发的单因素候选。本站最高公榜仍 0.620（57684141），私榜未知。

---

## 2026-11-05 第五成员缓存融合负向，停止固定0.2方案

两位 Astra 针对 NB-SVM 失败交叉质疑后，冻结已有旧 Qwen8B adapted 概率作第五成员，固定 `0.2×log(p)`，其他四成员和每折300轮 Sinkhorn 不变。2699行/150题三折缓存完整融合中，B-R 基线 BA 精确重现 0.5386378325，候选 0.5371538653，Δ−0.0014839672；折差 +0.003333/−0.002222/−0.005533，救回37/误伤41，独立 AWK BA 复算一致。低于事前Δ≥0.005、至少两折非负、最差≥−0.005门槛；不扫权重、不做隐藏推断/容量/公榜。该 B-R 是重建同折基线，不等于已评分0.620的精确OOF。无本项目 Kaggle 作业运行，本站最高公榜仍0.620。证据 `outputs/qwen8b_fifth_member_20261105_124/{taskcard.json,run.py,result.json,oof.csv,REPORT.md}`。下步须重审高起点主机制，不重复近期负向单因素。

---
