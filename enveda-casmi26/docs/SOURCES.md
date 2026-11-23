# 资料来源与核验状态

检索日期：2026-10-23。初期资料来自官方页面搜索索引；现已通过本机已认证的官方 Kaggle Python API 获取 11 个页面，原文保存在 official_pages.json。以下初轮待核实清单保留为历史，当前已确认事项见文末。

## 官方入口

| 页面 | 地址 | 用途及状态 |
| --- | --- | --- |
| 概览 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview | 已取得索引正文；任务、评分、日期、Notebook 限制 |
| 数据 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data | 已取得索引正文；字段、来源、文件、许可 |
| 规则 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules | 未取得正文 |
| 代码 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/code | 待查看具体公开 Notebook |
| 讨论 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion | 待检查官方公告与数据修订 |
| 排行榜 | https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard | 本轮未核验排名和成绩 |

未保存参赛人数、排名等易变统计。

## 报名前必须补齐

1. 团队人数、每日提交额度、最终可选择提交数、公私榜分配。
2. 资格、奖金领取条件、获奖方案交付及代码许可要求。
3. 外部数据及预训练模型的公开时间、可用性和许可细则。
4. 官方 metric Notebook 的真实链接、规范化流程及异常 SMILES 行为。
5. 测试数据是否有公开样例与隐藏替换、数据更新和谱库重叠说明。
6. 实际文件行数、字段、版本及 Kaggle 输入路径。

不把平台通用惯例当作本比赛已核实规则。

## 方法资源与参赛者线索

官方数据页提到 MIST-CF、SIRIUS、matchms、GNPS、MassIVE、DreaMS/GeMS、FragHub，可作为后续定向调研入口；尚未审计模型权重、依赖、许可或离线兼容性。

- [DreaMS 作者仓库](https://github.com/pluskal-lab/DreaMS)：谱表示学习资源入口，尚未评估适用性。
- [公开参赛者仓库](https://github.com/shiferaxa/casmi26)：描述候选检索与多证据排序，并报告域匹配验证的重要性。其性能与验证结论属于作者报告，本项目未复现。
- [公开数据记录](https://github.com/IreneGallini/Enveda_CASMI_2026/blob/main/DATA.md)：作者报告测试 1,213 条谱、400 个分子；未下载核对，不能当作本项目实测数量或隐藏测试规模。

以上外部仓库仅作线索，不作为官方规则或本项目性能证据；不直接复制其实现或信任其测试标签推断。

## 官方 API 核验更新

2026-10-23 已取得 data-description、rules、Evaluation、Timeline、Code Requirements 等页面正文，入口为 KaggleApi.competition_list_pages。未复制认证文件到项目或服务器。

- 团队最多 5 人，每日最多 5 次提交，最终最多选择 2 个提交；获奖许可类型为 MIT，详细义务以 rules 原文为准。
- 外部数据及工具须满足公开、同等可访问或规则的合理可访问标准；概览的离线执行约束仍适用。
- **可下载 test.parquet 是训练集样例，正式评分重跑时会替换为隐藏测试。** 当前 400 分子、1,213 谱不代表隐藏分子的身份。
- 隐藏测试分三类：公开谱库已有结构、公开结构库已有但无公开谱、PubChem 未收录的新结构；类别比例不公开。
- 隐藏测试允许 10 种加合物，包括三种失水形式；数据页字段说明仍残留“seven above”字样，应以上方完整列表为准。
- 官方 metric Notebook 已下载至 references/casmi-mean-reciprocal-rank.ipynb，只作参考。关键顺序为先去除立体化学，再进行互变异构规范化，最后比较 InChIKey14。
- 公私榜分配比例及官方输入实际挂载路径，本轮未确认。

官方评分入口：https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank 。
## 阶段一公开候选和预训练模型资源审计（2026-10-23）

以下是本轮实际审查的公开来源；只有 COCONUT CSV lite 进入本项目。出处 URL 保留到具体官方页面/仓库，尚未确认的条款不由代码许可推定。

| 资源 | 页面与实际观察 | 本轮状态 |
| --- | --- | --- |
| COCONUT | [官方 2026-10 下载页](https://coconut.naturalproducts.net/download) 标注 CC0，CSV lite 191 MB；[官方对象](https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv_lite-10-2026.zip) 实测 199,096,787 字节、SHA256 `a602502a4fa0950cc2e59eed690dccd082d1a051bf9f85ccda50a82dd22ca154`、ZIP CRC 通过；字段有 `canonical_smiles`、`exact_molecular_weight`、`molecular_formula`。 | 本项目 `data/coconut_csv_lite-10-2026.zip`；流式扫描及质量窗候选见 PHASE1.md。远端 DNS 不可解析官方对象，本机 Windows 对相同官方 URL 下载成功，未更换镜像或修改网络。 |
| MIST | [main_v2 仓库](https://github.com/samgoldman97/mist/tree/main_v2)、[MIT 代码许可](https://raw.githubusercontent.com/samgoldman97/mist/main_v2/LICENSE.md)、[官方快速运行步骤](https://raw.githubusercontent.com/samgoldman97/mist/main_v2/quickstart/01_run_models.sh)、[下载脚本](https://raw.githubusercontent.com/samgoldman97/mist/main_v2/quickstart/00_download_models.sh) 指向配套 [Zenodo 8316682](https://zenodo.org/records/8316682)：指纹权重 60.8 MB、MD5 `949947d5af162d6aeefbf698262167e7`。脚本会先按已给母体式分配碎片子式；这不等于自动推断母体式。[environment.yml](https://raw.githubusercontent.com/samgoldman97/mist/main_v2/environment.yml) 使用旧版 Python/RDKit/PyTorch Lightning。 | test.parquet 无 `molecular_formula`；现有 `.venv` 无 torch/Lightning，也无已审定的推理可用母体式入口。未下载权重、安装依赖或调用模型。旧 Zenodo 7897299 权重不可与 main_v2 混用。权重记录的许可字段未呈现明确条款；训练结构与本比赛谱库重叠程度未知。运行前须固定代码 commit、权重条款、formula hypotheses 来源与独立预算。 |
| DreaMS | [作者仓库](https://github.com/pluskal-lab/DreaMS)、[MIT 代码许可](https://raw.githubusercontent.com/pluskal-lab/DreaMS/main/LICENSE)；[Zenodo 10997887](https://zenodo.org/records/10997887) 的 embedding 1.2 GB 与 backbone 1.4 GB 合计约 2.6 GB。[API](https://raw.githubusercontent.com/pluskal-lab/DreaMS/main/dreams/api.py) 的 head 路径会在缺 backbone 时自动下载；公开接口只显示 embedding 模型入口。[DataFormatA](https://raw.githubusercontent.com/pluskal-lab/DreaMS/main/dreams/utils/dformats.py) 默认要求 +1 电荷且 m/z 不超过 1000。 | 超本阶段权重下载预算，未下载或调用；也未得到可直接做结构指纹排序的公开 head。权重许可字段与训练谱对本比赛的重叠未核实；不能从代码 MIT 推断权重条款。 |
| LOTUS | [原下载页](https://lotus.naturalproducts.net/download) 展示旧版 2021-02 导出，未见可靠包大小；[现行项目主页](https://lotus.nprod.net/) 表示数据 CC0，并说明旧门户不再维护。 | 仅版本/许可/体量可得性审计；没有下载第二个大型结构库或建立候选。 |
| PubChem | [官方下载说明](https://pubchem.ncbi.nlm.nih.gov/docs/downloads) 提供 Compound 全量及增量批量下载，内容来自众多提交来源，个别来源许可可能限制批量复用；全量规模超本阶段小库验证范围。 | 仅元数据/离线体量路径审计；未批量抓取。S3 人工去真值不代表 PubChem 外新结构生成能力。 |

公开仓库的分支网页可读，但本轮到 GitHub 的 `git ls-remote` 连接失败，未取得可验证 commit SHA；不声称已锁定模型代码。Zenodo 权重页未明确显示独立许可字段，待正式使用前逐项核实。上述模型均无本项目实测性能或 CPU smoke 结果。

## 阶段二 MIST 资源状态更新（2026-10-23）

阶段一上表的“未下载/未调用”仅描述当时状态。现已固定 [MIST main_v2](https://github.com/samgoldman97/mist/tree/main_v2) commit `126649324655f00f8a8707f75a0c0492b11cbb4f` 与配套 [Zenodo 8316682](https://zenodo.org/records/8316682) 指纹权重，本项目实测权重 MD5 `949947d5af162d6aeefbf698262167e7`、SHA256 `a5a37e2b269f9a5ec483bd7fd2d3402257ce5f223270d534610fe41c7c4c5dea`。同官方 PyTorch CPU/PyPI/Zenodo URL 的 Linux wheels 与权重由 Windows 获取并在项目内离线安装，远端网络失败记录仍留存；未改网络或使用 GPU/训练。`outputs/phase2_20261023_02/wheel_manifest.json` 及关联 dry-run 记录保存 wheel 来源与哈希，有效环境包清单为 `requirements_effective.lock`。代码 MIT；Zenodo 权重记录仅标 `other-open`，未确认其独立详细再分发条款，不能由代码许可代推。训练/预训练集与比赛数据重叠未知。正式 9 结构 CPU pilot 的方法、结果和资源见 [PHASE2.md](PHASE2.md)。DreaMS 仍未下载或运行。

## 阶段三资源沿用

剩余 76 个开发结构沿用同一 COCONUT 2026-10 CSV lite 原包及 MIST main_v2 / Zenodo 8316682 指纹权重；本轮没有新下载、镜像或依赖安装。候选只对本批查询质量宽窗命中的 CSV 行做官方规范化，窗口外未逐条规范化；模型权重独立详细许可和预训练重叠仍未知。新候选与运行证据见 [PHASE3.md](PHASE3.md)，原资源来源与许可边界仍按本文件前述记录。

## 阶段四资源沿用

多分子式审计和 FM 开发候选沿用 [阶段二已核的 MIST main_v2/Zenodo 8316682](https://zenodo.org/records/8316682) 权重及本项目独立 CPU 环境，并复用同一 COCONUT 2026-10 官方原包；没有新增外部数据、软件包、镜像或权重下载。代码 MIT 与权重仅标 `other-open` 的许可边界、预训练重叠 unknown 均未改变；本轮出处、哈希和运行范围见 [PHASE4.md](PHASE4.md)。

## 阶段五 MIST 实现来源补核

固定 commit 的 [quickstart/00_download_models.sh](https://raw.githubusercontent.com/samgoldman97/mist/126649324655f00f8a8707f75a0c0492b11cbb4f/quickstart/00_download_models.sh) 直接指向 Zenodo 8316682 指纹权重；项目固定压缩包 150 个普通文件与解压副本逐字节对照，仅已记录的 Ray 可选导入 guard 有差异，运行模块实际导入自该固定源码。作者仓库 [公开 issue #17](https://github.com/samgoldman97/mist/issues/17) 的第三方报告定位同一 attention mask 行，该旧仓库 issue 仍 Open、未见其中的作者回复/合并；作者新仓库 coleygroup/MIST 固定 commit 48e339be6535546c87ba4f6b81a0f787c2e4ac40 的 scaled/additive attention 已使用 new_attn_mask。新仓库模型、权重和运行栈不同，本项目未验证或自动替换。MIST 作者实现不是 Kaggle 官方推荐；用户授权后的本项目 fixed61 仅固定旧权重批次补位上下文，尚未修复其 attention。详细证据见 [PHASE5.md](PHASE5.md)。
