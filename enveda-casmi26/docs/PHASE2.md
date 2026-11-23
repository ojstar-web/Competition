# 阶段二：候选分子式审计与 MIST CPU 适配

日期：2026-10-23。沿用 net-4090d-jyx → net-4090d / Net-4090D 的已核验项目与普通 SSH，写入前核对了 SSHFS 来源、严格主机密钥检查、`ubuntu` / `yuanyn`、项目 realpath 和路径；未改邻近比赛。2 计算线程、12 GiB 虚拟内存、单次 900 秒，无 GPU、训练或 Kaggle 提交。阶段一固定划分与独立 COCONUT 候选原件均未改变：split SHA256 `fbe2195cbad63b66080415d7061dd07195a0bf2cbad0e6ae28bb2efcb5c11e99`，candidate SHA256 `5367ba42472fcdfe299bbd1b5b61d3935459371e543c3b24f5cb9185ba650bd9`。70 个 sealed 结构仍未推理或评分。

## 固定输入与模型

`outputs/phase2_20261023_01/formula_audit.json` 对 64 个 fresh_dev 开发结构的最终质量池审计：0/1/2/3/4+ 种中性分子式的结构数依次为 0/11/11/8/34；没有空池。公式仅由冻结候选的代表 SMILES 经项目官方规范化流程和 RDKit 计算，未使用 NP 真值分子式；参与最终池的候选没有显式同位素。`_02/model_input_manifest.json` 在评分前冻结，SHA256 `1cfb2329787fe4febc7e744a49d44887e2889a23cec75a0c56aa2fc0e7c190df`。可用分支固定为 q006、q015、q033、q036、q043、q044、q045、q059、q060：9/64 个结构，共 20 条 positive `[M+H]+` 谱。另 2 个单式结构无该加合物，53 个结构分子式歧义；这 55 个保持原 mass_only 排序。候选池始终用原全部 adduct 查询谱的中性质量中位数与 ±max(0.003 Da,15 ppm) 过滤，未因为模型只编码 H+ 重新定义质量池。分子式是中性母体式，MIST 内部处理质子化；模型 labels 不含 SMILES、InChIKey、结构名称或真值式，`compound` 仅为固定 qid。训练表的 SMILES 只用于将已有 NP 谱归入事先冻结的 qid。

MIST 使用官方 `main_v2` commit `126649324655f00f8a8707f75a0c0492b11cbb4f`，源码压缩包 SHA256 `f27543aef17c51f739b82162f16e8eb7015348e1f378af4a9127f06d9bde8c84`，MIT 代码许可。仅在本项目副本的 `utils/__init__.py` 对缺失 Ray 的可选导入加 guard；其他缺依赖仍报错，模型架构、state_dict、神经计算和官方预处理未改，补丁前文件保存在 `_02/mist_utils_init_before.py`，补丁后 SHA256 `25b08687e75062a13ef7c74c57b1f341816593e1d40e0aaffab6bc6d2d94841c`。配套 Zenodo 8316682 指纹权重 `references/mist_fp_canopus_pretrain.ckpt` 为 60,750,958 字节，MD5 `949947d5af162d6aeefbf698262167e7`、SHA256 `a5a37e2b269f9a5ec483bd7fd2d3402257ce5f223270d534610fe41c7c4c5dea`；CPU strict state_dict 加载 missing/unexpected keys 均为 0，epoch 233、MistNet、Morgan4096。独立 `.venv_mist_cpu` 为 Python 3.10.12、Torch 1.13.1+cpu、Lightning 1.6.5、RDKit 2022.09.5；有效包清单 `_02/requirements_effective.lock` SHA256 `29fa08b6cc0fb765342149002666ed9d0b2c89faaa3f49e46b3230fff05a1799`，pip 24.0、setuptools 80.9.0。项目 `.venv` 的 RDKit 2026.03.3 继续承担公式、官方规范键与评分。9 池共 466 个候选，用同一原代表 SMILES 比较 Morgan radius2/4096bit，跨两版 RDKit 逐位相同 466/466；这只证明本批候选一致，不声称训练时 RDKit 逐位复现。

官方 MGF 预处理默认每谱最多 60 峰、强度阈值 0.001，先最大值归一再开方；子式指派 20 ppm、1 worker；模型 CPU/no_grad、data loader 0 worker、torch intra-op 2/inter-op 1。固定 20 谱处理后各有 35–60 峰、23–57 个匹配子式，均非空。原始 `timsTOF` 在上游仪器字典映射为 unknown；该 checkpoint `embed_instrument=False`。全部 20 个预测为有限、非零的 4096 维指纹。候选向量用 MIST 官方 Morgan4096 对原代表 SMILES 编码；同一候选与每条支持谱算 L2 cosine 后跨该结构的支持谱取均值，规范键升序破同分。完整性断言核对 20 个独立谱名、每 qid 谱数、全部 9 个候选池的键与 SMILES、4096 二进制位与输入清单。

## 冻结预测后的开发集结果

预测先写 `outputs/phase2_20261023_05/predictions.json` 并冻结 SHA256 `f3afc71853e08218e14007fb55ac20673de79d7e299d5b69b0dd421277c4ef01`，之后 `_06` 才加载真值按官方分子级规则评分。正式源码 `src/mist_phase2.py` SHA256 `ebe650f460a67291dc2efe1c6fd414f0fce5797cf6dc85ef21dea6604a611177`；模型输出 pickle SHA256 `cbbe9b4474470506f370f5535531d527af6514544d58eddefb098daae7bf4899`。其余 55 个 qid 的 hybrid 与 mass_only top25 列表逐项完全一致；64/9 个评价键分别与冻结清单一致。

| 同一场景、同一候选池 | 分子数 | MRR@25 | Top1 | Top5 | Top25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 个可用结构：mass_only | 9 | 0.10159218 | 0/9 | 1/9 | 6/9 |
| 9 个可用结构：MIST | 9 | 0.87037037 | 7/9 | 9/9 | 9/9 |
| 全 64：mass_only | 64 | 0.16904619 | 3/64 | 17/64 | 44/64 |
| 全 64：9 MIST + 55 mass_only fallback | 64 | 0.27715563 | 10/64 | 25/64 | 47/64 |

9 个结构的 mass_only→MIST 名次：q006 14→1，q015 2→1，q033 22→1，q036 23→2，q043 9→1，q044 >25→1，q045 >25→1，q059 >25→1，q060 7→3。全 64 的 0.1081 MRR 差值完全来自固定 9 个结构，不代表运行了 64 个 MIST 预测。该结果仅支持把 MIST 保留为“单分子式、H+、外部结构库候选重排”的开发候选；已知谱参考条件的 S1 A 分箱检索仍是其场景默认方法，两个场景的数字不能直接互比。9 例按可用公式和加合物筛选，且模型预训练结构/谱与本比赛重叠未知；不能声称隐藏集提升、未见结构泛化、稳定模型或新结构生成。

## 资源、失败记录与回退

正式全 20 谱子式处理 1.42 秒 / 峰值 RSS 285,368 KiB；推理 2.26 秒 / 677,136 KiB；候选排序 0.45 秒 / 151,064 KiB；真值评价 32.44 秒 / 3,314,352 KiB，均退出 0。公式审计约 240 秒。模型环境约 1,180 MiB、离线 wheelhouse 约 320 MiB、源码约 8 MiB，权重约 58 MiB，低于本轮新增 3 GiB 限额；下载累计含早期 dry-run 临时包的保守估算约 599 MB，非精确实测总量，低于 750 MB。实际远端 PyTorch 网络不可达后改从 Windows 同官方 URL 获取 Linux wheels/权重写入本项目，逐 SHA 核验并远端 no-index 安装；未换镜像或改系统网络。初次安装的 setuptools 新版缺 `pkg_resources`、随后缺 pathos/multiprocess，按审查许可分别固定 setuptools 80.9.0 和补齐小型运行依赖；保留失败证据与 `_02` 的 dry-run、wheel 来源及哈希。代码 MIT；权重 Zenodo 仅标 `other-open`，独立详细许可条款尚未核实，不推定可再分发；预训练重叠仍未知。

针对性聚合回归 2 passed；Astra 基于关键源码、20 谱完整性、冻结预测/输入哈希、逐结构名次和运行资源做独立科学复审后通过，未 SSH 或亲自重跑。停止使用新分支和 `.venv_mist_cpu` 即可回退到阶段一已验证的 `src/baseline.py`/S1 A 与 COCONUT mass_only；不要覆盖阶段一或本轮历史产物。下一步若要扩展覆盖，先审计多公式候选的无 oracle 公式假设及其错误率，再决定模型调用；70 个 sealed 继续封存。

推理与环境的事后汇总绑定另见 outputs/phase2_20261023_04/full_infer_record.json；原始 full_subforms.time、full_infer.time、预测 pickle 和官方子式 JSON 保留在同目录。


