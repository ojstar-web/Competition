# 来源与后续核实

记录日期：2026-10-21。

## 官方入口

- [比赛主页](https://www.kaggle.com/competitions/name-that-llm)
- [概览与评估、时间、提交说明](https://www.kaggle.com/competitions/name-that-llm/overview/citation)
- [数据](https://www.kaggle.com/competitions/name-that-llm/data)
- [规则](https://www.kaggle.com/competitions/name-that-llm/rules)
- [讨论](https://www.kaggle.com/competitions/name-that-llm/discussion)
- [公开 Notebook](https://www.kaggle.com/competitions/name-that-llm/code)
- [排行榜](https://www.kaggle.com/competitions/name-that-llm/leaderboard)

证据边界：概览通过搜索索引取得正文，直接打开页面没有可读正文；数据与完整规则未取得，不把搜索不到视为不存在。README 中比赛事实来自官方概览，未用其他比赛补齐规则。

## 社区线索（待复核）

- [Wide Data Overview](https://www.kaggle.com/code/olegzholobov/name-that-llm-wide-data-overview)：供后续查看数据结构，尚未检查其代码。
- [The Prompt-Recovery Trap](https://www.kaggle.com/code/yangkuangou/name-that-llm-the-prompt-recovery-trap/log)：公开日志报告 2,699 行、150 个 prompt、18 个作者；这是社区运行输出，尚未用官方文件核验，不作为数据清单。
- [Name That LLM v5](https://www.kaggle.com/code/qasminumber/name-that-llm-v5)：公开基线线索，尚未检查代码或复现；不将网页分数作为本项目结果。

## 下一步（建议，未执行）

1. 用现有账号查看规则与数据清单，核对提交次数、团队限制、外部数据条件及数据许可。
2. 获取数据后，先核对列名、类别、样本量、缺失、重复和 prompt 身份；避免提前设计错误字段假设。
3. 若存在相同或近似 prompt 的多模型回答，优先评估按 prompt 分组的验证方案，检查随机划分是否高估泛化。
4. 再比较字符/词级 TF-IDF 等低成本基线，记录 Balanced Accuracy、各类召回率和混淆情况；有证据后再决定是否需要 GPU。
5. 实际开发沿用一套源码和清晰入口；数据、环境、模型及输出目录等到需要时再建。

本轮没有建立代码环境或使用 GPU，也没有复制其他服务器的文件。后续如重连或重挂载，写入前重新核验服务器与路径。


## 2026-10-21 后续进展

上面的“未执行”清单是初次收集时状态。现在已通过已认证 Kaggle CLI 取得官方页面全文（official_pages.json）及官方数据，完成 CPU 基线，详见 BASELINE.md。第三方 Notebook 的代码仍未审核或复现；本项目结果来自自己的 src/baseline.py。
