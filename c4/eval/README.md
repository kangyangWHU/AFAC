# c4/eval — 评测脚本

新管线（规则 decompose → 合并检索 → 领域 prompt 一次答题，见 CAMPAIGN.md §10）的评测。

| 脚本 | 作用 |
|---|---|
| `score_dev.py` | 全量跑 dev + exact-match 评分（按域 + high-conf）。默认本地 gpt-oss，增量存 `out/score_dev.jsonl`（可断点续） |
| `accuracy.py` | 给 `answer.csv` 按 `dev_labels.json` 评准确率（按域 + 置信度子集） |
| `calibrated_accuracy.py` | 校准准确率 |
| `grep_doc.py` | 文档内容检索小工具 |

**注**
- 都默认指向本地免费 gpt-oss（localhost:30000），只改 `config.load()` 内存副本，不动 `config.yaml`。
- silver 标签(`dev_labels.json`)有偏，high-conf 子集更可信；调参别只看 silver。
- 旧 agentic 诊断脚本（decompose_route / judge / miss_* / ab_decompose / decompose_eval）已随架构重构删除；`agent/agentic/loop.py` 保留作未来检索测试引擎。
