# c4/eval — agentic 流水线分步评测

每个脚本默认指向**本地免费 gpt-oss**（localhost:30000）；只改 `config.load()` 的内存副本，不动 `config.yaml`。
都用免费本地模型，可放心全量跑。

| 脚本 | 作用 | 产物 |
|---|---|---|
| `decompose_route.py` | 全题分解 + 路由，打印四类来源分布（pin-decompose / pin-idrouter / group-together / agg-split） | `out/decompose_route.json` |
| `retrieve.py` | 每条 fact 取一批(8)块，测 top1 BM25 分 + 关键词覆盖，统计**真检索失败** | 打印 |
| `judge.py` | 每条 fact 一批块 + 一次 `_judge`（无 re-query/synth），测 judge 抽值 **found 率（下界）** | `out/judge.jsonl`（增量、可断点续跑） |

**流水线**：先 `decompose_route.py`，`retrieve.py` / `judge.py` 读它产出的 `out/decompose_route.json`。

**注意**
- `retrieve.py` 关 `cache_retrieve`：BM25 检索缓存重建块 score=0（loop 只用顺序、不用分值），开缓存会污染分值分析。
- `judge.py` 慢（数百次 LLM）→ 建议后台跑；增量写 `out/judge.jsonl`，重跑自动跳过已完成的 key。
- 过滤了【情景人物 fact】(王某/李某…，违铁律5)——它们本不该是检索 fact。
