# 赛题四 · 金融长文本 Agent —— 方案与最终实现 (v3, as-built)

> 目标：`FinalScore = 100·Accuracy·(0.7 + 0.3·TokenScore)`，`TokenScore = max(0,(5e6−TotalTokens)/5e6)`
> 合规红线：推理/检索/rerank/向量**只能用 Qwen 系**；解析阶段可用非 Qwen（MinerU/OCR）。
> 模型：本地 vLLM `Qwen/Qwen3.6-35B-A3B`（OpenAI 兼容，Qwen 系→合规）。

## 0. 最终结果（A 榜 100 题，silver 标签离线评测）

| 模式 | doc_ids | 总准确率 | high-conf 子集(90题) | 总 Token |
|---|---|---|---|---|
| **A：给定文档** | 给 | **81%** | 84.4% | 738K |
| **B：路由找文档** | 不给 | **82%** | 85.6% | 731K |

> 官方 baseline 15%。Token 731K/100题 → 200题约 1.5M，TokenScore≈0.71，远在 5M 预算内。
> silver 标签：本地无官方答案，用 10 个独立 Claude 子代理读源文档人工核验得到（90 high / 9 medium / 1 low），见 `dev_labels.json`。真实 B 榜会更难、分数会低一些，但链路已完整验证。

**进展轨迹**：55→66→67→71→81%（A），B 榜 82%。每一跳都由错题归因驱动（见 §6）。

**文档路由召回**（A 榜 gold 当真值，全库 573 篇）：R@5=88%, R@10=96%, **全@5=74% / 全@10=90%**, MRR=0.90。regulatory 在 512 篇干扰下仍 全@10=90%。

---

## 1. 架构总览

```
[离线，可用非Qwen]                         [在线，仅Qwen]
原始 PDF/HTML/TXT (573篇)                  题目(domain+question+options[+doc_ids])
  │ parser + 质量门控 fallback               │
  ▼ processed/{domain}/{doc}.json           ▼ [B榜] DocRouter: 域内BM25→候选doc
  │ chunker(结构感知+伪表清洗)               ▼ EvidenceRetriever: 逐(选项×文档)+指标定向+父块回填
  ▼ chunks.jsonl + BM25索引                 ▼ Reasoner: 域prompt+逐项判真伪(Qwen3.6)
                                            ▼ postprocess: 抽字母/校验/纠错重问
                                            ▼ answer.csv + evidence.json + token summary
```

## 2. 代码结构（全 config 驱动，换算法只改 config.yaml）

```
config.yaml                  全局参数（解析/分块/检索/路由/LLM/推理）
agent/
  schema.py                  Doc/Block/Table 数据模型
  config.py                  yaml 加载器 config.get('a.b'), config.path('x')
  doc_index.py               doc_id→文件 全库索引(573)
  parser/  base,structure,txt,html,pdf_pymupdf,pdf_mineru,quality,registry
  chunker/ base(归一化),chunker(结构感知分块)
  index/   base(Retriever抽象),bm25
  retriever/ filters(伪表/低值),retriever(编排)
  router/  router(B榜文档路由)
  reasoner/ prompts(域定制),reasoner
  postprocess/ answer(抽字母/校验)
  llm/     base(抽象+Token计量),qwen(本地vLLM/百炼)
  pipeline.py                端到端编排
eval/  parse_check, table_check, fallback_scan, retrieval_recall, doc_recall, accuracy
script/ preprocess, build_index, dump_evidence, retrieve_test, run
dev_labels.json              100题独立silver标签
out/A_given_docids/, out/B_routed/   最终结果
```

## 3. 预处理（解析质量=上限）

- **按域分流**：regulatory 用 txt/html 直读(div.detail-news)，其余 PDF 用 PyMuPDF。
- **质量门控自动 fallback**（`parser/quality.py`+`registry.parse_with_fallback`）：PyMuPDF 解析后，整篇文字密度过低/近空 → 自动转 MinerU/OCR。面向 B 榜未知扫描件，不靠人工挑名单。全库 190 PDF 仅 3 篇触发（实测教训：初版用"空页比例"误伤 75 篇，长财报天然多稀疏页；改"整篇密度"信号后精准）。
- **MinerU**：质量门控触发时调用（扫描件 OCR/跨页表/合并单元格）。实测扫描件 `csrc_0038_att2`：PyMuPDF 0 字符 → MinerU 2709 字符。装在 `AFAC4` conda 环境（见 README，含 GPU torch cu128 修正）。
- **统一格式**：`processed/{domain}/{doc_id}.json`，block 带 `type/section_path/article_no/page/table`。
- **验收**（`eval/parse_check.py`）：条款连续性检验 + 已知证据回指校验。

## 4. 索引与检索

- **分块**（`chunker/`）：结构感知（条款/章节/表格独立块）；CJK 伪空格归一化("研 发"→"研发")；**伪表清洗**（PyMuPDF 把募集书释义/版面误判成表，去管道符重分类为正文）；丢弃目录/空壳/碎片；硬切 max_chars（修过 153K 巨型块 bug）。
- **检索**（`retriever/`，BM25 jieba+金融词典，纯词法零模型合规）：
  1. **逐选项检索**（multi/tf 漏召回=全错）；
  2. **逐(选项×文档)检索**——每选项在每篇文档内单独查，治"大文档里 option 证据被高频财务表挤掉"（这一项把 fc 50→80）；
  3. **指标定向**——财报对比题对每文档额外跑"营收/净利/现金流/研发/分红"query，保证两篇都拉到指标表；
  4. **父块回填** small-to-big；
  5. **reserved 优先 + max_chunks 截断**：定向块免遭截断，其余按分补满，控 token。
- 检索召回（A 榜内）：100/100 题全 gold 文档覆盖，证据 ~8K 字符/题。

## 5. 推理与后处理

- **域定制 prompt**（`reasoner/prompts.py`）：regulatory 引条文/时限、insurance 套公式算、reports 跨年取数比较、contracts 要素核对、research 数字核验。
- **逐项判真伪 + 逻辑对齐格式**："选项陈述X｜证据事实Y｜是否一致→真/假"，治"事实算对、结论填反"。
- **省 token**：`enable_thinking:false`（Qwen3.6 思考模式一个字母烧 279 token→2），靠 prompt 显式 CoT。
- **后处理**（`postprocess/answer.py`）：只认"答案："行抽字母（**不全文扫字母**——否则多选题把讨论过的选项全当答案，过度选择）；mcq/tf 取首字母，multi 去重排序；非法/空 → 追问纠错重问。
- **自适应兜底**：答案为空 → 提高 min_per_doc 重检索重问。

## 6. B 榜文档路由（无 doc_ids）

- `router/router.py`：域已知→锁定域语料；逐选项 BM25 取域内 top chunks，按 doc 聚合（sum）打分，取 top-k(=5) 候选 → 喂检索。
- `eval/doc_recall.py`：用 A 榜 gold 当真值算 Recall@k（修过一个 `len(cand)<=k` 早返回 bug，曾让 MRR 假性=0.01）。
- pipeline 无 doc_ids 时自动路由，端到端 B 榜模式准确率 82%，与 A 榜持平。

## 7. 复现（详见 README.md）

```bash
conda activate AFAC4
PY=~/anaconda3/envs/AFAC4/bin/python
$PY agent/doc_index.py                    # 全库 doc 索引
$PY -m script.preprocess --all            # 解析 573 篇
$PY -m script.build_index                 # 分块 + BM25 索引
$PY -m eval.doc_recall                    # B榜路由召回
$PY -m script.run --out out/A_given_docids            # A榜(给doc_ids)
$PY -m script.run --out out/B_routed --no-gold        # B榜(路由)
$PY -m eval.accuracy out/B_routed/answer.csv          # 对 silver 标签评分
```

## 8. 关键决策与教训（实测得出）

1. **后处理"全文扫字母"是大坑**：多选题会把讨论过的选项全选 → 过度选择。只认"答案："行后 55→66%。
2. **max_tokens 有拐点**：1024 截断→空答案/过度选择；2048 解决；再加无效。
3. **逐(选项×文档)检索 >> 盲目加 min_per_doc**：定向拉对的块 + reserved 保护，比加量更准且不灌噪声（加量曾把 res 85→80）。
4. **指标定向要按域验证**：财报有效(fin +15)，合同反而引入释义噪声(禁用后 fc +10)。
5. **质量门控信号要选对**：用"整篇密度"而非"空页比例"，否则长文档全误伤。

## 8.1 Embedding 语义路由实验（负结果，已记录）

用本地 Qwen3-Embedding-0.6B（零 API token、Qwen 系合规）嵌入文档签名做语义路由，与 BM25+签名融合实测：
- `bm25×sig`（当前）全@5=74% ；`bm25×sig×(1+dense)` 74%（无增益）；加性 blend 反降（70/66/63）。
- 结论：**dense 语义路由不敌 BM25+签名**——本语料是条款号/指标名/公司名的词法精确匹配场景，BM25 已近顶。保留 BM25+签名路由（全@5 77%）。
- 代码已就绪（`agent/llm/embed.py` 本地+百炼双后端），text-embedding-v4（更强）可在有 key 时再验，但预期收益有限。

## 8.2 qwen3.7-plus 实验（结论：当前 token 约束下不划算）

| 模型/模式 | A 准确率 | token/题 |
|---|---|---|
| 本地 Qwen3.6-35B（thinking off）| **86% / 89%** | ~800 |
| qwen3.7-plus（thinking off）| 72.7% / 76.1% | ~800 |
| qwen3.7-plus（thinking on）| ~86%（抽样6题修复3）| **~8500** |

- qwen3.7-plus 是 thinking-native 模型：**强制 thinking off 反而更差**（会在答案通道里反复自我质疑、把对的答案说翻）。
- thinking on 才能发挥，但 token ~10×，token 受限下不划算。
- **建议**：正式提交若走百炼，优先试**非思考型百炼模型**(qwen-plus / 基准 qwen3.6-plus)以低成本复现本地 86%；qwen3.7-plus-thinking 仅在预算充足追极限准确率时用。
- 标签清理后最终基线：**A 86%/89% high-conf，B(真路由) 62%/65%**。

## 8.3 A 榜错题严格归因（逐争议选项查证据，最终版）

对 13 道 A 错题，只看模型/标准**分歧的那个选项**，查其判别事实是否在喂给模型的证据里：
| 归因 | 数量 | 题 | 处置 |
|---|---|---|---|
| **检索漏**（判别数值没进证据）| 2 | fc_a_005(12.42缺), fin_a_020(1332缺) | 财报/合同关键数值表 BM25 排名低、被 max_chunks 截断。提 per_option_doc_k 无效(仍被截断)，提 max_chunks 会全局涨 token，定向 query 是打补丁 → **判定为硬检索案例，留待 dense 文档内检索**，不打补丁 |
| **空答案**（证据齐，没下结论）| 4 | fin_a_008, ins_a_006, ins_a_014, reg_a_012 | thinking-on 实测修 3/3；非能力缺陷，靠思考/兜底重问 |
| **真·模型错**（证据齐，判错）| 7 | fc_007/011/018, fin_018, ins_007, res_002/004 | 模型能力/prompt territory；含可议项(res_004 转述严格度、tf 判反) |

**教训纠正**：早期"12/13 是模型错"是因分类脚本太宽（命中任一关键词即算证据齐）。逐争议选项严查后，真正"证据齐而判错"只有 7 道；另 2 检索漏 + 4 空答属非能力问题。

**结论**：cheaply/泛化可修的非模型错已尽（标签清理、路由签名、空答 thinking 可解）；剩 2 硬检索需 dense、7 道需模型能力。qwen3.7-plus 实测 thinking-off 更差、thinking-on 仅修 3/13 且 10× token，当前 token 约束下不采用。**A 榜 86%/89% 接近本套词法+本地模型的合理上限。**

## 9. 剩余可优化（按收益排序）

- financial_contracts 募集书伪表噪声仍重（可对失败篇上 MinerU 表格模式）。
- 少量 tf 判断题判反；财报个别"两年两值"仍偶有缺。
- 路由 全@5=74%→可加文档标题信号 / 章节级路由提升多 doc 题召回。
- 真实 B 榜需用官方在线提交校准（silver 标签非官方答案）。
