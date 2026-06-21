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

---

# v4 重构：Agentic RAG（规划 + 进度）

## 进度（2026-06-20）
- ✅ **阶段1 面包屑+outline**：`Chunk.breadcrumb`(只进展示不进索引)、`agentic/outline.py`、`index/outlines.json`(573篇)。
  重建后 BM25 token md5 字节一致 → **检索零回归**(基线准确率不受影响, 无需重跑LLM确证)。
  例: ins_a_001 的 160% 块展示从 `[2|第五条]` → `[2|国寿增益宝终身寿险(万能型)(2025版)条款›第五条]`。
  名字覆盖: insurance 12/16、regulatory 483/513 好; contracts/research/fin报表多为空(安全回退定位)。
- ✅ **阶段2 分解器**：`agentic/decompose.py` + `eval/decompose_eval.py`。100题离线评测:
  archetype 分布合理(option_verdict 91 / value_compare 4 / single_fact 5, **fallback 0/空 0**);
  多选覆盖 **76/76=100%**(每选项一子问题); value_compare 正确按实体拆(ins_a_001=4实体);
  doc_hint 绑定率 **57%**(受 outline 名字覆盖限制, None=安全全搜); token 721/题。
  **待改进**: outline 名字空的 doc(research/保险阅读指引/财报)→ doc_hint 绑不上, 不阻塞但浪费检索范围。
- ✅ **阶段3 子问题 agentic loop**（先 value_compare）：`agentic/loop.py`（无状态: genQuery→BM25→judge读块吐值/否, 换词不退出, 上限3轮）。
  ins 4 题 12 子问题: **命中 10/12=83%, ~1835 token/子问题**(与旧管线/题相当)。
  **实测决策**: ① backfill 回填**关**(净负: 2x token 且命中 83→75%, 失败是解析丢失非切散); ② 实体名 doc 缩窄**开**(去跨产品污染)。
  **关键发现**: loop 是"诊断器"——干净条文块(增益宝/鑫享/众安/e生保)近乎全中且省token; 失败集中在 **doc1 智盈金生**(墙文本阅读指引, 无条号结构, 解析丢规则), 跨 ins_a_001/002 系统性复现; doc16 富鸿身故金条款解析丢失。旧管线靠"塞22块+LLM跨产品泛化公式"蒙对, agentic 精确读块反而暴露上游解析缺口 → 这些子问题 found=false, 按设计降级旧 reasoner(阶段4)。
  **上游待修**: doc1 类阅读指引墙文本的条文切分; 部分险种身故金条款解析丢失(疑表格/版面)。
- ✅ **阶段4 合并+降级**：`agentic/merge.py`(value_compare 排序比对 + option_verdict 收真/add-only) + `agentic/solver.py`(编排+降级)。
  merge 单测: 喂 gold 值→ ins_a_001=B✅ ins_a_002=A✅ defective→None降级✅。ins 4 题端到端 4/4(1 走 agentic, 3 降级)。
  **成本发现**: 降级"双付费"(先跑 agentic 再退旧 reasoner) → value_compare 4 题 ~2.6× 旧 token; 准确率没提(旧本就对); **唯一赢=可审计**。
- ✅ **阶段5(打通) option_verdict + 端到端入口**：`loop.solve(shape=verify)` + `script/run_agentic.py`(answer.csv + agentic_audit.json 可回放)。
  **分层 20 题端到端**: high-conf **15/17=88.2%**(旧 89%, 持平无回归); 路径 agentic 9/降级 11; token ~11.4k/题(≈2.3×旧)。
  错题 3 全是 option_verdict 验证: 跨文档主张单边锚定(fc_a_001 判 C"两文档均AAA"为真只看了一篇) / 过选漏选。
- ✅ **架构简化（用户决定）**：砍掉确定性 merge + 降级旧管线，改 `solver._synthesize`(原题+题型+子问题结论/证据 → 1 次 LLM 给答案)；子问题无证据则标注并强行让 LLM 续答。
  分层 20 题: token **227k→158k(-30%, 双付费消除)**; 准确率 **88.2%→76.5%**。
  **诊断**: 掉分 100% 是【子问题检索失败(无证据)】非 synthesize——loop 找到证据时 synthesize 判得对(res_a_001 A/C/D 全对); 错的全是 verify 返回"无证据"的选项(reg_a_004 四选项全无证据/res_a_002 BCD 无证据)。
  旧 88.2% 是【降级用暴力旧 reasoner 掩盖了弱检索】; 去掉掩盖后 76.5% 是真实检索能力。
- ✅ **优化子问题检索**：用 query 缓存让检索确定 → 按【确定性召回】诊断(脱离 judge 噪声), 定位真因并修:
  ① genQuery 抽词 bug(原 prompt 说"不要写数字", 但 8894.3/7日 正是命中点; 且被"选项/是否成立"套话带偏) → 改成抽实体/数字、忽略设问套话
  ② option_verdict 用【选项原文】当主张(decomposer 有时只写"C选项是否成立"丢了内容)
  ③ 以 BM25 命中为中心开窗(`loop_window_chars:500`)：块可大, 命中在顶/底都不漏, 比从头截 1000 省一半 token
  ④ 多文档轮询召回(治"搜两篇全召回一篇")
  ⑤ query 缓存(`index/query_cache.json`)+`llm.seed:42` → 检索跨运行可复现
  **结果: 子问题命中证据 84%→97%**(分层20题, 确定性验证 3 个硬 case 全召回)。
  **关于随机性(用户质疑)**: 查明非代码 bug——百炼 MoE 即便 temp0+seed 也 best-effort, 边界判定跨时间会翻; loop 多次 LLM 调用级联放大。query 缓存把检索这层固定了, 残留噪声在 judge。
- ✅ **原子事实重构（用户设计）**：分解时按【单篇文档+单个事实】拆，取值不判真；比较/排序/判真全留到最后 synthesize 整体推理。
  - decompose 输出 `facts:[{ask,doc_hint}]`，跨文档比较拆成每篇各一条（"第二份<第一份"→ text01发行额 + text02发行额）；问"值是什么"不问"是不是X"；拿不准 doc 填 null(全搜)。
  - loop judge 泛化为"取事实的值"(数值/名称/评级/日期/规则)。
  - solver: 跑所有 fact(全 compute) → synthesize 拿到所有事实**逐选项整体比对**出答案。
  - **效果(分层20题)**: **fc 33%→100%**(跨文档比较根治, fc_a_001 ABD 已确定性验证), 整体 **59%→76.5%**。
  - 残留错题集中在: research(无名 doc 绑定错/解析) + value_compare 解析缺口(doc1 智盈) + 财报跨年。
- ⬜ **下一步**: ① research 无名 doc 的 doc_hint 绑定(补 outline 主题词) ② decompose 鲁棒性(偶发 fallback/archetype 标错) ③ 上游解析(doc1)。

---

## (原规划) 动机与设计

> 动机：现 §4 是"一次性把 top-N 块塞进单次 reasoner 调用"的传统 RAG，两个痛点：
> ① **检索块对人极不友好**——块无定位信息，人工审核困难（ins_a_001 的 160% 比例表埋在长条款里，肉眼难定位）；
> ② **evidence.json 把 reasoning 截到 `[-800:]`**，关键推导丢失（144 万的算法被截掉，见复盘）。
> 目标：换成模拟人类的 **Agentic RAG**——拆子问题 → 每子问题独立的无状态检索 loop → 确定性合并。**更可审计、且预期更省 token**。

## v4.0 核心共识（已与用户对齐，逐条锁定）

1. **拆"自带答案的子问题"，不是拆"块相不相关"。**
   子问题要自包含、答案确定。ins_a_001 拆成 4 个：每个携带题干全部数字，块只需提供**规则/条款**，LLM 套规则直接吐值。
   - 反例（不要）："这块含不含增益宝身故金计算规则？"
   - 正例（要）："已交100万/现价80万, 增益宝(40岁)基本保额90万/账户85万, 身故金=?" → `144万`

2. **省 token 的杠杆是 `(调用次数)×P`，不是块大小。**
   `总输入 ≈ K×C + 调用次数×P`（K=看过的块数, C=块大小, P=每次重发的题干+指令）。
   - 无状态、不累积上下文（用户拍板）→ 避免 O(K²)。
   - 判块**小批量打包**（3-5 块/次）摊薄 P；判块 prompt 只发**分解后的子问题**，不发原题。

3. **判块和答子问题合一**：块有规则就吐值，没有就"否→下一块"。不要单独的"这块相关吗"调用。

4. **检索词在 loop 内生成（不在分解时一次性给）。**
   理由：换词/换索引应是 loop 内部动作，零退出成本。jieba 直切原句不行（题干数字是噪声），由 LLM 出检索词，只针对"规则"，低质则 loop 内重出词重搜（有上限）。

5. **合并是按题型写死的确定性代码，不是第二次 LLM 推理。**
   算术/排序永远留在合并层，绝不折进 LLM。

6. **任一子问题没找到规则块 → 降级回旧 reasoner**（缺证据不硬合并）。

## v4.1 题型 → 子问题形状 → 合并（最终定版）

| 题型 | 子问题形状 | 合并逻辑（确定性代码） |
|---|---|---|
| `value_compare`（含 "A的fact > B的fact"） | **开放算值**：每实体一个 "=?" | 排序 N 个值 → 逐选项比对其陈述的排序 → 命中字母 |
| `option_verdict`（"下列哪些正确"） | **判真**：每选项 "对吗?" → 是/否 | 收集判真选项 + **add-only 偏置**（榜上 06/07_addonly 最高） |
| `single_fact`（财报取数） | 开放算值，1 个 | 哪个选项 = 它 |
| `fallback`（不可拆） | —— | 交旧 `reasoner.answer` |

- **value_compare 一律开放算值**（4 子问题），不做"候选值去重验证"（那会变 ~7 子问题、更费 token，已砍）。
- "判真验证"形状只归 `option_verdict`（选项本就是要判真假的主张）。
- ⚠️ **关系/排序不折进验证**：选项 A "智盈(90)>增益宝(144)" 里每个值都对、错在排序；若整句验证会被对得上的数字带跑误判真。值的真伪归验证层，排序归合并层。

## v4.2 单题流程

```
分解(1次LLM,全局) → 认题型 + 列子问题
  ↓ 每子问题并行, 各自无状态 loop:
    [loop内] 生成BM25检索词(LLM,只输出几个词)
      → BM25Index.search(doc_ids=本子问题文档)   # 复用现成子集检索
      → 读 top-k(小批量3-5块/次) → LLM: 含规则吐值 / 否
      → 否: 换下一批; 连续低质: loop内重出检索词重搜
      → 命中/触顶退出, 返回 {value 或 verdict, 依据chunk_id, found}
  ↓
确定性合并(按题型写死) → 字母
  ↓ 任一子问题 found=false → 降级旧 reasoner
```

## v4.3 分阶段实现（每阶段独立可测、对拍 89-对基线）

总原则：新增 `agent/agentic/` 子包，**旧管线（`pipeline.py`→`reasoner.py`）原样保留**，config 加 `mode: legacy|agentic` 开关。复用 `BM25Index.search(doc_ids=)`、`processed/*/*.json`(已有 `section_path/article_no/page/table.caption`)、`QwenClient.complete`、`USAGE`、`eval/accuracy.py`。**每阶段写完整结构化 trace（不再截断）**。

**阶段 1 · 面包屑 + per-doc outline**（低风险，先让块可读）
- `chunker/chunker.py`：块文本前置 `[{title} › {section_path} › {article_no}标题]`。
- 新增 `agentic/outline.py`：扫 processed → 每篇 `{doc_id,title,toc,headings}` → `index/outlines.json`；表格条目用 `table.caption`。
- 重建 BM25 → 跑旧管线**对拍基线，准确率不得低于基线**（只加信息）。
- 测：抽 5 篇人工看面包屑；准确率对拍。

**阶段 2 · 分解器**（1 次 LLM，先离线验证，不接管线）
- 新增 `agentic/decompose.py`：题 → JSON `[{sq, type, shape, entity/option, doc_hint}]` + 题型，强制校验。
- 测（**纯离线 0 风险**）：拿 `relabel_opus/*.md` 当 gold（Opus 已标"答案在 doc2 第五条"）。评 ①题型认对率 ②子问题覆盖所有实体/选项 ③doc_hint 命中正确文档。认题型错就回炉。

**阶段 3 · 单子问题 agentic loop**（无状态，先只跑 value_compare）
- 新增 `agentic/loop.py`：检索词生成在 loop 内 / 小批量判块 / 换词不退出 / 硬上限 / 每步 trace。
- 测：仅 insurance·value_compare 子集。三件套一起报：①子答案正确率 ②**检索召回**(依据块是否真含规则, 对 relabel gold) ③**每题 token**。对拍旧管线同子集。

**阶段 4 · 确定性合并 + 降级兜底**
- 新增 `agentic/merge.py`：按 §v4.1 写死；任一 `found=false` → 降级 `reasoner.answer`。
- 测：value_compare 端到端 vs 基线；看降级触发率（过高=分解/检索没做好）。

**阶段 5 · 全题型 + 全量对拍**
- option_verdict / single_fact 接入，全 100 题。
- `eval/accuracy.py` 加 token 列；画 **准确率 × token 帕累托** vs `16_best89`。
- 替换条件：准确率≥基线 且 token≤基线，或明确帕累托占优。否则保留旧管线。

## v4.4 先定的数（阶段 1 前确认）

| 项 | 值 | 理由 |
|---|---|---|
| 每子问题 token 预算 | 6k，超即降级 | 防 loop 跑飞 |
| loop 最大迭代 | 3 | 上限 |
| 判块批量 | 3-5 块/次 | 摊薄固定开销 P |
| 旧管线 | 全程保留 | 永远能对拍 + 降级 |

## v4.5 测试资产（已就位，无需新标）

- **检索召回 gold**：`relabel_opus/*.md` 内 Opus 已标证据定位（"doc2 第五条"），直接转 `问题→答案所在块` gold 集，脱离 LLM 单独测 recall@k。
- **端到端**：`dev_labels.json`（silver 有偏，勿单看；务必带 token 一起报）。
- **审计产物**：新 evidence trace = 子问题 / 发出的检索词 / 读过哪些块 / 每步判定 / 依据 chunk_id，可逐步回放——即本次重构的初心交付物。
