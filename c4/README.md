# 赛题四 · 金融长文本 Agent

架构、设计与最终结果见 [plan.md](./plan.md)。

**最终结果（A 榜 100 题，silver 标签离线评测）**：A 模式(给 doc_ids) **81%**、B 模式(路由找文档) **82%**（high-conf 子集 84–86%）；Token ~730K/100题。官方 baseline 15%。

## 环境（conda env: `AFAC4`）

```bash
conda activate AFAC4          # python 3.11
# 核心依赖
pip install -r requirements.txt
# MinerU（扫描件/复杂表格 fallback，较重；模型走 modelscope）
pip install "mineru[core]"
# ⚠️ GPU 修正：mineru[core] 默认拉 torch 2.12+cu130(需 CUDA13 驱动)，
#    本机驱动 575.57(CUDA 12.9) 不支持 → cuda 不可用。改装 cu128 版：
pip uninstall -y torch torchvision
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
export MINERU_MODEL_SOURCE=modelscope
mineru-models-download -s modelscope -m pipeline   # 一次性下载 ~2.4G pipeline 模型
```

GPU 自检：
```bash
~/anaconda3/envs/AFAC4/bin/python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 4090
# 跑 mineru 时加 -d cuda；实测扫描件 5 页 GPU 40s（含模型加载），日志显示 GPU Memory 23GB
```

> ⚠️ MinerU 会把 numpy 升到 2.x，**不要装进 base 环境**（会破坏 base 的 scipy/sklearn）。
> 一律用 `AFAC4`。模型缓存在 `~/.cache/modelscope`，各 env 共享。

## 复现流程

```bash
PY=~/anaconda3/envs/AFAC4/bin/python
# 1) 解析 + 索引
$PY agent/doc_index.py                 # doc_id->文件 全库索引(573篇)
$PY -m script.preprocess --all         # 解析全库 573 篇(B榜路由需要全库)
$PY -m script.build_index              # 分块 + BM25 索引(19303 chunks)
# 2) 评测检索/路由(用 A 榜 gold 当真值)
$PY -m eval.parse_check                # 解析质量：条款连续性 + 回指校验
$PY -m eval.doc_recall                 # B榜文档路由召回 R@k / 全@k / MRR
# 3) 端到端出答案 + 评分
$PY -m script.run --out out/A_given_docids            # A榜：给 doc_ids
$PY -m script.run --out out/B_routed --no-gold        # B榜：路由找文档
$PY -m eval.accuracy out/B_routed/answer.csv          # 对 dev_labels.json 评分
```

辅助评测脚本：`eval/table_check`(财报科目可取性)、`eval/fallback_scan`(MinerU 触发)、`eval/retrieval_recall`(证据覆盖)。
`script/dump_evidence` 导出每题证据供人工/独立标注（dev_labels.json 由此而来）。

## LLM（本地 Qwen，合规）
- 本地 vLLM OpenAI 兼容端点：`http://localhost:12345/v1`，模型 `Qwen/Qwen3.6-35B-A3B`。
- `config.yaml::llm` 可调 base_url/model/temperature/max_tokens/enable_thinking。
- **thinking 模式**：Qwen3.6 默认开思考，一个字母答案要烧 279 token；`enable_thinking:false` 降到 2，靠 prompt 里显式逐项 CoT 保推理质量。实测 ~5.8K token/题。

## 已验证的关键结论

- **解析**：68 篇 A 榜文档 0 失败；条款连续性 16/25 通过，异常已分类（多段编号/交叉引用/非条文）。
- **质量门控**：全库 190 PDF 仅 **3 篇**触发 MinerU fallback（`csrc_0038_att2` 纯扫描件 + `csrc_0036_att1/att3` 单页近空），财报/合同 0 误伤。
- **MinerU OCR 增益（实测）**：扫描件 `csrc_0038_att2` —— PyMuPDF **0 字符** → MinerU **2709 字符**，标题/正文/章节结构全部正确识别。证明 fallback 链对 B 榜未知扫描件有效。
