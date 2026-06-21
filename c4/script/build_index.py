"""分块 + 建 BM25 索引。
用法：python -m script.build_index            # 处理 processed/ 下所有文档
"""
from __future__ import annotations
import os
import sys
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config                       # noqa: E402
from agent.chunker.chunker import chunk_doc     # noqa: E402
from agent.index.bm25 import BM25Index          # noqa: E402
from agent.router.signatures import build_signatures, SignatureIndex  # noqa: E402


def main():
    proc = config.path("processed_dir")
    chunks_dir = config.path("chunks_dir")
    index_dir = config.path("index_dir")
    os.makedirs(chunks_dir, exist_ok=True)

    all_chunks: list[dict] = []
    files = glob.glob(os.path.join(proc, "*", "*.json"))
    for f in files:
        doc = json.load(open(f, encoding="utf-8"))
        for ch in chunk_doc(doc):
            all_chunks.append(ch.to_dict())

    # 落 chunks（jsonl，便于检查）
    cpath = os.path.join(chunks_dir, "chunks.jsonl")
    with open(cpath, "w", encoding="utf-8") as w:
        for c in all_chunks:
            w.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 建索引
    idx = BM25Index().build(all_chunks)
    ipath = os.path.join(index_dir, "bm25.pkl")
    idx.save(ipath)

    # 文档签名（标题+目录）索引，供 B 榜路由
    sigs = build_signatures(proc)
    spath = os.path.join(index_dir, "doc_signatures.json")
    SignatureIndex(sigs).save(spath)
    print(f"signatures -> {spath} ({len(sigs)} docs)")

    # 财报"主要会计数据"结构化摘要（优先 MinerU 干净表），供检索保底注入
    from agent.parser.fin_summary import extract_summary  # noqa: E402
    fin_sum = {}
    for f in glob.glob(os.path.join(proc, "financial_reports", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        s = extract_summary(d, root=ROOT)
        if s:
            fin_sum[d["doc_id"]] = s
    fpath = os.path.join(index_dir, "fin_summaries.json")
    json.dump(fin_sum, open(fpath, "w"), ensure_ascii=False, indent=1)
    print(f"fin_summaries -> {fpath} ({len(fin_sum)}/10 财报)")

    # per-doc 大纲（文档名+条文目录），供 v4 分解器；纯规则零模型
    from agent.agentic.outline import build_all as build_outlines  # noqa: E402
    outlines = build_outlines(proc)
    opath = os.path.join(index_dir, "outlines.json")
    json.dump(outlines, open(opath, "w"), ensure_ascii=False, indent=1)
    print(f"outlines -> {opath} ({len(outlines)} docs)")

    # 统计
    import collections
    by_dom = collections.Counter(c["domain"] for c in all_chunks)
    by_typ = collections.Counter(c["type"] for c in all_chunks)
    print(f"docs={len(files)}  chunks={len(all_chunks)}")
    print(f"  by domain: {dict(by_dom)}")
    print(f"  by type:   {dict(by_typ)}")
    avg = sum(len(c['text']) for c in all_chunks) / max(len(all_chunks), 1)
    print(f"  avg chunk chars: {avg:.0f}")
    print(f"chunks -> {cpath}")
    print(f"index  -> {ipath}")


if __name__ == "__main__":
    main()
