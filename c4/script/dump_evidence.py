"""为人工/独立标注导出每题证据。
小文档(<阈值)整篇给出；大文档(财报/合同)用 BM25 逐选项取相关块。
输出 labeling/evidence_<domain>.jsonl
"""
from __future__ import annotations
import os
import sys
import json
import glob
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config                  # noqa: E402
from agent.index.bm25 import BM25Index     # noqa: E402

FULL_DOC_MAX_CHARS = 22000   # 文档索引块总字符 < 此 -> 整篇给标注者
BIG_DOC_TOPK = 30            # 大文档每题取的块数


def main():
    idx = BM25Index.from_file(os.path.join(config.path("index_dir"), "bm25.pkl"))
    # 文档 -> 块索引（按 seq 排序），及总字符
    doc_chars = {d: sum(len(idx.chunks[i]["text"]) for i in ii)
                 for d, ii in idx.by_doc.items()}

    out_dir = os.path.join(ROOT, "labeling")
    os.makedirs(out_dir, exist_ok=True)
    qfiles = glob.glob(os.path.join(config.path("questions_dir"), "*.json"))
    per_dom = collections.defaultdict(list)

    for f in qfiles:
        for q in json.load(open(f, encoding="utf-8")):
            gold = q["doc_ids"]
            evidence = []
            for d in gold:
                ii = idx.by_doc.get(d, [])
                if doc_chars.get(d, 0) <= FULL_DOC_MAX_CHARS:
                    # 整篇，按 seq
                    chunks = sorted((idx.chunks[i] for i in ii), key=lambda c: c.get("seq", 0))
                    for c in chunks:
                        evidence.append({"doc_id": d, "tag": c.get("article_no") or c.get("type"),
                                         "text": c["text"]})
                else:
                    # 大文档：逐选项 + 题干 BM25
                    seen = set()
                    for query in [q["question"]] + list(q["options"].values()):
                        for h in idx.search(query, k=8, doc_ids=[d]):
                            if h.chunk_id in seen:
                                continue
                            seen.add(h.chunk_id)
                            evidence.append({"doc_id": d, "tag": h.meta.get("article_no") or h.meta.get("type"),
                                             "text": h.text})
                            if len([e for e in evidence if e["doc_id"] == d]) >= BIG_DOC_TOPK:
                                break
            per_dom[q["domain"]].append({
                "qid": q["qid"], "domain": q["domain"], "question": q["question"],
                "options": q["options"], "answer_format": q["answer_format"],
                "doc_ids": gold, "evidence": evidence,
                "evidence_chars": sum(len(e["text"]) for e in evidence)})

    for dom, items in per_dom.items():
        p = os.path.join(out_dir, f"evidence_{dom}.jsonl")
        with open(p, "w", encoding="utf-8") as w:
            for it in items:
                w.write(json.dumps(it, ensure_ascii=False) + "\n")
        avg = sum(it["evidence_chars"] for it in items) // max(len(items), 1)
        print(f"{dom}: {len(items)} 题 -> {p}  avg证据字符={avg}")


if __name__ == "__main__":
    main()
