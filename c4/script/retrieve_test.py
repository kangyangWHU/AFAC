"""检索冒烟测试：对若干 A 榜题做逐选项 BM25 检索，看是否命中正确文档。
用法：python -m script.retrieve_test [n]
"""
from __future__ import annotations
import os
import sys
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config                  # noqa: E402
from agent.index.bm25 import BM25Index     # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    idx = BM25Index.from_file(os.path.join(config.path("index_dir"), "bm25.pkl"))
    qfiles = glob.glob(os.path.join(config.path("questions_dir"), "*.json"))
    per_opt = config.get("retrieval.per_option_query", True)

    questions = []
    for f in qfiles:
        questions += json.load(open(f, encoding="utf-8"))

    for q in questions[:n]:
        gold = set(q.get("doc_ids", []))
        # query: 题干 + 每个选项分别检索，合并命中文档
        queries = [q["question"]]
        if per_opt:
            queries += list(q["options"].values())
        hit_docs, top_chunks = [], []
        for qq in queries:
            hits = idx.search(qq, k=3, doc_ids=list(gold) or None)
            for h in hits:
                hit_docs.append(h.doc_id)
            if hits:
                top_chunks.append(hits[0])
        recovered = set(hit_docs)
        ok = "✓" if gold & recovered else "✗"
        print(f"\n[{q['qid']}] {q['domain']} gold={gold}")
        print(f"  逐选项检索覆盖文档: {recovered} 命中gold:{ok}")
        if top_chunks:
            best = max(top_chunks, key=lambda h: h.score)
            print(f"  最强证据[{best.meta.get('article_no') or best.meta.get('type')}] "
                  f"score={best.score:.1f}: {best.text[:90]}")


if __name__ == "__main__":
    main()
