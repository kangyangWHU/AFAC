"""B 榜文档路由评测：用 A 榜 gold doc_ids 当真值。
屏蔽 doc_ids，让 DocRouter 在域内找文档，算 Recall@k / MRR。
用法：python -m eval.doc_recall
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
from agent.router.router import DocRouter  # noqa: E402

KS = [1, 3, 5, 10]


def main():
    idx = BM25Index.from_file(os.path.join(config.path("index_dir"), "bm25.pkl"))
    router = DocRouter(idx)
    qs = []
    for f in glob.glob(os.path.join(config.path("questions_dir"), "*.json")):
        qs += json.load(open(f, encoding="utf-8"))

    # 域内文档规模（路由难度）
    dom_size = {d: len(s) for d, s in router.domain_docs.items()}

    by = collections.defaultdict(lambda: collections.defaultdict(float))
    by_n = collections.Counter()
    for q in qs:
        gold = set(q["doc_ids"])
        ranked = router.rank(q["question"], q["options"], q["domain"])
        pos = {d: ranked.index(d) for d in gold if d in ranked}
        dom = q["domain"]
        by_n[dom] += 1
        # Recall@k：gold 中有多少落在前 k（按比例）；full@k：是否全部 gold 都在前 k
        for k in KS:
            topk = set(ranked[:k])
            by[dom][f"recall@{k}"] += len(gold & topk) / len(gold)
            by[dom][f"full@{k}"] += float(gold <= topk)
        # MRR：第一个命中 gold 的名次
        if pos:
            by[dom]["mrr"] += 1.0 / (min(pos.values()) + 1)

    print(f"{'domain':20}{'docs':>5}{'Qs':>4}  " +
          "  ".join(f"R@{k}" for k in KS) + "   " +
          "  ".join(f"全@{k}" for k in KS) + "   MRR")
    agg = collections.defaultdict(float)
    tot = 0
    for dom in sorted(by):
        n = by_n[dom]
        tot += n
        row = by[dom]
        cells = []
        for k in KS:
            cells.append(f"{100*row[f'recall@{k}']/n:4.0f}")
        for k in KS:
            cells.append(f"{100*row[f'full@{k}']/n:4.0f}")
        mrr = row["mrr"] / n
        print(f"{dom:20}{dom_size.get(dom,0):>5}{n:>4}  " + " ".join(cells) + f"  {mrr:.2f}")
        for kk, vv in row.items():
            agg[kk] += vv
    cells = [f"{100*agg[f'recall@{k}']/tot:4.0f}" for k in KS]
    cells += [f"{100*agg[f'full@{k}']/tot:4.0f}" for k in KS]
    print(f"{'TOTAL':20}{'':>5}{tot:>4}  " + " ".join(cells) + f"  {agg['mrr']/tot:.2f}")
    print("\nR@k=gold命中比例均值; 全@k=该题所有gold都进前k的比例(决定能否答对); MRR=首个gold命中倒数名次")


if __name__ == "__main__":
    main()
