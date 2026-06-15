"""检索召回评测：A 榜内（doc_ids 已知）逐题跑 EvidenceRetriever，
统计每域"全部 gold 文档被覆盖"的比例 + 平均证据块数/字符（token 代理）。
用法：python -m eval.retrieval_recall
"""
from __future__ import annotations
import os
import sys
import json
import glob
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config                              # noqa: E402
from agent.index.bm25 import BM25Index                # noqa: E402
from agent.retriever.retriever import EvidenceRetriever  # noqa: E402


def main():
    idx = BM25Index.from_file(os.path.join(config.path("index_dir"), "bm25.pkl"))
    rt = EvidenceRetriever(idx)
    qs = []
    for f in glob.glob(os.path.join(config.path("questions_dir"), "*.json")):
        qs += json.load(open(f, encoding="utf-8"))

    by = collections.defaultdict(lambda: [0, 0, 0, 0])
    for q in qs:
        gold = set(q["doc_ids"])
        ev = rt.retrieve(q["question"], q["options"], doc_ids=list(gold))
        covered = set(e.doc_id for e in ev)
        full = gold <= covered
        v = by[q["domain"]]
        v[0] += 1
        v[1] += int(full)
        v[2] += sum(len(e.text) for e in ev)
        v[3] += len(ev)

    hdr = f"{'domain':22}{'Qs':>4}{'全docs覆盖':>11}{'avg块':>7}{'avg证据字符':>12}"
    print(hdr)
    tot = [0, 0, 0, 0]
    for d, v in sorted(by.items()):
        print(f"{d:22}{v[0]:>4}{v[1]:>8}/{v[0]:<3}{v[3]//v[0]:>6}{v[2]//v[0]:>12}")
        for i in range(4):
            tot[i] += v[i]
    print(f"{'TOTAL':22}{tot[0]:>4}{tot[1]:>8}/{tot[0]:<3}{tot[3]//tot[0]:>6}{tot[2]//tot[0]:>12}")


if __name__ == "__main__":
    main()
