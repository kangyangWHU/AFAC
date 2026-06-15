"""用独立 silver 标签(dev_labels.json)给 answer.csv 评准确率。
按域 + 按置信度子集分别统计（high-only 更可信）。
用法：python -m eval.accuracy out/answer.csv
"""
from __future__ import annotations
import os
import sys
import csv
import json
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_answers(path: str) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["qid"] == "summary":
                continue
            out[row["qid"]] = (row["answer"] or "").strip()
    return out


def norm(a: str) -> str:
    return "".join(sorted(set(c for c in a.upper() if c in "ABCD")))


def main():
    apath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "answer.csv")
    pred = load_answers(apath)
    labels = json.load(open(os.path.join(ROOT, "dev_labels.json"), encoding="utf-8"))

    dom = collections.defaultdict(lambda: [0, 0])       # all
    dom_hi = collections.defaultdict(lambda: [0, 0])    # high-conf only
    wrong = []
    for qid, lab in labels.items():
        if qid not in pred:
            continue
        if lab.get("confidence") == "defective":
            continue                                  # 题目本身有瑕疵，排除计分
        d = qid.split("_")[0]
        ok = norm(pred[qid]) == norm(lab["answer"])
        dom[d][0] += ok
        dom[d][1] += 1
        if lab.get("confidence") == "high":
            dom_hi[d][0] += ok
            dom_hi[d][1] += 1
        if not ok:
            wrong.append((qid, lab["confidence"], pred[qid], lab["answer"]))

    def report(title, dd):
        print(f"\n=== {title} ===")
        tc = tn = 0
        for d in sorted(dd):
            c, n = dd[d]
            tc += c
            tn += n
            print(f"  {d:6} {c:>3}/{n:<3} = {100*c/max(n,1):5.1f}%")
        print(f"  {'TOTAL':6} {tc:>3}/{tn:<3} = {100*tc/max(tn,1):5.1f}%")

    report("全部题(含 medium/low silver)", dom)
    report("仅 high-confidence silver 子集", dom_hi)
    print(f"\n错题({len(wrong)}): " +
          ", ".join(f"{q}[{c}] 预测{p}≠标{g}" for q, c, p, g in wrong))


if __name__ == "__main__":
    main()
