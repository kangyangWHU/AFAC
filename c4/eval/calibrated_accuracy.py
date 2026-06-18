"""Official-score calibrated evaluator for answer.csv.

Calibration source:
  - out/first_answer.csv official score: 70.9522 → true correct = 74
  - out/qwen3_235b_a22b_full_stream/answer.csv official score: 73.4616 → true correct = 77

After fixing 2 confirmed false-positive labels (fc_a_004 A→AC, ins_a_006 C→A):
  qwen3_235b:  raw silver 78/100, true=77  (over_credit=1, 1 agree_correct FP unidentified)
  first_answer: raw silver 76/100, true=74  (over_credit=2, 2 FPs unidentified)

Per-model CALIBRATED_OVER_CREDIT to achieve accurate score estimates:
  qwen3_235b:  set to 1
  first_answer: set to 2

CALIBRATED_OVER_CREDIT below defaults to 1 (correct for 235b submissions).
Change to 2 when evaluating first_answer.csv.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Per-file calibrated over-credit based on systematic label errors (2 fixed: fc_a_004, ins_a_006)
_OVER_CREDIT_MAP = {
    "qwen3_235b_a22b_full_stream": 1,   # raw=78, true=77, 1 unidentified agree_correct FP
    "first_answer": 2,                   # raw=76, true=74, 2 unidentified FPs
}
CALIBRATED_OVER_CREDIT = 0  # fallback for new submissions (assume labels are calibrated)
TOTAL_QUESTIONS = 100


def norm(a: str) -> str:
    return "".join(sorted(set(c for c in (a or "").upper() if c in "ABCD")))


def load_answers(path: str) -> tuple[dict[str, str], int]:
    answers: dict[str, str] = {}
    total_tokens = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["qid"] == "summary":
                total_tokens = int(row.get("total_tokens") or 0)
            else:
                answers[row["qid"]] = row["answer"]
    return answers, total_tokens


def token_factor(total_tokens: int) -> float:
    return 0.7 + 0.3 * ((5_000_000 - total_tokens) / 5_000_000)


def main() -> None:
    apath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out", "answer.csv")
    # Auto-select over-credit constant from known answer file patterns
    over_credit = CALIBRATED_OVER_CREDIT
    for key, val in _OVER_CREDIT_MAP.items():
        if key in apath:
            over_credit = val
            break
    pred, total_tokens = load_answers(apath)
    labels = json.load(open(os.path.join(ROOT, "dev_labels.json"), encoding="utf-8"))

    dom = defaultdict(lambda: [0, 0])
    wrong = []
    for qid, lab in labels.items():
        if qid not in pred:
            continue
        ok = norm(pred[qid]) == norm(lab["answer"])
        d = qid.split("_")[0]
        dom[d][0] += ok
        dom[d][1] += 1
        if not ok:
            wrong.append((qid, pred[qid], lab["answer"], lab.get("confidence", "")))

    raw_correct = sum(c for c, _ in dom.values())
    raw_seen = sum(n for _, n in dom.values())
    calibrated_correct = max(0, raw_correct - over_credit)
    factor = token_factor(total_tokens)
    calibrated_score = 100 * (calibrated_correct / TOTAL_QUESTIONS) * factor

    print("=== raw silver ===")
    for d in sorted(dom):
        c, n = dom[d]
        print(f"  {d:6} {c:>3}/{n:<3} = {100*c/max(n,1):5.1f}%")
    print(f"  {'TOTAL':6} {raw_correct:>3}/{raw_seen:<3} = {100*raw_correct/max(raw_seen,1):5.1f}%")

    print("\n=== official-calibrated estimate ===")
    print(f"  over_credit_correction: -{over_credit} questions")
    print(f"  calibrated_correct:    {calibrated_correct}/{TOTAL_QUESTIONS}")
    print(f"  total_tokens:          {total_tokens}")
    print(f"  token_factor:          {factor:.7f}")
    print(f"  estimated_score:       {calibrated_score:.4f}")

    print(f"\nwrong_vs_silver({len(wrong)}): " +
          ", ".join(f"{q} {p}!={g}[{conf}]" for q, p, g, conf in wrong))


if __name__ == "__main__":
    main()
