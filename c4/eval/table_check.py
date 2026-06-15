"""财报/合同表格可取性验证：关键科目能否取到且带年度数字。
命中率低于阈值的文档 → 标记为 MinerU 升级候选。
用法：python -m eval.table_check
"""
from __future__ import annotations
import os
import sys
import json
import glob
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NUM = re.compile(r"\d[\d,]{5,}")
REPORT_METRICS = ["营业收入", "营业总收入", "归属于上市公司股东的净利润",
                  "经营活动产生的现金流量净额", "研发投入", "现金分红"]
THRESHOLD = 4  # 命中 < 4/6 -> MinerU 候选


def check_dir(domain: str, metrics: list[str]):
    print(f"=== {domain} 关键科目可取性 (阈值 {THRESHOLD}/{len(metrics)}) ===")
    candidates = []
    for f in sorted(glob.glob(os.path.join(ROOT, "processed", domain, "*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        tab = "\n".join(b["text"] for b in d["blocks"] if b["type"] == "table")
        hits = [m for m in metrics
                if any(m in ln and NUM.search(ln) for ln in tab.splitlines())]
        flag = "" if len(hits) >= THRESHOLD else "  <-- MinerU候选"
        print(f"  {os.path.basename(f)[:30]:32} {len(hits)}/{len(metrics)}{flag}")
        if len(hits) < THRESHOLD:
            candidates.append(d["doc_id"])
    if candidates:
        print(f"  MinerU 升级候选: {candidates}")
    return candidates


if __name__ == "__main__":
    check_dir("financial_reports", REPORT_METRICS)
