# -*- coding: utf-8 -*-
"""拼 B 榜提交：新跑的 long 半边 + 沿用上一版的 table 半边。

只重跑 long 时用这个,保证 table 半边逐字不变 —— 一次提交只动一个变量,
线上分数的变化才能归因。旧版归档到 out/old/。

用法(在 src 目录)：
  python -m tools.build_submission_b --long ../out/_long_B_v5_raw.csv \
      --base ../out/submission_long_B_scorefix_v4.csv --out ../out/submission_B_v5.csv
"""
import os
import csv
import sys
import shutil
import argparse

from common.config import OUT_DIR, B_LONG_DIR, B_TABLE_DIR

csv.field_size_limit(sys.maxsize)          # 单元格可达数十万字符


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return {r["file_name"]: r["ground_truth"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, help="新跑的 long 结果 csv")
    ap.add_argument("--base", required=True, help="上一版完整提交(取其 table 半边)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    longs, base = _read(a.long), _read(a.base)
    lnames = sorted(os.listdir(os.path.join(B_LONG_DIR, "images")))
    tnames = sorted(os.listdir(os.path.join(B_TABLE_DIR, "images")))

    rows, empty = [], []
    for n in lnames + tnames:
        md = longs.get(n) if n in set(lnames) else base.get(n)
        if md is None:
            raise SystemExit(f"缺少 {n} 的结果")
        if not md.strip():
            empty.append(n)
        rows.append([n, md])

    if empty:
        print(f"[warn] {len(empty)} 份结果为空 —— 提交前必须查:")
        for n in empty:
            print("   ", n)

    out = os.path.abspath(a.out)
    if os.path.exists(out):                      # 同名旧版先归档,不覆盖
        old = os.path.join(OUT_DIR, "old")
        os.makedirs(old, exist_ok=True)
        shutil.move(out, os.path.join(old, os.path.basename(out)))
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["file_name", "ground_truth"])
        w.writerows(rows)
    print(f"{len(rows)} 行 (long {len(lnames)} + table {len(tnames)}) → {out}")


if __name__ == "__main__":
    main()
