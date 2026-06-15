"""跑问答出结果。
用法：
  python -m script.run --out out/ [--n 10] [--domain insurance] [--no-gold]
"""
from __future__ import annotations
import os
import sys
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config            # noqa: E402
from agent.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--n", type=int, default=None, help="只跑前 n 题")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--no-gold", action="store_true", help="不用 gold doc_ids（B 榜模式）")
    args = ap.parse_args()

    qs = []
    for f in sorted(glob.glob(os.path.join(config.path("questions_dir"), "*.json"))):
        qs += json.load(open(f, encoding="utf-8"))
    if args.domain:
        qs = [q for q in qs if q["domain"] == args.domain]
    if args.n:
        qs = qs[:args.n]

    out = os.path.join(ROOT, args.out)
    Pipeline().run(qs, out, use_gold_docs=not args.no_gold)


if __name__ == "__main__":
    main()
