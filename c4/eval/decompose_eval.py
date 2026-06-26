"""离线评测分解器（plan.md §v4 阶段2）：跑全部 A 题，统计客观指标 + 落人审 dump。
不碰旧管线、不跑检索/推理。指标：
- archetype 分布（按域）
- coverage：option_verdict 是否每选项一个子问题(#sq==#opts)
- binding：子问题里 doc_hint 非空占比（名字覆盖差则低，None 安全=全搜）
- fallback / 解析失败计数
用法：python -m eval.decompose_eval [--n N] [--domain D]
"""
from __future__ import annotations
import os
import sys
import json
import glob
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config                       # noqa: E402
from agent.llm.qwen import QwenClient          # noqa: E402
from agent.llm.base import USAGE               # noqa: E402
from agent.agentic.decompose import Decomposer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    qs = []
    for f in sorted(glob.glob(os.path.join(config.path("questions_dir"), "*.json"))):
        qs += json.load(open(f, encoding="utf-8"))
    if args.domain:
        qs = [q for q in qs if q["domain"] == args.domain]
    if args.n:
        qs = qs[:args.n]

    outlines = json.load(open(os.path.join(config.path("index_dir"), "outlines_vl.json"),
                               encoding="utf-8"))
    dec = Decomposer(QwenClient(), outlines)

    results: dict[str, dict] = {}

    def work(q):
        return q["qid"], q, dec.decompose(q, q["doc_ids"])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, q) for q in qs]
        for i, fut in enumerate(as_completed(futs), 1):
            qid, q, r = fut.result()
            results[qid] = {"q": q, "r": r}
            print(f"[{i}/{len(qs)}] {qid} -> {r['archetype']} (#sq={len(r['sub_questions'])})")

    # 统计
    by_dom = collections.defaultdict(lambda: collections.Counter())
    cover_ok = cover_tot = 0
    bind_hit = bind_tot = 0
    fallback = empty = 0
    for qid, d in results.items():
        q, r = d["q"], d["r"]
        arch = r["archetype"]
        by_dom[q["domain"]][arch] += 1
        sqs = r["sub_questions"]
        if arch == "fallback":
            fallback += 1
        if not sqs:
            empty += 1
        if arch == "option_verdict" and q.get("answer_format") != "tf":
            # tf 题选项是"正确/错误", 只需 1 个子问题验证那条主张; 仅多选要求每选项一个
            cover_tot += 1
            cover_ok += (len(sqs) == len(q["options"]))
        for s in sqs:
            bind_tot += 1
            bind_hit += (s.get("doc_hint") is not None)

    print("\n=== archetype 分布（按域）===")
    for dom in sorted(by_dom):
        print(f"  {dom}: {dict(by_dom[dom])}")
    print("\n=== 客观指标 ===")
    print(f"  题数: {len(results)}  fallback: {fallback}  空子问题: {empty}")
    if cover_tot:
        print(f"  option_verdict 覆盖(#sq==#opts): {cover_ok}/{cover_tot} = {cover_ok/cover_tot:.0%}")
    if bind_tot:
        print(f"  doc_hint 绑定率(非空): {bind_hit}/{bind_tot} = {bind_hit/bind_tot:.0%}  (None=安全全搜)")
    print(f"  token: {USAGE.as_dict()}")

    # 人审 dump
    out_dir = os.path.join(ROOT, "out", "decompose")
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for qid in sorted(results):
        q, r = results[qid]["q"], results[qid]["r"]
        lines.append(f"## {qid} [{q.get('answer_format')}] {r['archetype']}  gold_docs={q['doc_ids']}")
        for s in r["sub_questions"]:
            lines.append(f"- [{s.get('entity')}|{s.get('shape')}|doc={s.get('doc_hint')}] {s['sq']}")
        lines.append("")
    dump = os.path.join(out_dir, "decompose.md")
    open(dump, "w", encoding="utf-8").write("\n".join(lines))
    json.dump({k: v["r"] for k, v in results.items()},
              open(os.path.join(out_dir, "decompose.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n人审 dump -> {dump}")


if __name__ == "__main__":
    main()
