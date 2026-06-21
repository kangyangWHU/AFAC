"""v4 Agentic 端到端入口（plan.md §v4）。镜像 script/run.py，但走 AgenticSolver。
输出 answer.csv（含 token summary）+ agentic_audit.json（子问题/值/判真/来源块, 可逐步回放）。
用法：python -m script.run_agentic --out out/agentic [--n N] [--domain D]
"""
from __future__ import annotations
import os
import sys
import csv
import json
import glob
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config                       # noqa: E402
from agent.llm.base import USAGE               # noqa: E402
from agent.agentic.solver import AgenticSolver  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/agentic")
    ap.add_argument("--n", type=int, default=None, help="每域只跑前 n 题（分层抽样）")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    by_dom_q: dict[str, list] = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(config.path("questions_dir"), "*.json"))):
        for q in json.load(open(f, encoding="utf-8")):
            by_dom_q[q["domain"]].append(q)
    qs = []
    for dom, lst in by_dom_q.items():
        if args.domain and dom != args.domain:
            continue
        qs += lst[:args.n] if args.n else lst

    solver = AgenticSolver()
    n = len(qs)
    rows: list = [None] * n
    audit: list = [None] * n

    def work(i_q):
        i, q = i_q
        a = solver.answer(q, q["doc_ids"])
        return i, q, a

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, (i, q)) for i, q in enumerate(qs)]
        for fut in as_completed(futs):
            i, q, a = fut.result()
            rows[i] = (q["qid"], a.answer)
            audit[i] = {"qid": q["qid"], "answer": a.answer, "path": a.path,
                        "archetype": a.archetype, "sub": a.sub}
            done += 1
            print(f"[{done}/{n}] {q['qid']} -> {a.answer}  ({a.path})  tok={USAGE.total_tokens}")

    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "answer.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", USAGE.prompt_tokens, USAGE.completion_tokens, USAGE.total_tokens])
        for qid, ans in rows:
            w.writerow([qid, ans, "", "", ""])
    json.dump(audit, open(os.path.join(out, "agentic_audit.json"), "w"),
              ensure_ascii=False, indent=1)

    arch = collections.Counter(a["archetype"] for a in audit)
    nsub = sum(len(a["sub"]) for a in audit)
    nfound = sum(1 for a in audit for s in a["sub"] if s.get("found"))
    print(f"\narchetype 分布: {dict(arch)}")
    if nsub:
        print(f"子问题命中证据: {nfound}/{nsub} = {nfound/nsub:.0%}")
    print(f"token: {USAGE.as_dict()}\nanswers -> {out}/answer.csv")


if __name__ == "__main__":
    main()
