"""v4 Agentic 端到端入口（plan.md §v4）。镜像 script/run.py，但走 AgenticSolver。
输出 answer.csv（含 token summary）+ agentic_audit.json（子问题/值/判真/来源块, 可逐步回放）。
崩溃/限流安全：答案/审计/token 三者均【逐题落盘】，--resume 合并续跑（含重试 error 题），
单题异常被兜底捕获、不拖垮全量；agentic_audit.json 从 audit_live.jsonl 全量重建，crash/续跑都不丢明细。
用法：python -m script.run_agentic --out out/agentic [--n N] [--domain D] [--resume]
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
from agent import config                                       # noqa: E402
from agent.llm.base import USAGE                               # noqa: E402
from agent.agentic.solver import AgenticSolver, AgenticAnswer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/agentic")
    ap.add_argument("--n", type=int, default=None, help="每域只跑前 n 题（分层抽样）")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="跳过 answer_live.csv 已【成功】的 qid, 续跑剩余(含重试 error 题)")
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

    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    live_path = os.path.join(out, "answer_live.csv")     # 答案逐题落盘
    audit_path = os.path.join(out, "audit_live.jsonl")   # 审计明细逐题落盘(crash-safe + resume 合并)
    tok_path = os.path.join(out, "tokens.json")          # token 累计落盘(resume 续算 → summary 正确)
    resuming = args.resume and os.path.exists(live_path)
    if resuming:                                         # 断点续跑: 跳过已【成功】的 qid(error 题重试), append 续写
        done_qids = {r[0] for r in csv.reader(open(live_path))
                     if r and r[0] not in ("qid", "summary")
                     and not (len(r) > 2 and r[2].startswith("error"))}
        qs = [q for q in qs if q["qid"] not in done_qids]
        live = open(live_path, "a", newline="", encoding="utf-8"); lw = csv.writer(live)
        audit_live = open(audit_path, "a", encoding="utf-8")
        if os.path.exists(tok_path):                    # 续算之前批次 token, 让最终 summary 正确
            t = json.load(open(tok_path))
            USAGE.add(t.get("prompt_tokens", 0), t.get("completion_tokens", 0))
        print(f"[resume] 已成功 {len(done_qids)} 题, 续跑剩 {len(qs)} 题", flush=True)
    else:
        live = open(live_path, "w", newline="", encoding="utf-8"); lw = csv.writer(live)
        lw.writerow(["qid", "answer", "path"]); live.flush()
        audit_live = open(audit_path, "w", encoding="utf-8")

    solver = AgenticSolver()
    n = len(qs)

    def work(q):
        try:
            a = solver.answer(q, q["doc_ids"])
        except Exception as e:                          # 单题失败不拖垮全量: 兜底答案+error标记, 续跑可重试
            a = AgenticAnswer(q["qid"], "A", f"error:{type(e).__name__}", "error")
            print(f"[ERROR] {q['qid']}: {type(e).__name__}: {e}", flush=True)
        return q, a

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, q) for q in qs]
        for fut in as_completed(futs):
            q, a = fut.result()
            entry = {"qid": q["qid"], "answer": a.answer, "path": a.path,
                     "archetype": a.archetype, "sub": a.sub,
                     "verdicts": a.verdicts, "selection_rule": a.selection_rule}
            done += 1
            lw.writerow([q["qid"], a.answer, a.path]); live.flush()                       # 答案逐题落盘
            audit_live.write(json.dumps(entry, ensure_ascii=False) + "\n"); audit_live.flush()  # 审计逐题落盘
            json.dump({"prompt_tokens": USAGE.prompt_tokens,                              # token 累计落盘
                       "completion_tokens": USAGE.completion_tokens,
                       "total_tokens": USAGE.total_tokens}, open(tok_path, "w"))
            print(f"[{done}/{n}] {q['qid']} -> {a.answer}  ({a.path})  tok={USAGE.total_tokens}", flush=True)
    live.close(); audit_live.close()

    # answer.csv: 从 answer_live 全量读出(resume 含之前批次), 同 qid 取最后一次(重试覆盖 error)
    ans_by_qid: dict = {}
    for r in csv.reader(open(live_path)):
        if r and r[0] not in ("qid", "summary"):
            ans_by_qid[r[0]] = r[1]
    with open(os.path.join(out, "answer.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", USAGE.prompt_tokens, USAGE.completion_tokens, USAGE.total_tokens])
        for qid, ans in ans_by_qid.items():
            w.writerow([qid, ans, "", "", ""])
    # agentic_audit.json: 从 audit_live.jsonl 全量重建, 同 qid 取最后一次 → crash/续跑都不丢明细
    audit_by_qid: dict = {}
    for line in open(audit_path, encoding="utf-8"):
        line = line.strip()
        if line:
            e = json.loads(line); audit_by_qid[e["qid"]] = e
    audit = list(audit_by_qid.values())
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
