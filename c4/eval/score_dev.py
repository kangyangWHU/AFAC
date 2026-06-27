"""新简化管线(规则decompose→合并检索→领域prompt一次答题)全量跑 dev + 评分。
同本地 gpt-oss、现 PaddleOCR-VL 索引。增量存 out/score_dev.jsonl(可断点续)。
用法: python eval/score_dev.py
"""
import json, sys, os, glob, collections, threading
sys.path.insert(0, "/home/yk/code/AFAC/c4"); os.chdir("/home/yk/code/AFAC/c4")
from agent import config
config.load()["llm"].update(base_url="http://localhost:30000/v1", model="openai/gpt-oss-120b", api_key="EMPTY")
from agent.agentic.solver import AgenticSolver
from concurrent.futures import ThreadPoolExecutor, as_completed

labels = json.load(open("dev_labels.json", encoding="utf-8"))
Q = {}
for f in sorted(glob.glob(config.path("questions_dir") + "/*.json")):
    for q in json.load(open(f, encoding="utf-8")):
        Q[q["qid"]] = q
norm = lambda a: "".join(sorted(set(c for c in (a or "").upper() if c in "ABCD")))

OUT = "out/score_dev.jsonl"
done = set()
if os.path.exists(OUT):
    for ln in open(OUT, encoding="utf-8"):
        try:
            done.add(json.loads(ln)["qid"])
        except Exception:
            pass
S = AgenticSolver()
lock = threading.Lock(); fh = open(OUT, "a", encoding="utf-8")


def work(qid):
    q = Q[qid]
    a = S.answer(q, q["doc_ids"])
    lab = labels[qid]
    rec = {"qid": qid, "conf": lab.get("confidence"), "pred": norm(a.answer),
           "gold": norm(lab["answer"]), "ok": norm(a.answer) == norm(lab["answer"])}
    with lock:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    return rec


todo = [qid for qid in Q if qid in labels and qid not in done]
print("待跑 %d (已完成 %d)" % (len(todo), len(done)))
with ThreadPoolExecutor(max_workers=8) as ex:
    for fut in as_completed([ex.submit(work, qid) for qid in todo]):
        fut.result()
fh.close()

recs = [json.loads(ln) for ln in open(OUT, encoding="utf-8") if json.loads(ln)["conf"] != "defective"]


def rep(sub, name):
    dom = collections.defaultdict(lambda: [0, 0])
    for r in sub:
        d = dom[r["qid"][:3]]; d[0] += 1; d[1] += r["ok"]
    t = sum(d[0] for d in dom.values()); o = sum(d[1] for d in dom.values())
    print("\n=== %s (n=%d) ===" % (name, t))
    for k in sorted(dom):
        n, ok = dom[k]; print("  %-4s %2d/%2d = %3.0f%%" % (k, ok, n, 100*ok/n))
    print("  TOTAL %2d/%2d = %3.0f%%" % (o, t, 100*o/t))


rep(recs, "新管线 全部")
rep([r for r in recs if r["conf"] == "high"], "新管线 high-conf")
