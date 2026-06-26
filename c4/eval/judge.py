"""judge 单批验证：每条 fact 取一批(loop_batch=8)块 + 一次 _judge（无 re-query、无 synth/verdict），
测 judge 从【首批块】抽到值的 found 率。真 loop 还会 re-query 兜底，故这是【下界】。
慢（全量数百次 LLM）——建议后台跑：run_in_background。增量存 out/judge.jsonl，可断点续跑（重跑跳过已完成）。
用法：python eval/judge.py   （默认本地 gpt-oss）
"""
import json, sys, os, re, collections, threading
sys.path.insert(0, "/home/yk/code/AFAC/c4"); os.chdir("/home/yk/code/AFAC/c4")
from agent import config
config.load()["llm"].update(base_url="http://localhost:30000/v1", model="openai/gpt-oss-120b", api_key="EMPTY")
from agent.index.bm25 import BM25Index
from agent.agentic.loop import SubQLoop
from agent.agentic.solver import _DOC_REF
from agent.llm.qwen import QwenClient
from concurrent.futures import ThreadPoolExecutor, as_completed

SCEN = re.compile(r"[王李张赵刘陈]某|某人")
OUT = "out/judge.jsonl"
idx = BM25Index.from_file("processed_vl/bm25_vl.pkl"); loop = SubQLoop(QwenClient(), idx)
A = json.load(open("out/decompose_route.json"))

done = set()
if os.path.exists(OUT):
    for ln in open(OUT, encoding="utf-8"):
        try:
            done.add(json.loads(ln)["key"])
        except Exception:
            pass

jobs = []
for e in A:
    for i, f in enumerate(e["facts"]):
        if SCEN.search(f["ask"]):
            continue
        key = "%s#%d" % (e["qid"], i)
        if key not in done:
            jobs.append((key, e["qid"], f))

lock = threading.Lock(); fh = open(OUT, "a", encoding="utf-8")


def work(key, qid, f):
    ask = _DOC_REF.sub("", f["ask"]).strip()
    hits = loop._retrieve(ask, f["route"].split("/"))[:loop.batch]
    v = loop._judge(ask, hits, ask)
    rec = {"key": key, "qid": qid, "ask": ask, "found": bool(v.get("found")), "value": str(v.get("value", ""))[:60]}
    with lock:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    return rec


print("待跑 %d (已完成 %d)" % (len(jobs), len(done)))
with ThreadPoolExecutor(max_workers=12) as ex:
    for fut in as_completed([ex.submit(work, *j) for j in jobs]):
        fut.result()
fh.close()

dom = collections.defaultdict(lambda: [0, 0])
for ln in open(OUT, encoding="utf-8"):
    r = json.loads(ln); d = dom[r["qid"][:3]]; d[0] += 1; d[1] += int(r["found"])
tot = sum(d[0] for d in dom.values()); fnd = sum(d[1] for d in dom.values())
print("=== judge 单批 found 率(累计 %d) ===" % tot)
for k in sorted(dom):
    nn, fo = dom[k]; print("  %-4s found %3d/%3d (%.0f%%)" % (k, fo, nn, 100 * fo / nn))
print("总 found %d/%d (%.0f%%)" % (fnd, tot, 100 * fnd / tot if tot else 0))
