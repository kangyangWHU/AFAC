"""量化 judge 真·miss 的因：对每条 miss 取【最佳路由篇】(关键词覆盖最高的那篇)的 top-30 块再 judge 一次。
  A 答案在篇·只是排在 9-30 → boilerplate(封面/释义/目录)把它挤出 top-8 → 过滤低值块可救
  B 篇内 30 块也没有 → 不在该文档(policy param / scope 错 / 蒸馏题) 或 judge 抽不出
增量存 out/miss_deep.jsonl，可断点续。用法：python eval/miss_deep.py
"""
import json, sys, os, collections, threading
sys.path.insert(0, "/home/yk/code/AFAC/c4"); os.chdir("/home/yk/code/AFAC/c4")
from agent import config
config.load()["llm"].update(base_url="http://localhost:30000/v1", model="openai/gpt-oss-120b", api_key="EMPTY")
from agent.index.bm25 import BM25Index, tokenize
from agent.agentic.loop import SubQLoop
from agent.agentic.solver import _DOC_REF
from agent.llm.qwen import QwenClient
from concurrent.futures import ThreadPoolExecutor, as_completed

idx = BM25Index.from_file("processed_vl/bm25_vl.pkl"); loop = SubQLoop(QwenClient(), idx)
A = json.load(open("out/decompose_route.json"))
GEN = set("是 多少 是否 的 在 年 是什么 预计 为 元 中 有 哪 项 该 了 文档 报告 占 与".split())
kt = lambda a: [t for t in tokenize(a, True, "jieba") if t not in GEN and len(t) > 1]

byask = collections.defaultdict(lambda: [0, 0])
for ln in open("out/judge.jsonl"):
    r = json.loads(ln); byask[r["ask"]][1] += 1; byask[r["ask"]][0] += int(r["found"])
miss = [a for a, (fo, _t) in byask.items() if fo == 0]
a2r = collections.defaultdict(set)
for e in A:
    for f in e["facts"]:
        a2r[_DOC_REF.sub("", f["ask"]).strip()].add(f["route"])

OUT = "out/miss_deep.jsonl"
done = set()
if os.path.exists(OUT):
    for ln in open(OUT):
        try:
            done.add(json.loads(ln)["ask"])
        except Exception:
            pass
lock = threading.Lock(); fh = open(OUT, "a")


def work(ask):
    docs = sorted({d for r in a2r.get(ask, []) for d in r.split("/")})
    toks = set(kt(ask)); best = None; bc = -1.0; bh = []
    for d in docs:                                   # 最佳篇 = top-30 关键词覆盖最高(=最可能的右篇)
        h = loop._retrieve(ask, [d])[:30]
        cov = sum(1 for t in toks if t in " ".join(x.text for x in h)) / max(len(toks), 1)
        if cov > bc:
            bc, best, bh = cov, d, h
    v = loop._judge(ask, bh, ask) if bh else {"found": False}
    rec = {"ask": ask, "best_doc": best, "found30": bool(v.get("found")), "value": str(v.get("value", ""))[:50]}
    with lock:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
    return rec


todo = [a for a in miss if a not in done]
print("待跑 %d / 共 %d" % (len(todo), len(miss)))
with ThreadPoolExecutor(max_workers=10) as ex:
    for fut in as_completed([ex.submit(work, a) for a in todo]):
        fut.result()
fh.close()

A30 = B = 0; samp = collections.defaultdict(list)
for ln in open(OUT):
    r = json.loads(ln)
    if r["found30"]:
        A30 += 1; samp["A"].append((r["ask"], r["value"]))
    else:
        B += 1; samp["B"].append((r["ask"], ""))
print("真·miss %d 量化:" % (A30 + B))
print("  A 答案在篇·只排 9-30(boilerplate 挤出 top-8 → 过滤可救): %d" % A30)
print("  B 篇内 30 块也没有(不在文档/scope错/judge抽不出):       %d" % B)
for c in ("A", "B"):
    print("=== %s 样本 ===" % c)
    for ask, v in samp[c][:16]:
        print("  %s%s" % (ask[:46], ("  →" + v) if v else ""))
