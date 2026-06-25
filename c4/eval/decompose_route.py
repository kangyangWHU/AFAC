"""分解 + 路由评测：对全部题分解，每条 fact 路由到【篇组】，存 out/decompose_route.json + 打印来源分布。
路由四类：pin-decompose(题干显式位置/名) | pin-idrouter(实体字面命中) | group-together(单点查找整组) | agg-split(聚合题逐篇)。
用法：python eval/decompose_route.py   （默认本地 gpt-oss，免费）
"""
import json, sys, os, glob, collections
sys.path.insert(0, "/home/yk/code/AFAC/c4"); os.chdir("/home/yk/code/AFAC/c4")
from agent import config
config.load()["llm"].update(base_url="http://localhost:30000/v1", model="openai/gpt-oss-120b", api_key="EMPTY")
from agent.agentic.solver import AgenticSolver, _AGG_RE
from concurrent.futures import ThreadPoolExecutor, as_completed

S = AgenticSolver()
Q = []
for f in sorted(glob.glob(config.path("questions_dir") + "/*.json")):
    Q += json.load(open(f, encoding="utf-8"))


def route_fact(f, cand, agg_q):
    """复刻 solver._fact_docs 的判定，并标注来源类别。"""
    docs = f.get("doc") or cand
    docs = [d for d in (docs if isinstance(docs, list) else [docs]) if d in cand] or cand
    if len(docs) == 1:
        return "pin-decompose", [docs]
    if agg_q:
        return "agg-split", [[c] for c in docs]
    pin = S.idrouter.route(f["ask"], docs)
    return ("pin-idrouter", [pin]) if pin else ("group-together", [docs])


res = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for fut in as_completed([ex.submit(lambda q: (q, S.decomposer.decompose(q, q["doc_ids"])), q) for q in Q]):
        res.append(fut.result())

src = collections.Counter(); save = []
for q, d in sorted(res, key=lambda x: x[0]["qid"]):
    cand = [str(x) for x in q["doc_ids"]]
    agg_q = any(_AGG_RE.search(str(t or "")) for t in q["options"].values())
    seen, facts = set(), []
    for f in d["facts"]:
        s, groups = route_fact(f, cand, agg_q)
        for g in groups:
            k = (f["ask"], tuple(g))
            if k in seen:
                continue
            seen.add(k); src[s] += 1
            facts.append({"ask": f["ask"], "route": "/".join(g), "src": s})
    save.append({"qid": q["qid"], "agg": agg_q, "question": q["question"], "doc_ids": cand, "facts": facts})

json.dump(save, open("out/decompose_route.json", "w"), ensure_ascii=False, indent=1)
print("saved out/decompose_route.json  (%d 题, %d loop-jobs)" % (len(save), sum(src.values())))
for k, v in src.most_common():
    print("  %-16s %d" % (k, v))
