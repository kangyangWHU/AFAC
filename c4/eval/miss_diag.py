"""诊断 judge 单批未命中(不 re-query —— 8 块本应够)。对每条【真·miss】(该 ask 任一篇都没抽到值)，
在它路由到的每篇里取 top-8 / top-30，看 ask 关键词覆盖，定位失败点：
  A 答案块在前8 → judge 没抽出(judge/格式问题)
  B 答案在篇但排在 8 名之外 → 检索排序没把它排进 batch
  D 答案不在路由篇 → 路由错 或 文档根本没这数据(distractor)
打印分类 + 全部 65 条(带覆盖 + best 篇 top-1 块片段供眼检)。用法：python eval/miss_diag.py
"""
import json, sys, os, collections
sys.path.insert(0, "/home/yk/code/AFAC/c4"); os.chdir("/home/yk/code/AFAC/c4")
from agent.index.bm25 import BM25Index, tokenize
from agent.agentic.loop import SubQLoop
from agent.agentic.solver import _DOC_REF

idx = BM25Index.from_file("processed_vl/bm25_vl.pkl"); loop = SubQLoop(None, idx); loop.cache_retrieve = False
A = json.load(open("out/decompose_route.json"))
GEN = set("是 多少 是否 的 在 年 是什么 预计 为 元 中 有 哪 项 该 了 文档 报告 占 与".split())
kt = lambda a: [t for t in tokenize(a, True, "jieba") if t not in GEN and len(t) > 1]

byask = collections.defaultdict(lambda: [0, 0])
for ln in open("out/judge.jsonl"):
    r = json.loads(ln); byask[r["ask"]][1] += 1; byask[r["ask"]][0] += int(r["found"])
miss = [a for a, (fo, _t) in byask.items() if fo == 0]

ask2routes = collections.defaultdict(set)
for e in A:
    for f in e["facts"]:
        ask2routes[_DOC_REF.sub("", f["ask"]).strip()].add(f["route"])


def best(ask):
    docs = sorted({d for r in ask2routes.get(ask, []) for d in r.split("/")})
    toks = set(kt(ask)); b8 = b30 = 0.0; top = ""
    for d in docs:
        h = loop._retrieve(ask, [d])[:30]
        body8 = " ".join(x.text for x in h[:8]); body30 = " ".join(x.text for x in h)
        c8 = sum(1 for t in toks if t in body8) / max(len(toks), 1)
        c30 = sum(1 for t in toks if t in body30) / max(len(toks), 1)
        if c8 >= b8:
            b8, b30, top = c8, c30, (h[0].text[:56].replace("\n", " ") if h else "")
    return b8, b30, top


out = []
for ask in miss:
    c8, c30, top = best(ask)
    cat = "A judge没抽(块在前8)" if c8 >= 0.5 else ("B 检索排序(答案>8名)" if c30 >= 0.5 else "D 不在路由篇")
    out.append((cat, ask, c8, c30, top))

cc = collections.Counter(c for c, *_ in out)
print("真·miss %d 条(8块, 不requery)分类:" % len(miss))
for c, n in sorted(cc.items()):
    print("  %-26s %d" % (c, n))
print("=== 全部 ===")
for cat, ask, c8, c30, top in sorted(out):
    print("[%s] c8=%.2f c30=%.2f %-30s| %s" % (cat[0], c8, c30, ask[:30], top))
