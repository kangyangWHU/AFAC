"""检索评测：对 out/decompose_route.json 每条 fact 取一批(loop_batch=8)块，测 top1 BM25 分 + 关键词覆盖，
统计【真检索失败】(top1<1.5 且 cov<0.6 —— 既没强命中、内容也不在召回块里)。
注意：cache_retrieve=False。BM25 缓存重建块 score=0（loop 只用顺序，不用分值），开缓存会污染分值分析。
genQuery 默认关 → 用裸 ask 检索（与线上一致）。用法：python eval/retrieve.py
"""
import json, re, sys, os
sys.path.insert(0, "/home/yk/code/AFAC/c4"); os.chdir("/home/yk/code/AFAC/c4")
from agent.index.bm25 import BM25Index, tokenize
from agent.agentic.loop import SubQLoop
from agent.agentic.solver import _DOC_REF

SCEN = re.compile(r"[王李张赵刘陈]某|某人")        # 案例情景人物(违铁律5, 不该是检索 fact)
GEN = set("是 多少 是否 的 在 年 是什么 预计 为 元 中 有 哪 项 该 了 文档 报告 占 与".split())
kt = lambda a: [t for t in tokenize(a, True, "jieba") if t not in GEN and len(t) > 1]

idx = BM25Index.from_file("index/bm25.pkl")
loop = SubQLoop(None, idx); loop.cache_retrieve = False
A = json.load(open("out/decompose_route.json"))

n = 0; cov_sum = 0.0; fails = []
for e in A:
    for f in e["facts"]:
        if SCEN.search(f["ask"]):
            continue
        n += 1
        ask = _DOC_REF.sub("", f["ask"]).strip()        # 剥位置词/text0N 标签, 同线上
        hits = loop._retrieve(ask, f["route"].split("/"))[:8]
        top1 = hits[0].score if hits else 0.0
        body = " ".join(h.text for h in hits); k = set(kt(ask))
        cov = (sum(1 for t in k if t in body) / len(k)) if k else 1.0
        cov_sum += cov
        if top1 < 1.5 and cov < 0.6:
            fails.append((e["qid"], round(top1, 1), round(cov, 2), ask[:36]))

print("legit facts %d | 关键词覆盖均 %.2f | 真检索失败 %d (%.1f%%)" % (n, cov_sum / n, len(fails), 100 * len(fails) / n))
print("(top1<1.5 但 cov 高 = IDF 归零的伪失败：块已召回，judge 照读)")
for q, t, c, a in fails:
    print("  %-10s top1=%.1f cov=%.2f  %s" % (q, t, c, a))
