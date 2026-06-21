"""事实 → 文档 路由（plan.md §v4）。给一条原子事实的 ask + 候选文档，选最可能含它的那篇。
本质就是 B 榜文档路由（query→doc）的同一思路，只是候选被题目的 doc_ids 限定（少而已）。
纯 BM25 文档级打分，零模型；分解器只管产 ask，绑哪篇文档是这里的事（独立于分解）。
"""
from __future__ import annotations
import re
from ..index.bm25 import BM25Index
from .. import config

# 序数引用 → 候选文档的第几篇（同类文档靠"第一份/第二份"区分, 内容路由分不出, 按顺序映射）
_ORDINALS = [(re.compile(r"第一份|第1份|第一篇|前者|首份"), 0),
             (re.compile(r"第二份|第2份|第二篇|后者"), 1),
             (re.compile(r"第三份|第3份|第三篇"), 2),
             (re.compile(r"第四份|第4份|第四篇"), 3)]


def route(index: BM25Index, ask: str, candidate_docs: list[str]) -> list[str]:
    """返回最可能含该事实的 doc_ids（按分排序）。
    ① ask 里有"第N份"序数 → 直接取候选第 N 篇（同类文档靠顺序区分，内容分不出）。
    ② 否则按内容打分 = 该 doc 内 chunk 对 ask 的 BM25 分之和；某篇明显领先就缩到 1 篇。"""
    cands = [str(d) for d in candidate_docs]
    if len(cands) <= 1:
        return cands
    for pat, i in _ORDINALS:
        if pat.search(ask) and i < len(cands):
            return [cands[i]]
    # 年份引用 → 导向 doc_id 含该年份的那篇(财报命名带年份: annual_xxx_2024_report)
    for y in re.findall(r"20\d{2}", ask):
        hit = [d for d in cands if y in d]
        if len(hit) == 1:
            return hit
    a = config.load().get("agentic", {})
    chunk_k = a.get("route_chunk_k", 30)        # 每篇取多少 chunk 聚合打分
    min_lead = a.get("route_min_lead", 1.3)     # top 领先次高的倍数阈值, 够大才缩到1篇
    top_k = a.get("route_top_k", 2)             # 不够领先时返回前几篇

    score: dict[str, float] = {}
    for h in index.search(ask, k=chunk_k, doc_ids=cands):
        score[h.doc_id] = score.get(h.doc_id, 0.0) + h.score
    if not score:
        return cands
    ranked = sorted(score, key=lambda d: score[d], reverse=True)
    top = ranked[0]
    if len(ranked) == 1 or score[top] >= min_lead * score[ranked[1]]:
        return [top]
    return ranked[:top_k]
