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
    ② 否则按内容打分 = 该 doc【最佳几块】的 BM25 分（max-pool）；某篇明显领先就缩到 1 篇。
    ③ 内容分不开时（同公司不同年份的财报，正文几乎一样）才用年份判别。

    为何 max-pool 而非求和：路由只需"哪篇有【最好的证据块】"——一块命中就够。求和问的是
    "哪篇整体提这些词最多"，会被【块数 × 泛词密度】稀释：100 块的伤残表文档因满篇"保险"在
    求和上压过只在封面出现一次产品名的正解文档。max-pool 天然尊重 IDF——一处高 IDF 的产品名
    命中胜过一堆低 IDF 的"保险"，无需任何停用词表。
    注：年份是【弱信号】，排在内容之后——doc_id 带【报告年】、ask 常引【财年】，二者错位
       （比亚迪 2024 财年数据在 annual_byd_2025_report），年份抢跑会误导到同年份他司文档。"""
    cands = [str(d) for d in candidate_docs]
    if len(cands) <= 1:
        return cands
    for pat, i in _ORDINALS:
        if pat.search(ask) and i < len(cands):
            return [cands[i]]
    a = config.load().get("agentic", {})
    pool_n = a.get("route_pool_n", 3)           # 每篇取最佳 N 块求和(max-pool变体, 抗块数偏置)
    min_lead = a.get("route_min_lead", 1.3)     # top 领先次高的倍数阈值, 够大才缩到1篇
    top_k = a.get("route_top_k", 2)             # 不够领先时返回前几篇

    # 取足够多块, 保证每个候选都有 ≥pool_n 个代表(search 是跨候选全局排序)
    perdoc: dict[str, list[float]] = {}
    for h in index.search(ask, k=max(30, 20 * len(cands)), doc_ids=cands):
        perdoc.setdefault(h.doc_id, []).append(h.score)
    if not perdoc:
        return cands
    score = {d: sum(sorted(v, reverse=True)[:pool_n]) for d, v in perdoc.items()}
    ranked = sorted(score, key=lambda d: score[d], reverse=True)
    top = ranked[0]
    if len(ranked) == 1 or score[top] >= min_lead * score[ranked[1]]:
        return [top]
    # 内容分不开 → 仅当前两名是【同公司不同年】(去年份后同名)才用年份判别;
    # 否则尊重内容排序——年份决不能翻掉跨公司的内容判断(否则"比亚迪2024"又被含2024的他司文档夺走)
    stem = lambda d: re.sub(r"20\d{2}", "", d)
    if len(ranked) >= 2 and stem(ranked[0]) == stem(ranked[1]):
        for y in re.findall(r"20\d{2}", ask):
            hit = [d for d in ranked[:top_k] if y in d]
            if len(hit) == 1:
                return hit
    return ranked[:top_k]
