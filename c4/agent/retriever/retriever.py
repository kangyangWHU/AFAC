"""检索编排：把题目 -> 一组高质量证据块。
策略（全部 config 驱动）：
  1) 逐选项 query（multi/tf 漏召回=全错）
  2) 每候选文档保底召回 min_per_doc 块（多文档题防漏）
  3) 父块回填（small-to-big）补全上下文
  4) 目录/空壳过滤、去重、按分排序、max_chunks 截断（控 token）
"""
from __future__ import annotations
from dataclasses import dataclass
from .. import config
from ..index.base import Retriever, Hit
from ..index.synonyms import expand_query
from .filters import low_value


@dataclass
class Evidence:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    article_no: str | None
    page: int | None
    type: str


class EvidenceRetriever:
    def __init__(self, index: Retriever):
        self.index = index
        r = config.load()["retrieval"]
        self.per_option = r.get("per_option_query", True)
        self.min_per_doc = r.get("min_per_doc", 2)
        self.max_chunks = r.get("max_chunks", 12)
        self.per_query_k = r.get("per_query_k", 4)
        self.backfill = r.get("backfill_parent", True)
        self.filter_lv = r.get("filter_low_value", True)
        self.window = config.get("chunker.parent_window", 1)
        self.metric_queries = r.get("metric_queries", {})
        self.metric_per_doc = r.get("metric_per_doc", 3)
        self.per_option_per_doc = r.get("per_option_per_doc", False)
        self.per_option_doc_k = r.get("per_option_doc_k", 3)
        self.expand_syn = r.get("expand_synonyms", False)  # query同义词扩展(治词法不匹配漏召)
        # 财报结构化摘要（主要会计数据），财报题保底注入
        import os
        import json as _json
        fp = os.path.join(config.path("index_dir"), "fin_summaries.json")
        self.fin_summaries = _json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}

    def retrieve(self, question: str, options: dict[str, str],
                 doc_ids: list[str] | None = None,
                 domain: str | None = None,
                 min_per_doc: int | None = None) -> list[Evidence]:
        min_pd = min_per_doc if min_per_doc is not None else self.min_per_doc
        queries = [question]
        if self.per_option:
            queries += list(options.values())
        if self.expand_syn:
            # 选项/题干用词可能与文档不同(保单贷款vs借款)，补同义词命中被漏召的支撑块
            queries = [expand_query(q) for q in queries]

        best: dict[str, Hit] = {}        # chunk_id -> best Hit
        reserved: set[str] = set()       # 指标定向/保底块，免遭 max_chunks 截断
        doc_hits: dict[str, list[Hit]] = {}
        def _absorb(h, reserve=False):
            if self.filter_lv and low_value(h.text):
                return
            if h.chunk_id not in best or h.score > best[h.chunk_id].score:
                best[h.chunk_id] = h
            doc_hits.setdefault(h.doc_id, []).append(h)
            if reserve:
                reserved.add(h.chunk_id)

        for q in queries:
            if self.per_option_per_doc and doc_ids:
                # 每个选项/题干 在【每篇文档内】单独检索：保证 option 证据在每篇里都被找到，
                # 不被高频财务表/另一篇挤掉。每(query×doc)首选块 reserve，免遭截断。
                for d in doc_ids:
                    for rank, h in enumerate(self.index.search(
                            q, k=self.per_option_doc_k, doc_ids=[d])):
                        _absorb(h, reserve=(rank == 0))
            else:
                for h in self.index.search(q, k=self.per_query_k, doc_ids=doc_ids):
                    _absorb(h)

        # 指标定向：对比题里另一文档常被选项文本带偏而漏掉关键表。
        # 对每个候选文档用域指标 query 单独检索，保证关键指标表被各文档都拉到。
        mq = self.metric_queries.get(domain or "")
        if mq and doc_ids:
            for d in doc_ids:
                for h in self.index.search(mq, k=self.metric_per_doc, doc_ids=[d]):
                    if self.filter_lv and low_value(h.text):
                        continue
                    if h.chunk_id not in best or h.score > best[h.chunk_id].score:
                        best[h.chunk_id] = h
                    reserved.add(h.chunk_id)

        # 每文档保底：确保每个候选文档贡献 >= min_per_doc 块。
        # 若某文档在逐选项检索中零命中（被其他文档挤掉），主动对该文档单独检索。
        if doc_ids:
            combined = question + " " + " ".join(options.values())
            for d in doc_ids:
                got = {cid for cid, h in best.items() if h.doc_id == d}
                if len(got) >= min_pd:
                    continue
                for h in self.index.search(combined, k=min_pd, doc_ids=[d]):
                    if self.filter_lv and low_value(h.text):
                        continue
                    best.setdefault(h.chunk_id, h)
                    reserved.add(h.chunk_id)

        # 父块回填
        if self.backfill and hasattr(self.index, "neighbors"):
            for h in list(best.values()):
                seq = h.meta.get("seq")
                if seq is None:
                    continue
                for nb in self.index.neighbors(h.doc_id, seq, self.window):  # type: ignore
                    if self.filter_lv and low_value(nb["text"]):
                        continue
                    best.setdefault(nb["chunk_id"], Hit(
                        chunk_id=nb["chunk_id"], doc_id=nb["doc_id"],
                        score=h.score * 0.5, text=nb["text"],
                        meta={"type": nb.get("type"), "article_no": nb.get("article_no"),
                              "page": nb.get("page"), "seq": nb.get("seq")}))

        # 最终选择：先为每个候选文档预留 min_per_doc 高分块（防被高分文档挤掉），
        # 再用剩余预算按分数补满，保证多文档题不漏 doc。
        all_hits = sorted(best.values(), key=lambda x: x.score, reverse=True)
        selected: dict[str, Hit] = {}
        # 1) 指标定向/选项定向保底块优先纳入（按分数，最多占满 max_chunks）
        for h in all_hits:
            if h.chunk_id in reserved and len(selected) < self.max_chunks:
                selected[h.chunk_id] = h
        # 2) 每文档再补足 min_per_doc 高分块
        if doc_ids:
            for d in doc_ids:
                dh = [h for h in all_hits if h.doc_id == d][:min_pd]
                for h in dh:
                    selected[h.chunk_id] = h
        # 3) 剩余预算按分数补满
        for h in all_hits:
            if len(selected) >= self.max_chunks:
                break
            selected.setdefault(h.chunk_id, h)
        ranked = list(selected.values())
        # 同文档内按 seq 排序，证据更连贯
        ranked.sort(key=lambda x: (x.doc_id, x.meta.get("seq") or 0))
        out = [Evidence(chunk_id=h.chunk_id, doc_id=h.doc_id, text=h.text,
                        score=h.score, article_no=h.meta.get("article_no"),
                        page=h.meta.get("page"), type=h.meta.get("type") or "text")
               for h in ranked]
        # 财报题：保底注入"主要会计数据"结构化摘要（放最前，作权威数值来源）
        if domain == "financial_reports" and doc_ids:
            inj = [Evidence(chunk_id=f"{d}::fin_summary", doc_id=d,
                            text=self.fin_summaries[d], score=1e9,
                            article_no=None, page=None, type="fin_summary")
                   for d in doc_ids if d in self.fin_summaries]
            out = inj + out
        return out
