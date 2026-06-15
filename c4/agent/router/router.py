"""B 榜文档路由器：domain + 题目 -> 候选 doc_ids（不给 doc_ids 时用）。
纯 BM25，零模型，合规。打分（实测最优）：
  基础 = 正文 chunk BM25 分之和（保留分数量级，比 rank/max 强）；
  ① 文档签名(标题+目录+条标题)命中 → 对该 doc 做**乘性加成**（判别"这篇是关于什么的"）。
注：曾试 max-passage / RRF 融合，会丢掉 BM25 分数量级，反而掉点(全@5 74→46)，已弃用。
"""
from __future__ import annotations
import os
import collections
from ..index.base import Retriever
from ..import config
from .signatures import SignatureIndex


class DocRouter:
    def __init__(self, index: Retriever, sig_index: SignatureIndex | None = None):
        self.index = index
        self.domain_docs: dict[str, set[str]] = collections.defaultdict(set)
        for c in index.chunks:                       # type: ignore[attr-defined]
            self.domain_docs[c["domain"]].add(c["doc_id"])
        if sig_index is None:
            sp = os.path.join(config.path("index_dir"), "doc_signatures.json")
            sig_index = SignatureIndex.from_file(sp) if os.path.exists(sp) else None
        self.sig = sig_index
        r = config.load()["retrieval"]
        self.per_option = r.get("per_option_query", True)
        self.route_chunk_k = config.get("router.chunk_k", 30)
        self.sig_weight = config.get("router.sig_weight", 1.0)

    def _score(self, question: str, options: dict[str, str],
               domain: str) -> list[str]:
        cand = self.domain_docs.get(domain, set())
        if not cand:
            return []
        queries = [question]
        if self.per_option:
            queries += list(options.values())

        score: dict[str, float] = collections.defaultdict(float)
        for q in queries:
            # 基础：正文 chunk BM25 分之和
            csum: dict[str, float] = collections.defaultdict(float)
            for h in self.index.search(q, k=self.route_chunk_k, doc_ids=list(cand)):
                csum[h.doc_id] += h.score
            # ① 文档签名命中 → 乘性加成
            sig_hit = set()
            if self.sig is not None:
                sig_hit = {d for d, sc in self.sig.rank(q, domain) if sc > 0}
            for d, s in csum.items():
                score[d] += s * (1 + self.sig_weight * (d in sig_hit))

        ranked = sorted(score, key=lambda d: score[d], reverse=True)
        ranked += [d for d in cand if d not in score]
        return ranked

    def select(self, question: str, options: dict[str, str], domain: str,
               k: int = 5) -> list[str]:
        cand = self.domain_docs.get(domain, set())
        if len(cand) <= k:
            return list(cand)
        return self._score(question, options, domain)[:k]

    def rank(self, question: str, options: dict[str, str], domain: str) -> list[str]:
        return self._score(question, options, domain)
