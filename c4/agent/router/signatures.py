"""文档级签名：标题 + 章节目录(heading) + 条标题。
路由用——比正文 chunk 更紧凑、判别力更强的结构信号。纯 BM25，零模型，泛化。
"""
from __future__ import annotations
import os
import json
import glob
from ..index.bm25 import tokenize


def build_signatures(processed_dir: str) -> list[dict]:
    """每篇 doc -> {doc_id, domain, text}。text = 标题 + 所有 heading + 条标题。"""
    out = []
    for f in glob.glob(os.path.join(processed_dir, "*", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        parts = [d.get("title", "")]
        secs = []
        for b in d["blocks"]:
            if b["type"] == "heading":
                parts.append(b["text"])
            if b.get("article_no"):
                # 条标题：取条文首句(到第一个句号/分号)
                head = b["text"].split("\n")[0][:30]
                parts.append(head)
            for s in b.get("section_path", []):
                secs.append(s)
        parts += list(dict.fromkeys(secs))      # 去重的章节路径
        text = " ".join(p for p in parts if p).strip()
        out.append({"doc_id": d["doc_id"], "domain": d["domain"], "text": text[:2000]})
    return out


class SignatureIndex:
    """文档粒度 BM25（每篇 doc 一条签名）。"""
    def __init__(self, sigs: list[dict]):
        from rank_bm25 import BM25Okapi
        self.sigs = sigs
        self.doc_ids = [s["doc_id"] for s in sigs]
        self.by_domain: dict[str, list[int]] = {}
        for i, s in enumerate(sigs):
            self.by_domain.setdefault(s["domain"], []).append(i)
        self._tokens = [tokenize(s["text"]) for s in sigs]
        self._bm25 = BM25Okapi(self._tokens)

    def rank(self, query: str, domain: str) -> list[tuple[str, float]]:
        """返回域内文档按签名相关度排序 [(doc_id, score), ...]（含 0 分）。"""
        scores = self._bm25.get_scores(tokenize(query))
        idxs = self.by_domain.get(domain, [])
        ranked = sorted(idxs, key=lambda i: scores[i], reverse=True)
        return [(self.doc_ids[i], float(scores[i])) for i in ranked]

    def save(self, path: str):
        json.dump(self.sigs, open(path, "w"), ensure_ascii=False)

    @classmethod
    def from_file(cls, path: str) -> "SignatureIndex":
        return cls(json.load(open(path, encoding="utf-8")))
