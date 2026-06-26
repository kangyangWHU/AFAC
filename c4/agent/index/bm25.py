"""BM25 词法检索（纯算法，零模型，合规无风险）。
jieba 分词 + 金融词典；支持按 doc_ids 子集检索。索引可持久化到磁盘。
"""
from __future__ import annotations
import os
import pickle
import re
import jieba
from rank_bm25 import BM25Okapi
from .base import Retriever, Hit
from .. import config

# 金融领域高频术语，避免被 jieba 切碎
_FIN_TERMS = ["营业收入", "营业总收入", "归母净利润", "归属于上市公司股东的净利润",
              "经营活动产生的现金流量净额", "研发投入", "研发费用", "现金分红",
              "受益所有人", "尽职调查", "特别决议", "普通决议", "资产负债率",
              "身故保险金", "退保金额", "现金价值", "保单账户价值", "复合增速",
              "银保渠道", "募集资金", "受托管理人", "主体信用评级"]
for _t in _FIN_TERMS:
    jieba.add_word(_t)

_STOP = set("的 了 和 与 及 或 在 是 对 为 以 上 下 中 等 之 其 该 本 个 我 会 并 "
            "并且 以及 根据 关于 下列 哪些 正确 说法 描述 选项".split())
_TOKEN = re.compile(r"[\w一-鿿%]+")


def tokenize(text: str, drop_stop: bool = True, mode: str = "jieba") -> list[str]:
    if mode == "char":
        toks = [c for c in text if _TOKEN.match(c)]
    else:
        toks = [t for t in jieba.lcut(text) if t.strip() and _TOKEN.match(t)]
    if drop_stop:
        toks = [t for t in toks if t not in _STOP]
    return toks


class BM25Index(Retriever):
    def __init__(self):
        cfg = config.load()["bm25"]
        self.k1 = cfg["k1"]
        self.b = cfg["b"]
        self.topk = cfg["topk"]
        self.mode = cfg.get("tokenizer", "jieba")
        self.drop_stop = cfg.get("drop_stopwords", True)
        self.index_breadcrumb = cfg.get("index_breadcrumb", False)  # breadcrumb 进不进索引词
        self.chunks: list[dict] = []
        self.by_doc: dict[str, list[int]] = {}
        self._bm25: BM25Okapi | None = None
        self._tokens: list[list[str]] = []

    def _index_text(self, c: dict) -> str:
        bc = c.get("breadcrumb", "")
        return f"{bc} {c['text']}" if (self.index_breadcrumb and bc) else c["text"]

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        self._tokens = [tokenize(self._index_text(c), self.drop_stop, self.mode) for c in chunks]
        self.by_doc = {}
        for i, c in enumerate(chunks):
            self.by_doc.setdefault(c["doc_id"], []).append(i)
        self._bm25 = BM25Okapi(self._tokens, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, k: int | None = None,
               doc_ids: list[str] | None = None) -> list[Hit]:
        assert self._bm25 is not None, "index not built"
        k = k or self.topk
        q = list(dict.fromkeys(tokenize(query, self.drop_stop, self.mode)))  # query 去重: BM25逐token累加, 重复词会双倍加权
        scores = self._bm25.get_scores(q)
        if doc_ids:
            cand = [i for d in doc_ids for i in self.by_doc.get(d, [])]
        else:
            cand = range(len(self.chunks))
        ranked = sorted(cand, key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in ranked:
            c = self.chunks[i]
            out.append(Hit(chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                           score=float(scores[i]), text=c["text"],
                           meta={"type": c.get("type"), "section_path": c.get("section_path"),
                                 "article_no": c.get("article_no"), "page": c.get("page"),
                                 "seq": c.get("seq"), "breadcrumb": c.get("breadcrumb", "")}))
        return out

    def search_local(self, query: str, doc_ids: list[str],
                     k: int | None = None) -> list[Hit]:
        """per-candidate BM25：只在候选文档块上【重算 IDF】再检索。
        修全局 IDF 的坑——公司名/年份等"全局稀有但篇内遍地"的词在候选集内 df 高→IDF 低→
        不再带偏排序(如"比亚迪"不再把子公司表顶上来)。token 复用全局, 建子索引 ~10ms。"""
        k = k or self.topk
        cand = [i for d in doc_ids for i in self.by_doc.get(str(d), [])]
        if not cand:
            return []
        sub = BM25Okapi([self._tokens[i] for i in cand], k1=self.k1, b=self.b)
        scores = sub.get_scores(list(dict.fromkeys(tokenize(query, self.drop_stop, self.mode))))  # query 去重
        order = sorted(range(len(cand)), key=lambda j: scores[j], reverse=True)[:k]
        out = []
        for j in order:
            c = self.chunks[cand[j]]
            out.append(Hit(chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                           score=float(scores[j]), text=c["text"],
                           meta={"type": c.get("type"), "section_path": c.get("section_path"),
                                 "article_no": c.get("article_no"), "page": c.get("page"),
                                 "seq": c.get("seq"), "breadcrumb": c.get("breadcrumb", "")}))
        return out

    def hits_from_ids(self, chunk_ids: list[str]) -> list[Hit]:
        """从 chunk_id 列表(按序)重建 Hit(score 置0; 检索缓存命中时用——loop 只用顺序+text+meta, 不用分值)。"""
        m = getattr(self, "_by_cid", None)
        if m is None:
            m = self._by_cid = {c["chunk_id"]: c for c in self.chunks}
        out = []
        for cid in chunk_ids:
            c = m.get(cid)
            if c:
                out.append(Hit(chunk_id=c["chunk_id"], doc_id=c["doc_id"], score=0.0, text=c["text"],
                               meta={"type": c.get("type"), "section_path": c.get("section_path"),
                                     "article_no": c.get("article_no"), "page": c.get("page"),
                                     "seq": c.get("seq"), "breadcrumb": c.get("breadcrumb", "")}))
        return out

    def neighbors(self, doc_id: str, seq: int, window: int = 1) -> list[dict]:
        """按文档内 seq 取命中块的前后兄弟块（small-to-big 父块回填）。"""
        idxs = self.by_doc.get(doc_id, [])
        out = []
        for i in idxs:
            s = self.chunks[i].get("seq")
            if s is not None and 0 < abs(s - seq) <= window:
                out.append(self.chunks[i])
        return out

    # 持久化
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "tokens": self._tokens,
                         "by_doc": self.by_doc, "k1": self.k1, "b": self.b}, f)

    @classmethod
    def from_file(cls, path: str) -> "BM25Index":
        idx = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx.chunks = d["chunks"]
        idx._tokens = d["tokens"]
        idx.by_doc = d["by_doc"]
        idx._bm25 = BM25Okapi(idx._tokens, k1=d["k1"], b=d["b"])
        return idx
