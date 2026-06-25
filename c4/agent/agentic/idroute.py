"""语义身份路由：把一条 fact 的 ask 按【语义相似度】匹配到候选文档的【身份】(doc_identities.json)。
治内容路由的坑：泛词("院外特定药品")会把"安佑福重疾险"的 fact 路由到含该泛词更多的 e生保。
身份是 LLM 读文档开头生成的产品/主体全称(见 script/build_doc_identities.py)，用嵌入余弦比相似度——
语义匹配("安佑福重疾险"↔"平安安佑福重大疾病保险")而非字面匹配，红鲱鱼泛词不再带偏。
只在【置信(高相似+明显领先)】时定篇，否则返回 None 让调用方回退内容路由(避免对无命名实体的泛 fact 硬猜)。
"""
from __future__ import annotations
import os
import json
import math
import threading
import requests
from .. import config


class IdRouter:
    def __init__(self):
        a = config.load().get("agentic", {})
        self.url = a.get("embed_url", "http://localhost:8000/v1/embeddings")
        self.model = a.get("embed_model", "Qwen3-Embedding-8B")
        self.id_min = a.get("route_id_min", 0.58)        # 置信阈：top 相似度需 ≥ 此值
        self.id_margin = a.get("route_id_margin", 0.07)  # 领先阈：top 比次高至少高出此值
        self.lit_min = a.get("route_id_lit_min", 5)      # 字面阈：ask 里出现某篇身份的≥此长公共子串(产品/主体名)→直接定篇
        p = os.path.join(config.path("index_dir"), "doc_identities.json")
        self.identities = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        self.ok = bool(self.identities)
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _embed(self, texts: list[str]) -> list | None:
        need = [t for t in texts if t not in self._cache]
        if need:
            try:
                r = requests.post(self.url, json={"model": self.model, "input": need}, timeout=20)
                data = r.json()["data"]
            except Exception:
                return None
            with self._lock:
                for t, d in zip(need, data):
                    self._cache[t] = d["embedding"]
        return [self._cache.get(t) for t in texts]

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return s / (na * nb + 1e-9)

    @staticmethod
    def _lcs(a: str, b: str) -> int:
        """最长公共子串长度(字面匹配用): ask 与某篇身份名共有的最长连续片段。"""
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        best = 0
        for i in range(1, len(a) + 1):
            cur = [0] * (len(b) + 1)
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
            prev = cur
        return best

    def route(self, ask: str, cands: list[str]) -> list[str] | None:
        """置信时返回 [单篇]，否则 None(回退内容路由)。"""
        if not self.ok:
            return None
        ids = {d: self.identities.get(str(d), "") for d in cands}
        named = [d for d in cands if ids[d]]
        if len(named) < 2:
            return None
        # 字面优先: ask 里出现某篇身份的【≥lit_min 长公共子串】且【唯一】→ 直接定篇。
        # 治 A(近重复产品 embedding 被 margin 挡) 和 B(案例情景词把别篇顶上来: 产品名子串长、情景词短)。
        # 多篇都强匹配(如同公司多年报、名称互含)→ 字面不唯一, 落到嵌入消歧。
        lit = sorted(((d, self._lcs(ask, ids[d])) for d in named), key=lambda x: x[1], reverse=True)
        if lit[0][1] >= self.lit_min and (len(lit) < 2 or lit[1][1] < self.lit_min):
            return [str(lit[0][0])]
        vecs = self._embed([ask] + [ids[d] for d in named])
        if not vecs or any(v is None for v in vecs):
            return None
        q = vecs[0]
        sims = {d: self._cos(q, vecs[i + 1]) for i, d in enumerate(named)}
        ranked = sorted(sims, key=lambda d: sims[d], reverse=True)
        top = ranked[0]
        if sims[top] >= self.id_min and sims[top] - sims[ranked[1]] >= self.id_margin:
            return [str(top)]
        return None
