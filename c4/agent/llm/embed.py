"""Qwen 系 embedding（百炼 text-embedding-v4）。用于 B 榜语义路由（合规：Qwen 系向量）。
带磁盘缓存，避免重复嵌入语料。token 计入全局 USAGE。
"""
from __future__ import annotations
import os
import json
import hashlib
from openai import OpenAI
from .base import USAGE
from .. import config


class Embedder:
    """backend=local: 本地 Qwen3-Embedding(sentence-transformers, 零 API token, 合规);
       backend=bailian: 百炼 text-embedding-v4(需 key, API token 计入)。"""
    def __init__(self, cache_path: str | None = None):
        e = config.load().get("embed", {})
        self.backend = e.get("backend", "local")
        self.batch = e.get("batch", 32)
        self.dim = e.get("dim", 1024)
        self._st = None
        self._client = None
        if self.backend == "local":
            from sentence_transformers import SentenceTransformer
            path = e.get("local_path") or _modelscope_path(
                e.get("local_model", "Qwen/Qwen3-Embedding-0.6B"))
            # 默认 CPU：GPU 被本地 vLLM 占满。573 签名+query 量小，CPU 可接受。
            self._st = SentenceTransformer(path, device=e.get("device", "cpu"))
        else:
            self.model = e.get("model", "text-embedding-v4")
            self._client = OpenAI(
                base_url=e.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                api_key=config.resolve_key(e.get("api_key")))
        self.cache_path = cache_path or os.path.join(config.path("index_dir"),
                                                      f"emb_cache_{self.backend}.json")
        self._cache: dict[str, list[float]] = {}
        if os.path.exists(self.cache_path):
            self._cache = json.load(open(self.cache_path, encoding="utf-8"))

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        need = [t for t in texts if self._key(t) not in self._cache]
        for i in range(0, len(need), self.batch):
            chunk = need[i:i + self.batch]
            if self.backend == "local":
                vecs = self._st.encode(chunk, normalize_embeddings=True,
                                       batch_size=self.batch)
                for t, v in zip(chunk, vecs):
                    self._cache[self._key(t)] = [float(x) for x in v]
            else:
                r = self._client.embeddings.create(model=self.model, input=chunk,
                                                    dimensions=self.dim)
                if r.usage:
                    USAGE.add(r.usage.prompt_tokens, 0)
                for t, d in zip(chunk, r.data):
                    self._cache[self._key(t)] = d.embedding
        return [self._cache[self._key(t)] for t in texts]

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        json.dump(self._cache, open(self.cache_path, "w"))


def _modelscope_path(model_id: str) -> str:
    from modelscope import snapshot_download
    return snapshot_download(model_id)


def cosine(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return s / (na * nb + 1e-9)
