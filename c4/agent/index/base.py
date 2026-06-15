"""检索器抽象接口 + 命中结构。BM25/Dense/Hybrid 均实现 Retriever。"""
from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    meta: dict


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int | None = None,
               doc_ids: list[str] | None = None) -> list[Hit]:
        """doc_ids 非空时只在这些文档内检索（A 榜 / B 榜路由后）。"""
        ...
