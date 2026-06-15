"""统一数据模型：所有 parser 产出 Doc，下游索引/检索消费 Block。"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Table:
    caption: str = ""
    markdown: str = ""
    records: list[dict] = field(default_factory=list)


@dataclass
class Block:
    block_id: str
    type: str                       # heading | text | article | table | list
    text: str
    section_path: list[str] = field(default_factory=list)
    article_no: Optional[str] = None   # 第X条 / 报表科目锚点
    page: Optional[int] = None
    table: Optional[Table] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.table is None:
            d.pop("table", None)
        return d


@dataclass
class Doc:
    doc_id: str
    domain: str
    title: str
    source_path: str
    blocks: list[Block] = field(default_factory=list)

    def char_len(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "domain": self.domain,
            "title": self.title,
            "source_path": self.source_path,
            "blocks": [b.to_dict() for b in self.blocks],
        }
