"""法规 txt 直读解析器。"""
from __future__ import annotations
import os
from .base import Parser
from .structure import build_blocks
from ..schema import Doc


class TxtParser(Parser):
    domain = "regulatory"

    def parse(self, doc_id: str, path: str) -> Doc:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        lines = text.splitlines()
        # 标题取首个非空行或文件名
        title = next((l.strip() for l in lines if l.strip()), doc_id)
        blocks = build_blocks(doc_id, lines)
        return Doc(doc_id=doc_id, domain=self.domain, title=title[:80],
                   source_path=path, blocks=blocks)
