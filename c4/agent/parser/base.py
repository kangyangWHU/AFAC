"""Parser 抽象接口 + 通用工具。换解析器只需实现 parse()。"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from ..schema import Doc

# 中文条款号锚点：第一条 / 第八十二条 / 第41条
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+条")
# 章节锚点：第六章 / 第二节
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+[章节编]")


class Parser(ABC):
    domain: str = ""

    @abstractmethod
    def parse(self, doc_id: str, path: str) -> Doc:
        ...


def detect_article_no(line: str) -> str | None:
    m = ARTICLE_RE.match(line.strip())
    return m.group(0) if m else None


def is_section_heading(line: str) -> bool:
    return bool(SECTION_RE.match(line.strip()))


def dedup_repeated_lines(lines: list[str], min_repeat: int = 4) -> list[str]:
    """剔除跨页重复的页眉/页脚/水印行。"""
    from collections import Counter
    norm = [l.strip() for l in lines]
    cnt = Counter(l for l in norm if 0 < len(l) <= 40)
    drop = {l for l, c in cnt.items() if c >= min_repeat}
    return [l for l in lines if l.strip() not in drop]
