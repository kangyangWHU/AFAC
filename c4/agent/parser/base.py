"""Parser 抽象接口 + 通用工具。换解析器只需实现 parse()。"""
from __future__ import annotations
import re
from abc import ABC, abstractmethod
from ..schema import Doc

# 中文条款号锚点：第一条 / 第八十二条 / 第41条
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+条")
# 章节锚点：第六章 / 第二节
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+[章节编]")
# 标题文本：CJK 起头、≥3 字（滤掉"亿元/万元"等单位后缀）、≤22 字、无句中标点
_T = r"[一-龥][^。；：，%\n]{2,21}"
# 多级编号标题：2.1 / 2.1.1（首段≤2位防"390.7亿元"这类小数值；必须有内部点）
_NUM_MULTI = re.compile(r"^(\d{1,2}(?:\.\d{1,3})+)[\.、．]?\s*(" + _T + r")$")
# 单级编号标题：必须有显式分隔符 1、/ 2．（无分隔的"270 家网点"不算）
_NUM_SEP = re.compile(r"^(\d{1,2})[、．.]\s*(" + _T + r")$")
# 中文序号标题：一、主要会计数据
_CN_HEAD = re.compile(r"^[一二三四五六七八九十]{1,3}[、．.]\s*(" + _T + r")$")
# 括号序号标题：（一）期内累计新签合同额
_PAREN_HEAD = re.compile(r"^[（(][一二三四五六七八九十]{1,3}[)）]\s*(" + _T + r")$")


def heading_level(line: str) -> tuple[int, str] | None:
    """识别标题并返回 (层级, 标题文本)；非标题返回 None。层级用于 section_path 栈嵌套：
    章/编=1 节=2；多级数字按点数(2.1=2,2.1.1=3)；单级 1、=1；一、=1；（一）=2。
    保守：编号须带分隔符或多级，标题须 CJK 起头、≤22字、无句中标点，避免数字行/列表项/正文误判。"""
    s = line.strip()
    if SECTION_RE.match(s) and len(s) <= 40:
        return (1 if ("章" in s[:6] or "编" in s[:6]) else 2, s)
    m = _NUM_MULTI.match(s)
    if m:
        return (1 + m.group(1).count("."), s)
    if _NUM_SEP.match(s) or _CN_HEAD.match(s):
        return (1, s)
    if _PAREN_HEAD.match(s):
        return (2, s)
    return None


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
