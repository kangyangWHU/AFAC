"""共享文本工具:中文标题/条款号识别 + 归一化。
(原 parser/base.py + chunker/base.py 的纯工具部分,集中于此,供 vl.chunker 与 agentic.decompose 用。)
"""
from __future__ import annotations
import re

# ---- 中文条款号 / 标题层级 ----
ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+条")
SECTION_RE = re.compile(r"^第[一二三四五六七八九十百零〇\d]+[章节编]")
_T = r"[一-龥][^。；：，%\n]{1,21}"                       # CJK 起头、2-22 字、无句中标点
_NUM_MULTI = re.compile(r"^(\d{1,2}(?:\.\d{1,3})+)[\.、．]?\s*(" + _T + r")$")
_NUM_SEP = re.compile(r"^(\d{1,2})[、．.]\s*(" + _T + r")$")
_CN_HEAD = re.compile(r"^[一二三四五六七八九十]{1,3}[、．.]\s*(" + _T + r")$")
_PAREN_HEAD = re.compile(r"^[（(][一二三四五六七八九十]{1,3}[)）]\s*(" + _T + r")$")
_PAREN_NUM = re.compile(r"^[（(]\d{1,2}[)）]\s*(" + _T + r")$")
# 大纲层级按记号类型固定次序排 rank(数值大=更深),保证 1、嵌在（一）下、（一）嵌在 一、下
_RANK_SECTION_TOP, _RANK_SECTION_SUB = 1, 2
_RANK_CN, _RANK_PAREN_CN, _RANK_NUM, _RANK_PAREN_NUM = 3, 4, 5, 6


def heading_level(line: str) -> tuple[int, str] | None:
    """识别标题→(层级rank, 标题文本);非标题→None。rank 仅用于 section_path 栈的相对嵌套。"""
    s = line.strip()
    if SECTION_RE.match(s) and len(s) <= 40:
        return (_RANK_SECTION_TOP if ("章" in s[:6] or "编" in s[:6]) else _RANK_SECTION_SUB, s)
    m = _NUM_MULTI.match(s)
    if m:
        return (1 + m.group(1).count("."), s)
    if _CN_HEAD.match(s):
        return (_RANK_CN, s)
    if _PAREN_HEAD.match(s):
        return (_RANK_PAREN_CN, s)
    if _NUM_SEP.match(s):
        return (_RANK_NUM, s)
    if _PAREN_NUM.match(s):
        return (_RANK_PAREN_NUM, s)
    return None


def detect_article_no(line: str) -> str | None:
    m = ARTICLE_RE.match(line.strip())
    return m.group(0) if m else None


# ---- 归一化(去 PDF 伪空格) ----
_CJK = r"一-鿿　-〿＀-￯"
_HSP = r"[^\S\n]+"                                        # 横向空白(不含换行)
_CJK_SPACE = re.compile(rf"(?<=[{_CJK}]){_HSP}(?=[{_CJK}])")
_NUM_CJK = re.compile(rf"(?<=[0-9]){_HSP}(?=[{_CJK}])|(?<=[{_CJK}]){_HSP}(?=[0-9])")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def normalize(text: str, strip_cjk_spaces: bool = True, collapse_spaces: bool = True) -> str:
    if strip_cjk_spaces:
        text = _CJK_SPACE.sub("", text)
        text = _NUM_CJK.sub("", text)
    if collapse_spaces:
        text = _MULTISPACE.sub(" ", text)
    return text.strip()
