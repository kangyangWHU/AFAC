"""文本归一化工具：解析后、入块前清洗。"""
from __future__ import annotations
import re

# 中文字符（含全角标点）之间的伪空格
_CJK = r"一-鿿　-〿＀-￯"
_CJK_SPACE = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
# 数字内部被拆的空格："803, 964" 不动，但 "2025 年" -> "2025年"
_NUM_CJK = re.compile(rf"(?<=[0-9])\s+(?=[{_CJK}])|(?<=[{_CJK}])\s+(?=[0-9])")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def normalize(text: str, strip_cjk_spaces: bool = True,
              collapse_spaces: bool = True) -> str:
    if strip_cjk_spaces:
        text = _CJK_SPACE.sub("", text)
        text = _NUM_CJK.sub("", text)
    if collapse_spaces:
        text = _MULTISPACE.sub(" ", text)
    return text.strip()
