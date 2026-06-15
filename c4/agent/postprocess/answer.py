"""答案后处理：从模型输出抽取合法字母，按题型规范化。
mcq/tf: 取首个合法字母；multi: 去重+字母序。"""
from __future__ import annotations
import re

_LETTERS = "ABCD"


def extract_letters(text: str) -> list[str]:
    """只从显式结论行 '答案：ABC' 抽字母。找不到则返回 []（交给上层重问）。
    绝不"全文扫字母"——多选题里那会把讨论到的所有选项都当成答案(过度选择)。"""
    m = re.search(r"(?:答案|最终答案|正确答案|answer)\s*[:：]?\s*([A-D][A-D，,、\s]*)",
                  text, re.I)
    if not m:
        return []
    seen = []
    for ch in m.group(1).upper():
        if ch in _LETTERS and ch not in seen:
            seen.append(ch)
    return seen


def normalize_answer(text: str, answer_format: str) -> str:
    letters = extract_letters(text)
    if not letters:
        return ""
    if answer_format in ("mcq", "tf"):
        return letters[0]
    # multi: 去重 + 字母序
    return "".join(sorted(set(letters)))


def is_valid(ans: str, answer_format: str) -> bool:
    if not ans or any(c not in _LETTERS for c in ans):
        return False
    if answer_format in ("mcq", "tf"):
        return len(ans) == 1
    return ans == "".join(sorted(set(ans)))  # multi 必须有序去重
