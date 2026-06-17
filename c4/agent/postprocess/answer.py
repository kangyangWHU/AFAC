"""答案后处理：从模型输出抽取合法字母，按题型规范化。
mcq/tf: 取首个合法字母；multi: 去重+字母序。"""
from __future__ import annotations
import re

_LETTERS = "ABCD"


def extract_letters(text: str) -> list[str]:
    """只从显式结论行 '答案：ABC' 抽字母。找不到则返回 []（交给上层重问）。
    绝不"全文扫字母"——多选题里那会把讨论到的所有选项都当成答案(过度选择)。
    冒号后允许包裹符: 模型常照抄模板写成 '答案：<ABD>' 或加粗 '答案：**ABD**'，
    必须跳过 < * ( （ 【 # 「 等前缀再抓字母, 否则正则在 '<' 处失配 -> 整条答案丢失。"""
    m = re.search(r"(?:答案|最终答案|正确答案|answer)\s*[:：]?\s*[<*（(【「#\s]*([A-D][A-D，,、\s]*)",
                  text, re.I)
    if not m:
        return []
    seen = []
    for ch in m.group(1).upper():
        if ch in _LETTERS and ch not in seen:
            seen.append(ch)
    return seen


def parse_option_verdicts(text: str, answer_format: str) -> str:
    """无"答案："行时的兜底：解析逐项判定 '选项X…→真/假' 拼出答案。
    模型常因 max_tokens 截断在写"答案："行前，但前面已逐项判过真假。"""
    verdicts: dict[str, bool] = {}
    verdict_re = re.compile(r"(不正确|不成立|不符合|错误|假|正确|成立|符合|真)")
    # 逐行取最终判定，优先看箭头/冒号后的结论，避免把“不符合”里的“符合”误判为真。
    for m in re.finditer(r"选项\s*([A-D])[:：.、\s]*(.*?)(?=\n\s*选项\s*[A-D]|$)",
                         text, re.S):
        opt, body = m.group(1).upper(), m.group(2)
        tail = body.rsplit("→", 1)[-1].splitlines()[-1] if "→" in body else body
        hits = list(verdict_re.finditer(tail))
        if not hits:
            hits = list(verdict_re.finditer(body[-300:]))
        if not hits:
            continue
        v = hits[-1].group(1)
        good = v in ("真", "正确", "成立", "符合")
        verdicts.setdefault(opt, good)  # 取该选项首个判定
    if not verdicts:
        return ""
    true_opts = sorted(o for o, g in verdicts.items() if g)
    if answer_format == "multi":
        return "".join(true_opts)
    # mcq/tf: 取判真的第一个
    return true_opts[0] if true_opts else ""


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
