"""低价值块识别 + 伪表清洗。治理 PyMuPDF 过度表格检测、目录页噪声。"""
from __future__ import annotations
import re

_ART = re.compile(r"第[一二三四五六七八九十百零〇\d]+条")


def is_toc(text: str) -> bool:
    """目录页：多个'第X条'但每条后内容极短（只有标题）。"""
    arts = _ART.findall(text)
    if len(arts) < 4:
        return False
    # 去掉条号后剩余内容平均长度
    body = _ART.sub("", text).replace("\n", "")
    return len(body) / max(len(arts), 1) < 12


def is_shell_table(text: str) -> bool:
    """空壳表：大量管道符但几乎无实质内容。"""
    pipes = text.count("|")
    body = re.sub(r"[|\-\s]", "", text)
    return pipes > 10 and len(body) < 30


def is_fragment(text: str) -> bool:
    return len(re.sub(r"[|\-\s]", "", text)) < 15


def low_value(text: str) -> bool:
    return is_toc(text) or is_shell_table(text) or is_fragment(text)


def clean_pseudo_table(text: str, min_digits: int = 5) -> tuple[str, bool]:
    """伪表（释义/正文被当表）→ 去管道符转纯文本。
    返回 (清洗后文本, 是否被重分类为text)。真数据表(数字多)保持原样。"""
    if "|" not in text:
        return text, False
    digits = sum(c.isdigit() for c in text)
    if digits >= min_digits:
        return text, False     # 真数据表，保留
    # 像释义/正文：去掉管道与分隔行
    lines = []
    for ln in text.split("\n"):
        if set(ln.strip()) <= set("|-: "):
            continue
        ln = re.sub(r"\s*\|\s*", " ", ln).strip(" |")
        if ln.strip():
            lines.append(ln)
    return "\n".join(lines), True
