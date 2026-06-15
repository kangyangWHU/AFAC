# -*- coding: utf-8 -*-
"""LONG 拼接：把各横条的 Markdown 输出合并为一篇，处理接缝重复。

由于 slicer_long 在行间空白带下刀，接缝通常已经干净；这里再做一层
**行级模糊去重**兜底（API 偶尔在边界重复读一行），并清理每条首尾空行。
"""
from rapidfuzz.distance import Levenshtein


def _trim_blanks(lines):
    i, j = 0, len(lines)
    while i < j and not lines[i].strip():
        i += 1
    while j > i and not lines[j - 1].strip():
        j -= 1
    return lines[i:j]


def _line_sim(a, b):
    """两行相似度 ∈[0,1]（归一化编辑距离的补）。"""
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / m


def _seam_overlap(acc, lines, max_k, sim_thresh):
    """找接缝重叠行数 k：acc 末 k 行 ≈ lines 前 k 行。返回最大可信 k。"""
    kmax = min(max_k, len(acc), len(lines))
    for k in range(kmax, 0, -1):
        a, b = acc[-k:], lines[:k]
        sims = [_line_sim(x, y) for x, y in zip(a, b)]
        if sims and sum(sims) / len(sims) >= sim_thresh:
            return k
    return 0


def merge_strips(outputs, max_overlap_lines=8, sim_thresh=0.85):
    """顺序合并条输出，去接缝重复。返回整篇 Markdown。"""
    acc = []
    for out in outputs:
        lines = _trim_blanks((out or "").split("\n"))
        if not lines:
            continue
        if not acc:
            acc = lines
            continue
        k = _seam_overlap(acc, lines, max_overlap_lines, sim_thresh)
        acc.extend(lines[k:])
    # 规整：连续空行压成一个
    out_lines = []
    blank = False
    for ln in acc:
        if ln.strip():
            out_lines.append(ln)
            blank = False
        else:
            if not blank:
                out_lines.append("")
            blank = True
    return "\n".join(out_lines).strip()
