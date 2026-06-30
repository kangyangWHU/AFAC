# -*- coding: utf-8 -*-
"""LONG 拼接：把各横条的 Markdown 输出合并为一篇，处理接缝重复。

slicer_long 在墨量最小处下刀，长表格中部没有空白带 → **切点会落在表格内部**。
于是上条把上半截表收口成 `...</table>`、下条把下半截重新开 `<table>...`。

旧逻辑用**字符级模糊行去重**兜底：`</table>`(8字) 与 `<table>`(7字) 相似度
=1−1/8=0.875 ≥ 0.85 阈值 → 误把下条的 `<table>` 开标签删掉 → 行块悬空、多出一个
`</table>` → 整篇 HTML 标签不配平 = malformed；即便配平，被切开的同一张表也变成
两张独立 `<table>`、接缝行还重复 → TEDS 结构错。

本版**表感知接缝**：当上条以 `</table>` 收口、下条紧接着以裸 `<table>` 起头
(中间无标题文字，说明切点穿过了同一张表)，把两段拼成一张表——去掉内侧的
`</table>`/`<table>` 边界标签，按 `<tr>` **行级**去重重叠行后续接。非表格接缝仍走
模糊行去重，但加固使其永不删除结构标签行。
"""
import re
from rapidfuzz.distance import Levenshtein

_TABLE_CLOSE = re.compile(r"</table>\s*$", re.I)
_TABLE_OPEN_FULL = re.compile(r"^\s*<table[^>]*>\s*$", re.I)
_TABLE_OPEN_LEAD = re.compile(r"^\s*<table[^>]*>", re.I)
_STRUCT_LINE = re.compile(r"^\s*</?(table|tr|thead|tbody)\b", re.I)


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


def _is_split_table_seam(acc, lines):
    """上条以 </table> 收口、下条紧接以裸 <table> 起头 ⇒ 切点穿过同一张表。
    下条首个非空内容必须是 <table> 开标签本身(无前置标题文字),否则视为两张不同表。"""
    return bool(_TABLE_CLOSE.search(acc[-1])) and \
        bool(_TABLE_OPEN_LEAD.match(lines[0]))


def _splice_table(acc, lines, max_overlap, sim_thresh):
    """把被切开的两段表拼成一张：去内侧边界标签 + 行级去重。

    判据只用结构信号(上段裸 </table> 收口、下段裸 <table> 起头、中间无标题文字)——
    相邻两张真表之间必有标题行，下段就不会以裸 <table> 起头，故无需再用列数判据
    (rowspan 续行列数本就不齐，列数判据反会误拒)。"""
    a = acc[:]
    # 去掉上段末尾的 </table>（可能独占一行，也可能附在行尾）
    a[-1] = _TABLE_CLOSE.sub("", a[-1]).rstrip()
    if not a[-1].strip():
        a.pop()
    # 去掉下段开头的 <table>（可能独占一行，也可能与首行内容同行）
    b = lines[:]
    if _TABLE_OPEN_FULL.match(b[0]):
        b.pop(0)
    else:
        b[0] = _TABLE_OPEN_LEAD.sub("", b[0], count=1)
        if not b[0].strip():
            b.pop(0)
    if not a or not b:
        return acc + lines
    k = _seam_overlap(a, b, max_overlap, sim_thresh)
    return a + b[k:]


def _guarded_overlap(acc, lines, max_k, sim_thresh):
    """非表格接缝的模糊去重，但绝不删除结构标签行(<table>/<tr>…)。"""
    k = _seam_overlap(acc, lines, max_k, sim_thresh)
    while k > 0 and _STRUCT_LINE.match(lines[k - 1]):
        k -= 1
    return k


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
        if _is_split_table_seam(acc, lines):
            acc = _splice_table(acc, lines, max_overlap_lines, sim_thresh)
        else:
            k = _guarded_overlap(acc, lines, max_overlap_lines, sim_thresh)
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
