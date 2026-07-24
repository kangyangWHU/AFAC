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
_TABLE_CLOSE_BODY = re.compile(r"</tbody>\s*</table>\s*$", re.I)
_TABLE_OPEN_FULL = re.compile(r"^\s*<table[^>]*>\s*$", re.I)
_TABLE_OPEN_LEAD = re.compile(r"^\s*<table[^>]*>", re.I)
_TBODY_OPEN_LEAD = re.compile(r"^\s*<tbody[^>]*>", re.I)
_STRUCT_LINE = re.compile(r"^\s*</?(table|tr|thead|tbody)\b", re.I)
_CELL_OPEN = re.compile(r"<t[dh]\b[^>]*>", re.I)
_FIRST_ROW = re.compile(r"<tr\b[^>]*>.*?</tr>", re.I | re.S)

# 续行判据:下段首行是不是"新块"(标题/列表/编号/表格/引用)
_BLOCK_START = re.compile(
    r"^\s*(#|＃|[-*+]\s|>|\||<|第[一二三四五六七八九十百]|[（(]|"
    r"[一二三四五六七八九十]+\s*[、.]|\d+\s*[).、]|[①②③④⑤⑥⑦⑧⑨⑩])")
_TERMINAL = "。.；;！!？?：:」』）)】》\"'"    # 末行以此结尾=写完了,不续


def _is_line_continuation(a, b):
    """接缝处 a(上段末行) 与 b(下段首行) 是否是被切断的同一行 → 应直接接上。"""
    a, b = a.rstrip(), b.strip()
    if not a or not b:
        return False
    if a.lstrip().startswith(("#", "＃")):        # 标题不与正文续
        return False
    if _STRUCT_LINE.match(a) or _STRUCT_LINE.match(b):
        return False
    if a[-1] in _TERMINAL:                         # 末行已收尾
        return False
    if _BLOCK_START.match(b):                      # 下段首行是新块
        return False
    return True


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


def _split_table_seam_loose(acc, lines, max_orphan=2):
    """上条以 </table> 收口；下条在【前 max_orphan 个非空行内】重新 <table> 起头,
    中间只夹极少量非标题短文字 ⇒ 切点穿过同一张表、且把被切断的表头/单元格读成了散行。

    返回下条 <table> 起头的行号(>0 表命中)；否则 -1。
    与 _is_split_table_seam(下条必须裸 <table> 起头)互补:那个要求中间无任何文字,
    这个容忍极少量"孤儿散行"(被切断的表头),但遇到标题/结构行/过多散行即判为两张表
    (避免把 over-tabularize 的正文误并:正文块散行多、且不会紧接 <table>)。"""
    if not _TABLE_CLOSE.search(acc[-1]):
        return -1
    cnt = 0
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        if _TABLE_OPEN_LEAD.match(ln):
            return i if cnt else -1        # cnt==0 时归 _is_split_table_seam 处理
        if _STRUCT_LINE.match(ln) or ln.lstrip().startswith(("#", "＃")):
            return -1                      # 标题/结构 ⇒ 新表,不续
        cnt += 1
        if cnt > max_orphan:
            return -1
    return -1


def _splice_table_loose(acc, lines, idx, max_overlap, sim_thresh):
    """续表拼接:去内侧边界标签,idx>0 时把 <table> 前的孤儿散行包成 <tr><td> 行插回
    两段表之间,再行级去重(idx=0 即紧邻续表,无孤儿行)。

    判据只用结构信号(上段裸 </table> 收口、下段裸 <table> 起头、中间无标题文字)——
    相邻两张真表之间必有标题行,下段就不会以裸 <table> 起头,故无需再用列数判据
    (rowspan 续行列数本就不齐,列数判据反会误拒)。"""
    a = acc[:]
    a[-1] = _TABLE_CLOSE.sub("", a[-1]).rstrip()
    if not a[-1].strip():
        a.pop()
    orphan_rows = ["<tr><td>%s</td></tr>" % ln.strip()
                   for ln in lines[:idx] if ln.strip()]
    b = lines[idx:]
    if _TABLE_OPEN_FULL.match(b[0]):
        b.pop(0)
    else:
        b[0] = _TABLE_OPEN_LEAD.sub("", b[0], count=1)
        if not b[0].strip():
            b.pop(0)
    if not a or not b:
        return acc + lines
    k = _seam_overlap(a, b, max_overlap, sim_thresh)
    return a + orphan_rows + b[k:]


def _trailing_orphan_table_seam(acc, lines, max_orphan=2):
    """识别「上条已闭表 + 少量被读到表外的单元格前缀 + 下条续表」。

    典型情形：切点穿过计划十三的长单元格，上条在 </table> 后多读出
    「基本部分、可选部分的疾病关爱保险金、」，下条则从同一行的后半格开始。
    只在两侧列数相同、孤儿文本很短且以连接标点结尾时命中，避免误合
    「上表 + 标题 + 下表」。返回 (上表结束行下标, 孤儿行列表)。
    """
    if not acc or not lines or not _TABLE_OPEN_LEAD.match(lines[0]):
        return None

    orphan = []
    i = len(acc) - 1
    while i >= 0 and len(orphan) < max_orphan and not _TABLE_CLOSE.search(acc[i]):
        s = acc[i].strip()
        if not s:
            i -= 1
            continue
        if (_STRUCT_LINE.match(s) or s.startswith(("#", "＃")) or len(s) > 100):
            return None
        orphan.insert(0, s)
        i -= 1
    if i < 0 or not orphan or not _TABLE_CLOSE.search(acc[i]):
        return None
    joined = "".join(orphan)
    if not joined.endswith(("、", "，", ",", "：", ":", "；", ";")):
        return None

    prev_rows = list(_FIRST_ROW.finditer(acc[i]))
    prev_row = prev_rows[-1] if prev_rows else None
    next_row = _FIRST_ROW.search(lines[0])
    if not prev_row or not next_row:
        return None
    prev_cols = len(_CELL_OPEN.findall(prev_row.group(0)))
    next_cols = len(_CELL_OPEN.findall(next_row.group(0)))
    if prev_cols < 2 or prev_cols != next_cols:
        return None
    return i, orphan


def _prepend_to_last_cell_of_first_row(s, prefix):
    """把被读到表外的前缀塞回续表首行的最后一个单元格。"""
    row = _FIRST_ROW.search(s)
    if not row:
        return s
    opens = list(_CELL_OPEN.finditer(row.group(0)))
    if not opens:
        return s
    pos = row.start() + opens[-1].end()
    return s[:pos] + prefix + s[pos:]


def _splice_trailing_orphan_table(acc, lines, seam):
    """合并尾部孤儿续表，并恢复被切断的首行末单元格。"""
    close_i, orphan = seam
    a = acc[:close_i + 1]
    b = lines[:]

    # 去掉两段内侧的 table 边界。若上段使用 tbody，把续表的 tr 也放进
    # 同一 tbody，避免依赖 HTML 容错规则归并节点。
    had_tbody = bool(_TABLE_CLOSE_BODY.search(a[-1]))
    if had_tbody:
        a[-1] = _TABLE_CLOSE_BODY.sub("", a[-1]).rstrip()
    else:
        a[-1] = _TABLE_CLOSE.sub("", a[-1]).rstrip()
    b[0] = _TABLE_OPEN_LEAD.sub("", b[0], count=1)
    if had_tbody:
        b[0] = _TBODY_OPEN_LEAD.sub("", b[0], count=1)
        for i, line in enumerate(b):
            if "</table>" in line.lower():
                if "</tbody>" not in line.lower():
                    b[i] = re.sub(r"</table>", "</tbody></table>", line,
                                  count=1, flags=re.I)
                break
    b[0] = _prepend_to_last_cell_of_first_row(b[0], "".join(orphan))
    return a + b


def _guarded_overlap(acc, lines, max_k, sim_thresh):
    """非表格条带不重叠，禁止模糊去重误删相似正文。

    slicer_long 以相邻、不重叠的坐标裁条；正文接缝只需续行，不应删除内容。
    旧的多行平均相似度会被空行和保险条款套话抬高，曾一次删掉完整的“脑恶性
    肿瘤”段。表内 OCR 自带的重叠仍由 _splice_table_loose 单独处理。
    """
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
        trailing_seam = _trailing_orphan_table_seam(acc, lines)
        if trailing_seam is not None:
            acc = _splice_trailing_orphan_table(acc, lines, trailing_seam)
        elif _is_split_table_seam(acc, lines):
            acc = _splice_table_loose(acc, lines, 0, max_overlap_lines, sim_thresh)
        elif (idx := _split_table_seam_loose(acc, lines)) > 0:
            acc = _splice_table_loose(acc, lines, idx, max_overlap_lines, sim_thresh)
        else:
            k = _guarded_overlap(acc, lines, max_overlap_lines, sim_thresh)
            rest = lines[k:]
            if acc and rest and _is_line_continuation(acc[-1], rest[0]):
                acc[-1] = acc[-1].rstrip() + rest[0].strip()   # 被切断的行接回一行
                acc.extend(rest[1:])
            else:
                acc.extend(rest)
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
