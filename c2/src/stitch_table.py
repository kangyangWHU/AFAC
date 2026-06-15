# -*- coding: utf-8 -*-
"""TABLE 2D 重组（支持多子表）。

关键事实（实测）：API 对**跨子表边界**的 tile，会自己返回**多个 `<table>` + 中间标题**
（如 a300a942 的某 tile 返回 表1尾3行 + "上海人寿…女性" + 表2表头）。之前 `_first_table`
只取第一个 `<table>`、丢掉了后面的表和标题 → 多子表被压成一个、结构崩。

本模块：
  - `parse_tile_segments`：把一个 tile 输出切成 [(标题, 单元格网格), ...]（按 <table> 分段）；
  - `_reconstruct_grid`：单表的 2D 重组（全局列宽 W_c 一致，原逻辑）；
  - `stitch_multi`：扫各 tile 的分段，在"≥2 段"的边界 band 处把网格拆成多个子表，
    每个子表各自重建 + 还原标题，按 "标题\n\n<table>" 拼回（对齐 GT 多 <table> 结构）。
  单表（所有 tile 只 1 段）走原路径，零回归。
"""
import re
from collections import Counter
from teds import _first_table, _parse_grid

_GT_TABLE_OPEN = '<table border="1" cellpadding="8" cellspacing="0">'
_TABLE_RE = re.compile(r"<table.*?</table>", re.S | re.I)


# ---------------------------------------------------------------------------
# tile 输出 → 分段 [(caption, grid), ...]
# ---------------------------------------------------------------------------
def _clean_caption(s):
    """清洗标题/表前文字，**保留换行**：GT 把多行标题(主标题/副标题/单位)拆成
    独立逻辑块，阅读流按块计分。若把换行也压成空格→多行标题合并成 1 块→
    与 GT 的 N 块错位，read-order 被严重低估(实测完美表 read 仅 25)。
    故按行清洗、丢空行、用 \n 连接，让每行各自成块。"""
    s = re.sub(r"```\w*", "", s or "")
    s = re.sub(r"<\s*br\s*/?>", "\n\n", s, flags=re.I)   # 显式换行标签 → 段落断
    s = re.sub(r"<[^>]+>", " ", s)                        # 其余标签去掉
    # 按空行分段(阅读流按段成块)，段内压空白，段间保留空行 → 多行标题=多个块
    paras = [" ".join(p.split()).strip() for p in re.split(r"\n\s*\n", s)]
    return "\n\n".join(p for p in paras if p)


def parse_tile_segments(html):
    """把一个 tile 输出按 <table> 切成 [(caption_before, cell_grid), ...]。
    caption_before = 该表前的文字（去标签）。无表则返回 []。

    **鲁棒于被截断的表**：tile 输出若因 ~12k 字符上限被截断、缺少 `</table>`，
    旧逻辑会整块丢弃。这里对未闭合的表补上 `</table>` 再解析，**保住已读内容**。
    """
    if not html:
        return []
    starts = [m.start() for m in re.finditer(r"<table", html, re.I)]
    if not starts:
        return []
    captions = [_clean_caption(html[:starts[0]])]
    segs = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(html)
        chunk = html[s:end]
        ci = chunk.lower().rfind("</table>")
        if ci == -1:                              # 截断未闭合 → 补 </table>
            tbl, after = chunk + "</table>", ""
        else:
            tbl, after = chunk[:ci + 8], _clean_caption(chunk[ci + 8:])
        el = _first_table(tbl)
        grid = _parse_grid(el) if el is not None else []
        segs.append((captions[i], grid))
        captions.append(after)
    return segs


def parse_tile(html):
    """兼容旧接口：只取第一个表的网格。"""
    segs = parse_tile_segments(html)
    return segs[0][1] if segs else []


def _mode_width(grid):
    if not grid:
        return 0
    return Counter(len(r) for r in grid).most_common(1)[0][0]


def _mode(vals):
    return Counter(vals).most_common(1)[0][0] if vals else 0


# ---------------------------------------------------------------------------
# 单表 2D 重组（全局列宽一致 W_c）—— 原 stitch_table 逻辑，重构成可复用
# ---------------------------------------------------------------------------
def _reconstruct_grid(parsed, col_cells, col_cuts=None, deg_hi=1.8, deg_lo=0.4):
    """parsed[r][c] = 单元格网格 或 None。返回 rows（list[list[str]]）。

    空列块（API 无读数）的列宽不再用过检测的 col_cells（会膨胀），而是按
    **已读列块的列密度（列/像素）× 该空块像素宽** 估计，避免列爆炸。
    """
    n_band = len(parsed)
    n_col = max((len(row) for row in parsed), default=0)

    # 第一遍：每个 tile-column 的规范列宽 W_c
    col_widths = [[] for _ in range(n_col)]
    for r in range(n_band):
        counts = [len(g) for g in parsed[r] if g]
        if not counts:
            continue
        med = sorted(counts)[len(counts) // 2]
        for c, g in enumerate(parsed[r]):
            if g and not (len(g) > deg_hi * med or len(g) < deg_lo * med):
                col_widths[c].append(_mode_width(g))

    # 列密度（列/像素）：用有读数的列块估计，给空块兜底用
    def pix_w(c):
        if col_cuts and c + 1 < len(col_cuts):
            return max(1, col_cuts[c + 1] - col_cuts[c])
        return 0
    read_cols = sum(_mode(col_widths[c]) for c in range(n_col) if col_widths[c])
    read_pix = sum(pix_w(c) for c in range(n_col) if col_widths[c])
    density = (read_cols / read_pix) if read_pix > 0 else 0.0

    W = []
    for c in range(n_col):
        if col_widths[c]:
            W.append(max(1, _mode(col_widths[c])))
        else:
            cc = col_cells[c] if c < len(col_cells) else 1
            if density > 0 and pix_w(c) > 0:
                cc = min(cc, round(density * pix_w(c)))   # 用密度估封顶 col_cells，防爆炸
            W.append(max(1, cc))

    # 第二遍：按 W_c 重组。band 行数取中位数（用 max 恢复截断行实测净中性——恢复的行
    # 多为空格、内容已随截断丢失，且 max 会在别处过补；故仍用中位数 + 退化块剔除）。
    all_rows = []
    for r in range(n_band):
        grids = parsed[r]
        counts = [len(g) for g in grids if g]
        if not counts:
            continue
        med = sorted(counts)[len(counts) // 2]
        if med == 0:
            continue
        content = [bool(g) and not (len(g) > deg_hi * med or len(g) < deg_lo * med)
                   for g in grids]
        if not any(content):
            continue
        band = [[] for _ in range(med)]
        for c in range(n_col):
            g = grids[c]
            wc = W[c]
            if not content[c]:
                for i in range(med):
                    band[i].extend([""] * wc)
                continue
            for i in range(med):
                cells = list(g[i]) if i < len(g) else []
                cells = (cells + [""] * wc)[:wc]
                band[i].extend(cells)
        all_rows.extend(band)
    return all_rows


_CJK = re.compile(r"[一-鿿]")
_DIGIT = re.compile(r"\d")


def _is_caption_like(text):
    """leading-text 是否像子表标题：CJK≥4 且中文占比高于数字（排除"106 21630.74"这类数据碎片）。"""
    t = (text or "").strip()
    if len(t) < 4:
        return False
    cjk = len(_CJK.findall(t))
    dig = len(_DIGIT.findall(t))
    return cjk >= 4 and cjk > dig


def _is_caption_row(cells):
    """纯文字行（子表标题/重复表头）：CJK≥3、无数字、非空格≤4。数据行必有数字，故不会误判。"""
    txt = "".join(cells)
    if len(_DIGIT.findall(txt)) > 0:
        return False
    if len(_CJK.findall(txt)) < 3:
        return False
    return sum(1 for c in cells if c and c.strip()) <= 4


def _split_at_headers(rows, min_seg=3):
    """在"纯文字表头/标题行"处把行序列切成多个子表段（标题行归入其下方子表）。
    仅当能切出 ≥2 段、且每段 ≥min_seg 行时才生效，避免误切。
    """
    segs = []
    cur = []
    for row in rows:
        if _is_caption_row(row) and len(cur) >= min_seg:
            segs.append(cur)
            cur = [row]
        else:
            cur.append(row)
    if cur:
        segs.append(cur)
    # 末段太短则并回上一段（避免碎尾）
    if len(segs) >= 2 and len(segs[-1]) < min_seg:
        segs[-2].extend(segs[-1])
        segs.pop()
    return segs


def _trim_trailing_empty_cols(rows):
    """删除"所有行都为空"的尾部列（W_c 过补出来的幻影列）。
    安全：三角表尾列在上部行有数据→非全空→保留；只裁真正全空的尾列。
    """
    if not rows:
        return rows
    maxlen = max(len(r) for r in rows)
    keep = maxlen
    for c in range(maxlen - 1, -1, -1):
        if all(len(r) <= c or not (r[c] or "").strip() for r in rows):
            keep = c
        else:
            break
    if keep == maxlen:
        return rows
    return [r[:keep] for r in rows]


def rows_to_html(rows):
    rows = _trim_trailing_empty_cols(rows)
    out = [_GT_TABLE_OPEN]
    for row in rows:
        tds = "".join("<td>%s</td>" % (c if c is not None else "") for c in row)
        out.append("      <tr>%s</tr>" % tds)
    out.append("</table>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 多子表重组主入口
# ---------------------------------------------------------------------------
def stitch_multi(tile_outputs, meta):
    """按 API 自带的多 <table> 边界拆子表、各自重建、拼回。单表走单表路径。"""
    n_band = len(tile_outputs)
    n_col = max((len(row) for row in tile_outputs), default=0)
    col_cells = meta.get("col_cells", [])
    col_cuts = meta.get("col_cuts")
    row_cuts = meta.get("row_cuts")
    split_bands = meta.get("split_bands", set())
    blank = meta.get("blank", [[False] * n_col for _ in range(n_band)])

    # 每个 tile → 分段，并**丢弃展平幻觉段**（行数 > 像素高/最小行高，不可能真实）
    MIN_ROW_PX = 8
    MIN_COL_PX = 6
    def _band_max_rows(r):
        h = (row_cuts[r + 1] - row_cuts[r]) if (row_cuts and r + 1 < len(row_cuts)) else 1500
        return max(4, h // MIN_ROW_PX)

    def _tile_max_cols(c):
        w = (col_cuts[c + 1] - col_cuts[c]) if (col_cuts and c + 1 < len(col_cuts)) else 1500
        return max(4, w // MIN_COL_PX)

    def _clean_segs(r, c):
        if blank[r][c] or tile_outputs[r][c] is None:
            return []
        rlim, clim = _band_max_rows(r), _tile_max_cols(c)
        # 丢弃展平幻觉：超高（行数>像素高/8）或超宽（列数>像素宽/6）的段
        return [(cap, g) for (cap, g) in parse_tile_segments(tile_outputs[r][c])
                if len(g) <= rlim and _mode_width(g) <= clim]

    segs = [[_clean_segs(r, c) for c in range(len(tile_outputs[r]))]
            for r in range(n_band)]

    # 子表 = 在"某 band 的 tile 出现 ≥2 段"处拆分。收集每个子表的 (caption, parsed_bands)
    subtables = []                                  # [(caption, parsed[][])]
    cur_bands = []                                  # 当前子表的 band 列表，每元素是 [grid/None]*n_col
    cur_caption = ""
    first_caption = ""

    for r in range(n_band):
        max_seg = max((len(segs[r][c]) for c in range(len(segs[r]))), default=0)
        if max_seg == 0:
            continue
        # 取该 band 第 j 段（j 从 0..max_seg-1）：每个 tile 的第 j 段网格
        for j in range(max_seg):
            band_row = []
            seg_cap = ""
            for c in range(n_col):
                sc = segs[r][c] if c < len(segs[r]) else []
                if j < len(sc):
                    cap, grid = sc[j]
                    band_row.append(grid)
                    if cap and not seg_cap and _is_caption_like(cap):
                        seg_cap = cap               # 该段前的"标题样"文字
                else:
                    band_row.append(None)
            # 边界判定：① 同 tile 出现新表段(j>0)；② 段前有标题样文字
            # (竖线缝 split_bands 实测净负，已弃用——会过切，offset 多子表收益)
            has_data = any(any(g for g in br) for br in cur_bands)
            boundary = (j > 0) or (seg_cap and has_data)
            if boundary and cur_bands:
                subtables.append((cur_caption, cur_bands))
                cur_caption = seg_cap
                cur_bands = [band_row]
            else:
                if seg_cap and not first_caption and not cur_bands:
                    first_caption = seg_cap         # 文档顶标题
                cur_bands.append(band_row)
    if cur_bands:
        subtables.append((cur_caption, cur_bands))

    # 单表：API 未自带拆分。再用"纯文字表头行"做一次泛化拆分（多子表被合并的兜底）
    if len(subtables) <= 1:
        rows = _reconstruct_grid(cur_bands if subtables else [], col_cells, col_cuts)
        segs = _split_at_headers(rows)
        if len(segs) >= 2:
            html = "\n\n".join(rows_to_html(s) for s in segs)
        else:
            html = rows_to_html(rows)
        if first_caption:
            html = first_caption + "\n\n" + html
        return html

    # 多子表：逐个重建 + 拼标题。每个 API 子表再用"纯文字表头行"递归拆（重复表头=内部边界）
    parts = []
    for k, (cap, bands) in enumerate(subtables):
        rows = _reconstruct_grid(bands, col_cells, col_cuts)
        if not rows:
            continue
        segs = _split_at_headers(rows)
        html = ("\n\n".join(rows_to_html(s) for s in segs)
                if len(segs) >= 2 else rows_to_html(rows))
        caption = (first_caption if k == 0 else cap)
        parts.append((caption + "\n\n" + html) if caption else html)
    return "\n\n".join(parts)


# 兼容旧调用名
def stitch_table(tile_outputs, meta, **kw):
    return stitch_multi(tile_outputs, meta)
