# -*- coding: utf-8 -*-
"""TABLE 2D 重组（单表）。

职责：
  - `parse_tile_segments`：把一个 tile 输出切成 [(标题, 单元格网格), ...]（按 <table> 分段）；
  - `_reconstruct_grid`：单表的 2D 重组（全局列宽 W_c 一致）；
  - `stitch_single_table`：全部 band 平铺 → 剥 ragged 顶 → 重建 → HTML。

**不做子表检测**：多子表在几何层(crop 三段式)已切好，落到这里的单元(全宽模式 /
列错位回退)都是单张表(训练集 gt_tables=1 实证)。旧 stitch_multi 的 caption/表头行
边界检测在单表上只会误拆(单表被误拆成多张)，已随多子表前置切分退役。
"""
import re
from collections import Counter
from metrics.teds import first_table, parse_grid

_GT_TABLE_OPEN = '<table border="1" cellpadding="8" cellspacing="0">'


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
        # 纯文本兜底：稀疏表底部"只有一列数据(行号)、其余全空"的 tile 会被 API 读成纯文本
        # (如 '70\n71\n…\n106')而非 <table>。这些是真内容(年度号),不能丢。多行→当单列 grid
        # (每行一个 cell)，由下游按骨架列数补空 td；单行→当表外文字(caption)。
        lines = [ln.strip() for ln in html.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return [("", [[ln] for ln in lines])]
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
        el = first_table(tbl)
        grid = parse_grid(el) if el is not None else []
        segs.append((captions[i], grid))
        captions.append(after)
    return segs


def parse_tile(html):
    """tile 解析入口(两条流水线共用)：只取第一个表的网格。"""
    segs = parse_tile_segments(html)
    return segs[0][1] if segs else []


def _mode_width(grid):
    if not grid:
        return 0
    return Counter(len(r) for r in grid).most_common(1)[0][0]


def _mode(vals):
    return Counter(vals).most_common(1)[0][0] if vals else 0


# ---------------------------------------------------------------------------
# 单表 2D 重组（全局列宽一致 W_c）
# ---------------------------------------------------------------------------
def _width_segments(parsed, n_col):
    """检测「不同宽度子表上下堆叠」：某 tile-column 的众数列宽在某 band 突变
    （如某 tile-column 上下两半列数不同），说明两个不同宽度的子表叠在
    一起。返回按该突变 band 切的 band 分段 [(b0,b1),...]。

    必须分段重组的原因：W_c 取全 band 众数时，会被占多数 band 的子表主导
    （众数取窄的一半 → 宽子表被压窄 → 列数与 TEDS 双降）。
    分段后每段各取自己的 W_c。单一宽度（绝大多数表）→ 返回单段，行为完全不变、零回归。
    """
    n_band = len(parsed)
    bw = [[(_mode_width(g) if g else 0)
           for g in (parsed[r] + [None] * n_col)[:n_col]] for r in range(n_band)]
    cuts = []
    for c in range(n_col):
        # 排除 ≤2 列的 band:纯文本兜底/近空 tile 产生的"1列"是 OCR 没读出数据的**假**宽度
        # 突变(非窄子表),不排除会被误判成"不同宽度子表"、分段成 1 列不补齐(TEDS 大跌)。
        # 真子表最少 2-3 列,排除 ≤2 安全。
        seq = [(r, bw[r][c]) for r in range(n_band) if bw[r][c] > 2]
        if len(seq) < 6:                            # 太短不足以判双峰
            continue
        ws = [w for _, w in seq]
        if max(ws) / max(1, min(ws)) < 1.6:         # 非双峰 → 该列无 regime 变化
            continue
        mid = (max(ws) + min(ws)) / 2
        hi = [w >= mid for w in ws]                 # 高/低簇标记
        for i in range(2, len(seq) - 1):            # 找一处**干净翻转**（两侧各≥2 band 一致）
            if (hi[i] != hi[i - 1]
                    and hi[i - 2] == hi[i - 1] and hi[i + 1] == hi[i]):
                cuts.append(seq[i][0])              # 排除单点尾噪(如近空 tile 的 width=1)
                break
    cuts = [b for b in cuts if 1 < b < n_band - 1]   # 边界 band 不算
    if not cuts:
        return [(0, n_band)]
    b = Counter(cuts).most_common(1)[0][0]          # 多列共同突变的 band = 子表边界
    return [(0, b), (b, n_band)]


_DEG_HI, _DEG_LO = 1.8, 0.4   # 离群 tile 列宽剔除带(>1.8×/<0.4×中位=坏读,不进 W_c 投票)


def _reconstruct_grid(parsed, col_cells, col_cuts=None, framed=False):
    """parsed[r][c] = 单元格网格 或 None。返回 rows（list[list[str]]）。

    空列块（API 无读数）的列宽不再用过检测的 col_cells（会膨胀），而是按
    **已读列块的列密度（列/像素）× 该空块像素宽** 估计，避免列爆炸。
    不同宽度子表上下叠时（`_width_segments` 检出），按子表分段、各取自己的 W_c。
    """
    n_col = max((len(row) for row in parsed), default=0)
    segs = _width_segments(parsed, n_col)
    if len(segs) > 1:                               # 多宽度子表 → 逐段递归(各自 W_c)
        out = []
        for b0, b1 in segs:
            out.extend(_reconstruct_grid(parsed[b0:b1], col_cells, col_cuts, framed))
        return out
    n_band = len(parsed)

    # 第一遍：每个 tile-column 的规范列宽 W_c
    col_widths = [[] for _ in range(n_col)]
    for r in range(n_band):
        counts = [len(g) for g in parsed[r] if g]
        if not counts:
            continue
        med = sorted(counts)[len(counts) // 2]
        for c, g in enumerate(parsed[r]):
            if g and not (len(g) > _DEG_HI * med or len(g) < _DEG_LO * med):
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
            w = max(1, _mode(col_widths[c]))
            if framed and c < len(col_cells) and w < 0.7 * col_cells[c]:
                # 有框且 OCR 读出明显少于框线列数(漏列):把漏读的空列补到框线列数
                # (OCR 读出列数远少于框线列数 —— GT 把空列都标了 td)。
                # 阈值 0.7:OCR 读出接近框线时不补,免过补(读出≈框线时不补)。
                w = col_cells[c]
            W.append(w)
        else:
            cc = col_cells[c] if c < len(col_cells) else 1
            if not framed and density > 0 and pix_w(c) > 0:
                cc = min(cc, round(density * pix_w(c)))   # 无框才用密度封顶防列爆炸;有框直接用框线列数
            W.append(max(1, cc))

    # 第二遍：按 W_c 重组。band 行数取 max(各列行数)=最密集列行数:稀疏列(三角数据列)折叠
    # 空格行读得少,密集列(行标签列)读满=真实行数。不能用中位(稀疏列占多数时中位被拖低、
    # 密集列反被当离群剔除:GT 250 行表中位重建只剩 125)。幻觉超高行已由 _clean_segs 封顶
    # (≤band高/8),故取 max 安全;稀疏列短读的行底部补空对齐。
    all_rows = []
    for r in range(n_band):
        grids = parsed[r]
        lens = [len(g) for g in grids if g]
        if not lens:
            continue
        nrows = max(lens)
        band = [[] for _ in range(nrows)]
        for c in range(n_col):
            g = grids[c]
            wc = W[c]
            for i in range(nrows):
                cells = list(g[i]) if (g and i < len(g)) else []
                band[i].extend((cells + [""] * wc)[:wc])
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


# 结构哨兵(grid_ocr 表头重建写入,渲染层折叠):
# COLSPAN=被左侧格横向吞并的格位 → 左格发 colspan=N
# ROWSPAN=被上方格纵向吞并的格位 → 上格发 rowspan=N,本格不发 td(GT口径)
COLSPAN = "\x00cs"
ROWSPAN = "\x00rs"


def _one_table(rows):
    # 不修剪尾空列:有框表的尾空列由框线定义、GT保留为空td(尾部空列全空,
    # 修剪致td丢失);自由读时代的补齐残留不值得为它杀真列
    out = [_GT_TABLE_OPEN]
    for ri, row in enumerate(rows):
        tds = []
        j = 0
        while j < len(row):
            c = row[j] if row[j] is not None else ""
            if c == ROWSPAN:                     # 被上格纵向吞并,不发td
                j += 1
                continue
            span = 1
            while j + span < len(row) and row[j + span] == COLSPAN:
                span += 1
            rs = 1
            while (ri + rs < len(rows) and j < len(rows[ri + rs])
                   and rows[ri + rs][j] == ROWSPAN):
                rs += 1
            attrs = ""
            if span > 1:
                attrs += ' colspan="%d"' % span
            if rs > 1:
                attrs += ' rowspan="%d"' % rs
            tds.append("<td%s>%s</td>" % (attrs, "" if c == COLSPAN else c))
            j += span
        out.append("      <tr>%s</tr>" % "".join(tds))
    out.append("</table>")
    return "\n".join(out)


def rows_to_html(rows, panel_n=1):
    """左右并排子表 → 按几何中缝数 panel_n 等分成 N 个独立 <table>（与 GT 的 N 个
    <table> 1:1 对齐）；否则原样输出单表。

    panel_n 来自 slicer 的「横线断裂」几何信号（_panel_seams），取代旧的列类型周期
    (_panel_period)：后者会把无横线的单表按列类型周期误拆成 N 表(TEDS 大跌)，
    而横线断裂只在真并排(横线在中缝断开)时触发 → 不误拆。"""
    if panel_n and panel_n >= 2 and rows:
        W = Counter(len(r) for r in rows).most_common(1)[0][0]
        if W >= panel_n and W % panel_n == 0:
            K = W // panel_n
            parts = []
            for n in range(panel_n):
                pr = [r[n * K:(n + 1) * K] for r in rows
                      if len(r) >= (n + 1) * K
                      and any((c or "").strip() for c in r[n * K:(n + 1) * K])]
                parts.append(_one_table(pr))
            return "\n\n".join(parts)
    return _one_table(rows)


# ---------------------------------------------------------------------------
# 单表组装主入口
# ---------------------------------------------------------------------------
def _filled(row):
    return [x for x in row if (x or "").strip()]


def _mode_len(grid):
    """grid 各非空行填充格数的众数 = 这块数据的满列宽。"""
    L = [len(_filled(r)) for r in grid if _filled(r)]
    return Counter(L).most_common(1)[0][0] if L else 0


def _peel_ragged_top(bands, framed=False):
    """band0 的最左 tile 若比其它 tile 多出【顶部短行】(标签只出现在最左 tile)→ 弹出作
    caption、从 grid 去掉。修「标签揉进表头致对角错位」:子表标签"男性 3年交"是只占左 2-3
    格的半行,只在最左列 tile,stitch 按 nrows=max 逐行对齐时它和其它 tile 的表头对齐 →
    整表每行=上行左半+下行右半、对角劈裂(多子表 TEDS 崩)。弹掉后行对齐、标签成
    caption(对 ro)。判据:最左 tile 行数 > 其它 tile,且多出的顶行格数 ≤3。

    **向下合并**:遇到单格文字行(如 API 把表头角"年度/年龄"单独成行)、且其下一行恰比
    数据满宽少一列(== 满宽-1)→ 不剥进 caption,而把这格 prepend 回下一行补回首列、停。
    修 API 把表头角单独成行致数据整体左移一列。仅此一格、条件严格
    (下行须正好缺一列),有框正常表(下行=满宽,不缺列)不触发。"""
    if not bands or not bands[0]:
        return ""
    row0 = bands[0]
    grids = [g for g in row0 if g]
    if len(grids) < 2:
        return ""                                   # 单 tile 无法判"只在最左"
    left = grids[0]
    base = min(len(g) for g in grids[1:])           # 其它 tile 的行数
    caps = []
    while len(left) > base and left and len(_filled(left[0])) <= 3:
        cell = _filled(left[0])
        # 结构判据①:下一行首格(角格)空 → 候选是被 API 劈开的角格 → 填回不剥。区分子表标签
        # vs 角格轴表头:标签下方是合规表头(角格非空)→剥;角格下方是裸数据行(角格空)→塞回。
        # 纯结构不靠词表,修无框多子表把角格剥成 caption 的泄漏。
        if not framed and len(left) >= 2 and left[1] and not (left[1][0] or "").strip():
            left[1][0] = " ".join(cell)
            left.pop(0)
            break
        # 判据②(原向下合并):单格文字 + 下行恰缺一列 → prepend 补首列
        if len(cell) == 1 and len(left) >= 2 and not cell[0].strip().replace(".", "").isdigit():
            full = _mode_len(left[1:])              # 下方数据满列宽
            if full >= 3 and len(_filled(left[1])) == full - 1:
                left[1] = [cell[0]] + left[1]
                left.pop(0)
                break
        caps.append(" ".join(cell))
        left.pop(0)                                 # 原地改 bands → 下游对齐
    return "\n".join(caps)


def stitch_single_table(tile_outputs, meta):
    """单表组装：清洗各 tile 分段(丢展平幻觉) → 全部 band 平铺 → 剥 ragged 顶 →
    `_reconstruct_grid` 重建(列重建/稀疏补位) → HTML(+caption)。"""
    n_band = len(tile_outputs)
    n_col = max((len(row) for row in tile_outputs), default=0)
    col_cells = meta.get("col_cells", [])
    col_cuts = meta.get("col_cuts")
    row_cuts = meta.get("row_cuts")
    panel_n = meta.get("panel_n", 1)
    # 有框补空列只对单表:panel(左右并排)的列含中间缝、补空列会破坏 panel_n 拆分(TEDS 崩)
    framed = meta.get("col_framed", False) and panel_n < 2
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

    # 全部分段平铺成 band 序列（不做子表边界判定）。
    # 段索引对齐:稀疏列(段数<max_seg)在三角形列分段宽表里缺的是**靠前**的段,
    # 按 max_seg 末端对齐(idx=j-(max_seg-len(sc)));max_seg=1 时 offset=0 行为不变。
    bands = []
    first_caption = ""
    for r in range(n_band):
        max_seg = max((len(segs[r][c]) for c in range(len(segs[r]))), default=0)
        for j in range(max_seg):
            band_row = []
            seg_cap = ""
            for c in range(n_col):
                sc = segs[r][c] if c < len(segs[r]) else []
                idx = j - (max_seg - len(sc))
                if 0 <= idx < len(sc):
                    cap, grid = sc[idx]
                    band_row.append(grid)
                    if cap and not seg_cap and _is_caption_like(cap):
                        seg_cap = cap               # 该段前的"标题样"文字
                else:
                    band_row.append(None)
            if seg_cap and not first_caption and not bands:
                first_caption = seg_cap             # 文档顶标题
            bands.append(band_row)

    lead = _peel_ragged_top(bands, framed=framed)  # 无框:空角格→填回(去泄漏);剥标签去对角错位
    rows = _reconstruct_grid(bands, col_cells, col_cuts, framed=framed)
    html = rows_to_html(rows, panel_n)
    cap = (first_caption + "\n\n" + lead).strip("\n") if (first_caption and lead) else (first_caption or lead)
    return (cap + "\n\n" + html) if cap else html
