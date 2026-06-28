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
from rapidfuzz.distance import Levenshtein as _Lev
from teds import _first_table, _parse_grid

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


def _seq_sim(a, b):
    sa = "|".join((x or "").strip() for x in a)
    sb = "|".join((x or "").strip() for x in b)
    m = max(len(sa), len(sb))
    return 1.0 if m == 0 else 1.0 - _Lev.distance(sa, sb) / m


def _append_cells_dedup(row, cells, max_overlap=8, sim_thresh=0.82):
    """横向 overlap 去重：当前行尾部与新 tile 行头部相似时丢掉重复前缀。"""
    cells = list(cells)
    best_k, best_s = 0, 0.0
    kmax = min(max_overlap, len(row), len(cells))
    for k in range(kmax, 1, -1):
        s = _seq_sim(row[-k:], cells[:k])
        if s > best_s:
            best_k, best_s = k, s
    if best_k and best_s >= sim_thresh:
        cells = cells[best_k:]
    row.extend(cells)


def _row_sim_cells(a, b):
    return _seq_sim(a, b)


def _extend_rows_dedup(all_rows, band, max_overlap=4, sim_thresh=0.86):
    """纵向 overlap 去重：上一 band 尾行与下一 band 头行相似时丢掉重复行。"""
    if not all_rows or not band:
        all_rows.extend(band)
        return
    best_k, best_s = 0, 0.0
    kmax = min(max_overlap, len(all_rows), len(band))
    for k in range(kmax, 0, -1):
        sims = [_row_sim_cells(a, b) for a, b in zip(all_rows[-k:], band[:k])]
        s = sum(sims) / len(sims)
        if s > best_s:
            best_k, best_s = k, s
    if best_k and best_s >= sim_thresh:
        band = band[best_k:]
    all_rows.extend(band)


# ---------------------------------------------------------------------------
# 单表 2D 重组（全局列宽一致 W_c）—— 原 stitch_table 逻辑，重构成可复用
# ---------------------------------------------------------------------------
def _width_segments(parsed, n_col):
    """检测「不同宽度子表上下堆叠」：某 tile-column 的众数列宽在某 band 突变
    （如 945104ed 的 t2：上半子表=9 列、下半子表=27 列），说明两个不同宽度的子表叠在
    一起。返回按该突变 band 切的 band 分段 [(b0,b1),...]。

    必须分段重组的原因：W_c 取全 band 众数时，会被占多数 band 的子表主导
    （t2 众数=9 → 下半子表的 t2 从 27 被压到 9 → 列数 77→61，TEDS 0.758 而非 0.865）。
    分段后每段各取自己的 W_c。单一宽度（绝大多数表）→ 返回单段，行为完全不变、零回归。
    """
    n_band = len(parsed)
    bw = [[(_mode_width(g) if g else 0)
           for g in (parsed[r] + [None] * n_col)[:n_col]] for r in range(n_band)]
    cuts = []
    for c in range(n_col):
        # 排除 ≤2 列的 band：纯文本兜底(只读出年度号单列)、近空 tile 会产生"1列"，
        # 那是 OCR 没读出数据的**假**宽度突变，不是真的窄子表。若不排除，连续多个兜底
        # band 会被误判成"不同宽度子表"、分段后独立重组成 1 列、不补齐(090853cd 行切小后
        # 底部稀疏区 47 行只剩 1 列、TEDS 0.997→0.69)。真子表最少也 2-3 列,排除 ≤2 安全。
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


def _reconstruct_grid(parsed, col_cells, col_cuts=None, deg_hi=1.8, deg_lo=0.4,
                      overlap_x=0, overlap_y=0, framed=False):
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
            out.extend(_reconstruct_grid(parsed[b0:b1], col_cells, col_cuts,
                                         deg_hi, deg_lo, overlap_x, overlap_y, framed))
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
            w = max(1, _mode(col_widths[c]))
            if framed and c < len(col_cells) and w < 0.7 * col_cells[c]:
                # 有框且 OCR 读出明显少于框线列数(漏列):把漏读的空列补到框线列数
                # (00332 表1 读33/框线67=49%、19a15357 读9/框线69 —— GT 把空列都标了 td)。
                # 阈值 0.7:OCR 读出接近框线时不补,免过补(3fa0851c 读6/框线7=86% 不补)。
                w = col_cells[c]
            W.append(w)
        else:
            cc = col_cells[c] if c < len(col_cells) else 1
            if not framed and density > 0 and pix_w(c) > 0:
                cc = min(cc, round(density * pix_w(c)))   # 无框才用密度封顶防列爆炸;有框直接用框线列数
            W.append(max(1, cc))

    # 第二遍：按 W_c 重组。band 行数取「最密集列的行数」= max(各列行数)。
    # 稀疏列(三角表数据列)会把空格行折叠→读得少；密集列(行标签列)读满全行=真实行数。
    # 不能用"中位/比中位"判断:稀疏列占多数时中位被拖低，密集列反被当成离群剔除
    # (实测 GT 250 行的表，中位重建只剩 125、按比中位剔除后也只 152)。
    # 幻觉(超高行)已由上游 _clean_segs 物理封顶(≤band高/8)，故此处直接取 max 安全。
    # 稀疏列短读的行在底部补空对齐。
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
                cells = (cells + [""] * wc)[:wc]
                if overlap_x:
                    _append_cells_dedup(band[i], cells)
                else:
                    band[i].extend(cells)
        if overlap_y:
            _extend_rows_dedup(all_rows, band)
        else:
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


def _row_sim(a, b):
    """两行整行签名的归一化相似度 ∈[0,1]（数字也参与，1=完全相同）。"""
    sa, sb = "|".join(a), "|".join(b)
    m = max(len(sa), len(sb))
    return 1.0 - _Lev.distance(sa, sb) / m if m else 1.0


_NUM_RE = re.compile(r"-?[\d,]+\.?\d*%?")


def _is_num(s):
    return bool(_NUM_RE.fullmatch((s or "").replace(" ", "")))


def _to_int(s):
    s = (s or "").replace(",", "").strip()
    return int(s) if re.fullmatch(r"-?\d+", s) else None


def _label_col(rows, thr=0.5):
    """标签列 = 自左起第一个填充率 ≥thr 的列。跳过『合并单元格只在段顶填值』的稀疏前导
    列（如险种/性别），否则逐行 _first_cell 会在空行漂移到右侧数据列、把数据行误判成
    『数字运行→文本』的子表边界（3fa0851c：前3列稀疏 → 误拆 → 0.456）。"""
    if not rows:
        return 0
    W = max(len(r) for r in rows)
    n = len(rows)
    for c in range(W):
        if sum(1 for r in rows if len(r) > c and (r[c] or "").strip()) / n >= thr:
            return c
    return 0


def _split_at_headers(rows, min_seg=3, sim_hi=0.95, max_frac=0.4):
    """在"子表边界行"处把行序列切成多个子表段（边界行归入其下方子表）。

    两个**与"标题长短/有无数字"无关**的边界信号(取并集)：
      B. **首列『数字运行→文本』**(主力)：数据行首列是数字、子表标题/表头首列是文本。
         沿首列向下,**只用数据行累积"数字运行"**,**忽略每段开头的文本行(表头/多级表头)**——
         否则重复表头会污染运行、让后续边界全漏(实测召回掉到 61%)。当数字运行 ≥2 后遇到
         文本行 → 该处即子表边界。对 OCR 鲁棒(认数字)、短标题不漏、无状态污染。
         GT 实测:多表召回 100%、单表仅 1 误检(交费期间这类文本列被合并的个案)。
         注:曾试"翻转后须重现表头"做确认,虽去掉那 1 误检,但召回暴跌到 85%(误杀边界后
         表头未在窗口内重现的真子表)——得不偿失,不用确认。
      A. **重复表头**(备份)：与首行整行高度相似的后续行，带 max_frac 防误判闸。GT 上被 B
         覆盖，但 B 依赖数字首列；纯文本首列的多子表由 A 兜底；0 误检故保留。
         sim_hi=0.95：真·重复表头是**完全相同**(sim=1.0)；上采样读准后，相邻投保年龄
         数据行(金额随年龄缓变)整行 sim 会冲到 ~0.85 被误判成表头而过切(2358594b
         上采样 0.65→0.45、块4→5)，故阈值从 0.85 提到 0.95 挡住"相似但非相同"的数据行。
    仅当能切出 ≥2 段、每段 ≥min_seg 行时才生效。
    """
    n = len(rows)
    if n < 2 * min_seg:
        return [rows]
    header = rows[0]
    bnds = set()

    # B. 标签列 数字运行(≥2,忽略开局文本) → 文本/数字序列表头 = 边界（主力）。
    # 用固定标签列(跳过稀疏合并列)而非逐行 _first_cell，避免空行漂移到数据列致误判。
    lc = _label_col(rows)
    fc = [(r[lc].strip() if len(r) > lc and r[lc] else "") for r in rows]
    run = 0
    for i in range(1, n):
        cur = fc[i]
        if not cur:
            continue
        if _is_num(cur):
            run += 1                          # 数据行,累积数字运行
        else:
            if run >= 2:                      # 已建立数字运行 → 该文本行是子表边界
                bnds.add(i)
                run = 0
            # run<2: 仍在该段开头表头区,忽略此文本行(不计边界、不重置已有运行)

    # A. 重复表头（仅当 B 一个边界都没找到时兜底：纯文本首列的多子表，B 的数字运行建不起来）。
    # 不与 B 取并集——并集会让 A 在 B 已正确切分的数字表上过切(实测召回 100%→96%)。
    if not bnds:
        hdr_match = [i for i in range(1, n) if _row_sim(rows[i], header) >= sim_hi]
        if 0 < len(hdr_match) <= max_frac * n:
            bnds |= set(hdr_match)

    segs, cur = [], []
    for i, row in enumerate(rows):
        if i in bnds and len(cur) >= min_seg:
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


def _one_table(rows):
    rows = _trim_trailing_empty_cols(rows)
    out = [_GT_TABLE_OPEN]
    for row in rows:
        tds = "".join("<td>%s</td>" % (c if c is not None else "") for c in row)
        out.append("      <tr>%s</tr>" % tds)
    out.append("</table>")
    return "\n".join(out)


def rows_to_html(rows, panel_n=1):
    """左右并排子表 → 按几何中缝数 panel_n 等分成 N 个独立 <table>（与 GT 的 N 个
    <table> 1:1 对齐）；否则原样输出单表。

    panel_n 来自 slicer 的「横线断裂」几何信号（_panel_seams），取代旧的列类型周期
    (_panel_period)：后者会把无横线的单表(3fa0851c 6列)按列类型周期误拆成 N 表(0.416)，
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
# 多子表重组主入口
# ---------------------------------------------------------------------------
def _filled(row):
    return [x for x in row if (x or "").strip()]


def _mode_len(grid):
    """grid 各非空行填充格数的众数 = 这块数据的满列宽。"""
    L = [len(_filled(r)) for r in grid if _filled(r)]
    return Counter(L).most_common(1)[0][0] if L else 0


def _peel_ragged_top(bands):
    """band0 的最左 tile 若比其它 tile 多出【顶部短行】(标签只出现在最左 tile)→ 弹出作
    caption、从 grid 去掉。修「标签揉进表头致对角错位」:子表标签"男性 3年交"是只占左 2-3
    格的半行,只在最左列 tile,stitch 按 nrows=max 逐行对齐时它和其它 tile 的表头对齐 →
    整表每行=上行左半+下行右半、对角劈裂(a4924b6d 12子表 TEDS 48)。弹掉后行对齐、表去
    错位 + 标签成 caption(对 ro)。判据:最左 tile 行数 > 其它 tile,且多出的顶行格数 ≤3。

    **向下合并**:遇到单格文字行(如 API 把表头角"年度/年龄"单独成行)、且其下一行恰比
    数据满宽少一列(== 满宽-1)→ 不剥进 caption,而把这格 prepend 回下一行补回首列、停。
    修 API 把表头角单独成行致年龄整体左移一列(a4924b6d 男性10年交)。仅此一格、条件严格
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
        if len(cell) == 1 and len(left) >= 2 and not cell[0].strip().replace(".", "").isdigit():
            full = _mode_len(left[1:])              # 下方数据满列宽
            if full >= 3 and len(_filled(left[1])) == full - 1:
                left[1] = [cell[0]] + left[1]       # 向下合并:补回缺的首列
                left.pop(0)
                break
        caps.append(" ".join(cell))
        left.pop(0)                                 # 原地改 bands → 下游对齐
    return "\n".join(caps)


def stitch_multi(tile_outputs, meta, single=False):
    """按 API 自带的多 <table> 边界拆子表、各自重建、拼回。单表走单表路径。

    single=True(全宽模式用):**不做任何子表检测** —— 几何层 subtables/run_one_split 已把子表
    切好,进来的 unit 就是单张子表。关掉 band 边界(caption 误判)和 _split_at_headers(表头行
    误判),只把所有带重建成一张表(列重建/稀疏补位仍走 _reconstruct_grid)。修「全宽单列tile
    被 stitch 误拆成多表」(8a4 被拆4张→11.3)。"""
    n_band = len(tile_outputs)
    n_col = max((len(row) for row in tile_outputs), default=0)
    col_cells = meta.get("col_cells", [])
    col_cuts = meta.get("col_cuts")
    row_cuts = meta.get("row_cuts")
    overlap_x = meta.get("overlap_x", 0)
    overlap_y = meta.get("overlap_y", 0)
    panel_n = meta.get("panel_n", 1)
    # 有框补空列只对单表:panel(左右并排)的列含中间缝、补空列会破坏 panel_n 拆分(8c8c784c 1.0→0.23)
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
                # 段索引对齐:稀疏列(段数<max_seg)在三角形列分段宽表里缺的是**靠前**的
                # 子表段——右列段(大保单年度末)对大投保年龄无数据,只在靠后子表才出现。
                # 故按 max_seg 末端对齐(idx=j-(max_seg-len(sc))),而非一律取第 j 段;
                # 否则后段的右列表头会被错配进前段、被 _is_numeric_header_row 误判成
                # 表头而过切(2358594b GT4→7、3c3f1666 GT3→5 的根因)。max_seg=1 时
                # offset=0,行为不变,不影响普通单段 band。
                idx = j - (max_seg - len(sc))
                if 0 <= idx < len(sc):
                    cap, grid = sc[idx]
                    band_row.append(grid)
                    if cap and not seg_cap and _is_caption_like(cap):
                        seg_cap = cap               # 该段前的"标题样"文字
                else:
                    band_row.append(None)
            # 边界判定：① 同 tile 出现新表段(j>0)；② 段前有标题样文字
            # (竖线缝 split_bands 实测净负，已弃用——会过切，offset 多子表收益)
            has_data = any(any(g for g in br) for br in cur_bands)
            boundary = False if single else ((j > 0) or (seg_cap and has_data))
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
        bands = cur_bands if subtables else []
        lead = _peel_ragged_top(bands)              # 弹掉最左tile多出的顶部标签行→caption(去对角错位)
        rows = _reconstruct_grid(bands, col_cells, col_cuts,
                                 overlap_x=overlap_x, overlap_y=overlap_y, framed=framed)
        segs = [rows] if (single and rows) else _split_at_headers(rows)
        if len(segs) >= 2:
            html = "\n\n".join(rows_to_html(s, panel_n) for s in segs)
        else:
            html = rows_to_html(rows, panel_n)
        cap = (first_caption + "\n\n" + lead).strip("\n") if (first_caption and lead) else (first_caption or lead)
        if cap:
            html = cap + "\n\n" + html
        return html

    # 多子表：逐个重建 + 拼标题。每个 API 子表再用"纯文字表头行"递归拆（重复表头=内部边界）
    parts = []
    for k, (cap, bands) in enumerate(subtables):
        rows = _reconstruct_grid(bands, col_cells, col_cuts,
                                 overlap_x=overlap_x, overlap_y=overlap_y, framed=framed)
        if not rows:
            continue
        segs = _split_at_headers(rows)
        html = ("\n\n".join(rows_to_html(s, panel_n) for s in segs)
                if len(segs) >= 2 else rows_to_html(rows, panel_n))
        caption = (first_caption if k == 0 else cap)
        parts.append((caption + "\n\n" + html) if caption else html)
    return "\n\n".join(parts)


# 兼容旧调用名
def stitch_table(tile_outputs, meta, **kw):
    return stitch_multi(tile_outputs, meta, single=kw.get("single", False))
