# -*- coding: utf-8 -*-
"""TABLE 切片器：严格在网格线处切分，并预判空白块。

设计依据（已实测）：
  - 网格线清晰（列线~125px 间距，行线密集）→ 在网格线处下刀，每个 tile 含
    **整数行、整数列**，避免半行 → 根治"差1行"与"展平幻觉"。
  - API 不能 prompt → 空白/稀疏区会幻觉。故**预判空白 tile（墨量极低）→ 标记跳过**，
    由 stitch 端按已知行列数填空 `<td></td>`，既免幻觉又省调用。
  - 暴露每个 tile 的**已知行列数**作为重组骨架，让 stitch 不必猜测 API 的行分割。

tile 规模由 ≤MAX_TILE_ROWS 行 × ≤MAX_TILE_COLS 列 + 像素 ≤TILE_MAX 直接限定
（本是图片 OCR API、无 token 截断，不需 CELL_BUDGET 格数中介）。
"""
import os
import numpy as np
from PIL import Image

from geom import (_runlen_lines, _panel_seams, split_table_texts,
                  _gap_lines, _boundaries, column_cuts)

Image.MAX_IMAGE_PIXELS = None
MAX_TILE_COLS = int(os.environ.get("C2_MTC", "15"))   # 单 tile 最大列数 = 15。
# OCR 在"宽 tile × 长数字"上会漏列(b326 26列/tile 读不全),限 15 列后 015bd47c +16、
# b326dfb6 +15、密集表零退化。靠解耦(max_rows 用不限列的真实列数算,列上限只切列、不挪
# 行带切点)避免误伤子表表头检测(00332 列上限耦合行带→丢表头→崩的根因)。C2_MTC 可调。

TILE_MAX = 1500             # tile 像素硬上限
UP_EDGE = 40                # cell 边长(√像素/cell) < 此值 = 密集小字 → 上采样重读
UP_TARGET = 75             # 上采样目标 cell 边长(放大到 API 读得清的密度)
# overlap 关闭（实测净负）：重叠让每个 tile 多读邻块一条，靠 stitch 去重还原。但在密集
# 表(18px 行、数字列)上，① overlap_x 的逐行 _append_cells_dedup 每行删掉的格数不一致 →
# 列数被打乱成 58/66/74/84(非均匀)；② overlap_y 的行去重在数字行上 sim 匹配失败 → 行翻倍。
# 945104ed 实测:开 overlap TEDS=0.230(326行/列乱)、关 overlap TEDS=0.758(196行/列均匀)。
# 无 overlap 时 _reconstruct_grid 走 extend 路径,每行强制对齐 W[c] → 结构均匀。
OVERLAP_X = 0
OVERLAP_Y = 0
MIN_COL_PX = 40             # 真实表格列的最小像素宽。无框表回退到空白缝检测时，会把
# 单元格内的字间空隙(中位~14px)误判成列线→列数虚高数倍(实测某表 468 列 vs 真实 109)→
# max_cols 虚高→每 tile 行预算被压到 3→tile 数爆炸(1843)。用"1500px 内最多容纳
# TILE_MAX/MIN_COL_PX 个真实列"作物理上限：≥40px 的真列不会被误删(物理放不下更多),
# 只剔除 <40px 的字间幻影列。仅用于行预算估计，不改变实际切列与重建。
MAX_TILE_ROWS = 25          # 单 tile 最大行数。与 MAX_TILE_COLS=15 一起直接限定 tile 规模
# (≤25行 × ≤15列)。本是图片 OCR API、无 token 截断,故移除了 CELL_BUDGET(格数中介)。
# 行25 优于行20:行20 切得太碎、表头行易被 OCR 漏读→子表切分错(1674392a +52、583ac07b +41、
# a1aaef73 +18、015bd47c +7),而行20 不救任何表(090/b326 持平)。隔离实验定论:行25。
BLANK_INK = 0.0015          # tile 平均墨量低于此 → 判为空白块（仅跳真正空白；0.003 的faint表头不再误跳）


def _textink_cols(dark, frac=0.15):
    """每列**文字墨占比**：只数属于短竖段(run < frac×H = 笔画)的墨，排除长墨柱(框线)。
    用作落刀安全分——切线 snap 到此剖面的局部最小点，就避开了文字、且不怕切在框线上。
    """
    H, W = dark.shape
    thr = frac * H
    runlen = np.zeros((H, W), np.int32)
    c = np.zeros(W, np.int32)
    for y in range(H):
        c = (c + 1) * dark[y]
        runlen[y] = c
    total = np.zeros((H, W), np.int32)
    last = np.zeros(W, np.int32)
    for y in range(H - 1, -1, -1):
        last = np.where(dark[y], np.maximum(last, runlen[y]), 0)
        total[y] = last
    return (dark & (total < thr)).mean(axis=0)


def _snap_cuts(cuts, textink, win=25):
    """把每个内部切点 snap 到 ±win 内**文字墨最少**的 x：避免把单元格/数字从中间切开。
    实测无框白缝切点的字间割裂随之趋零、有框墨柱误检的文字列切割从 16.6%→0.4%。
    端点(0/总宽)不动。"""
    W = len(textink)
    out = [cuts[0]]
    for x in cuts[1:-1]:
        a, b = max(0, x - win), min(W, x + win + 1)
        out.append(a + int(np.argmin(textink[a:b])))
    out.append(cuts[-1])
    # snap 后可能撞重/乱序 → 去重保序
    return sorted(set(out))


def _group_boundaries(bnd, max_px=TILE_MAX, max_cells=10 ** 9):
    """把单元格边界贪心分组成 tile 切点。**保证每个 tile ≤ max_px**：
    优先在检测到的网格线处下刀；若 max_px 内没有网格线（无框表检测稀疏），
    则在 start+max_px 处**强制切**（避免出现 4000+px 的超大 tile 被 API 降采样糊读）。
    返回 (cuts, cell_counts)。
    """
    bnd = sorted(set(bnd))
    total = bnd[-1]
    cuts = [bnd[0]]
    cell_counts = []
    while cuts[-1] < total:
        start = cuts[-1]
        limit = start + max_px
        chosen = None
        for b in bnd:                           # 选 max_px 内最远、且 cell 数 ≤max_cells 的网格线
            if b <= start:
                continue
            if b > limit:
                break
            ncell = sum(1 for x in bnd if start < x <= b)
            if ncell <= max_cells:
                chosen = b
            else:
                break
        if chosen is None or chosen == start:   # max_px 内无可用网格线 → 强制切
            chosen = min(total, limit)
        cuts.append(chosen)
        cell_counts.append(max(1, sum(1 for x in bnd if start < x <= chosen)))
    return cuts, cell_counts


def slice_table(im, tile_max=TILE_MAX, blank_ink=BLANK_INK, peel=True,
                col_tile_max=None, max_rows=None):
    """密度自适应 + 严格网格线 2D 切分。

    返回:
      tiles : 2D 列表 tiles[r][c] = PIL.Image 或 None（空白块，跳过 API）
      meta  : {
        'row_cuts','col_cuts'     : 切点像素位置（落在网格线上）
        'row_cells','col_cells'   : 每个 tile-row/col 跨的单元格行/列数（重组骨架）
        'blank'                   : 2D bool，True=空白块
        'grid'                    : 网格线是否可靠
      }

    peel: 是否跑 split_table_texts 剥离表外文字。仅整页图需要(剥页眉页脚水印)。
    **run 里子表裁块传 peel=False**——子表已被 subtables 剥过页垃圾、
    已切好,这里再剥会把矮子表的薄数据行误判成"窄+矮"文字整片剥掉(90a4388e 表1 被剥 79%
    墨量、只剩表头)。子表标题不必剥,API 会当 caption 识别。
    """
    # 先切出主表、剥离表外孤立文字块（页眉/页脚/水印/页码）：否则它们会被当成多出来的
    # 列/行污染网格（实测列虚增，整图 slice 比真值多 3 列）。文字块坐标记进 meta，由 runner
    # 单独识别后按位置拼回表格前/后。无文字块时等价于裁到内容 bbox（兼顾原 margin 裁剪职责）。
    tb, _texts = split_table_texts(im) if peel else (None, [])
    text_blocks = []
    if tb is not None:
        x0, y0, x1, y1 = tb
        for (bx0, by0, bx1, by1) in _texts:
            where = "before" if by1 <= y0 else "after"   # 表上方=前，下方=后
            text_blocks.append((where, (bx0, by0, bx1, by1)))
        cx0 = max(0, x0 - 8); cy0 = max(0, y0 - 8)
        cx1 = min(im.width, x1 + 9); cy1 = min(im.height, y1 + 9)
        if (cx1 - cx0) < im.width or (cy1 - cy0) < im.height:
            im = im.crop((cx0, cy0, cx1, cy1))

    g = np.asarray(im.convert("L"))
    H, W = g.shape
    dark = (g < 128)
    dark180 = (g < 180)                       # 较松二值化，专给框线检测（救浅灰线）

    # 列分流(column_cuts,与 Stage I 剥标题共用):墨柱找框线,不够回退白缝(Otsu宽度门杀数字白河)。
    col_lines, col_framed = column_cuts(dark, dark180)
    # 行：与列对称用墨柱（横向最长墨段）。关键纠正——文字行**不会连成一条线**：
    # 一行 `8504 8504 …` 的字/格之间有缝，最长横墨段只有一个数字宽(中位~2px)，而横框线
    # 全宽贯穿(几千px)，120px 阈值有 ~60× 余量，不会把文字行误判成线。横墨柱比密度法更准
    # （行估对 59→71/100），且连 15px 挤死的行都能救（015bd47c 101%）。无框→回退低墨缝。
    runl_rows = _runlen_lines(dark180.T, min_run=120)
    gap_rows = _gap_lines(dark.mean(axis=1), width_gate=False)
    if len(runl_rows) > 1 and len(runl_rows) >= 0.5 * len(gap_rows):
        row_lines = runl_rows                # 有横框线：切在线上
    else:
        row_lines = gap_rows                 # 只有边框/残横线时不被劫持 → 回退横白缝
    grid_ok = len(col_lines) > 1 and len(row_lines) > 1   # col_framed 由 column_cuts 给出

    col_bnd = _boundaries(col_lines, W)
    row_bnd = _boundaries(row_lines, H)

    textink = _textink_cols(dark)
    # tile 直接限行列：每 tile ≤MAX_TILE_ROWS 行 × ≤MAX_TILE_COLS 列。不再用 CELL_BUDGET 中介
    # (本是图片 OCR API、无 token 截断,格数由行列上限直接限定);max_rows 固定也天然不受列上限
    # 影响 → 不会经"行带挪动"破坏 _split_at_headers 子表切分(00332 列上限耦合丢表头的旧根因)。
    # col_tile_max: 列方向像素上限,默认=tile_max。设成很大(=W)则列不拆、全宽一个列tile
    # (8a4 那种"宽刚过limit、拆列会孤立出重复列小块→幻觉"的表用)。
    # max_rows: 每 tile 行数上限,默认 MAX_TILE_ROWS。全宽时配短行(如8)防长输出截断。
    _ctm = col_tile_max or tile_max
    _mr = max_rows or MAX_TILE_ROWS
    # 全宽模式自动门控:表宽刚过 limit(1~1.4×)——拆列会把表切成不均衡的小块,8a4 那种"重复
    # 标签列被孤立成小tile"会触发 API 复读幻觉/丢列。改成不拆列、全宽短行(8行),下游 ocr_table
    # 走 stitch 单表模式(不拆子表、保留列重建)。稀疏表(94352240)全宽也无害——stitch 单表模式
    # 照样补稀疏列(实测仍 98),故只需宽度门控、不需密度判据。
    fullwidth = False
    if col_tile_max is None and tile_max < W <= 1.4 * tile_max:
        fullwidth = True
        _ctm, _mr = W, 8
    row_cuts, row_cells = _group_boundaries(row_bnd, tile_max, max_cells=_mr)
    # 列 tile 切分单独用列上限（仅切列，不影响上面已定的行带划分）。
    # 安全落刀：列切点 snap 到 ±25px 内文字墨最少处，避免把单元格/数字从中间切开。
    col_cuts, col_cells = _group_boundaries(col_bnd, _ctm, max_cells=MAX_TILE_COLS)
    col_cuts = _snap_cuts(col_cuts, textink, win=25)
    # 安全落刀（行，对称于列）：snap 到横向文字墨最少处，避免把一行文字从中间横切。
    # _textink_cols 作用在 dark.T 上 = 每原始行的横向短墨段(笔画)占比。
    row_cuts = _snap_cuts(row_cuts, _textink_cols(dark.T), win=25)
    total_rows = sum(row_cells)
    # overlap 只给多 row-band 的复杂表启用。
    # 小表、少 band 的规则表原本结构稳定，overlap 会引入重复内容。
    use_overlap = total_rows >= 20 and len(row_cells) >= 8
    overlap_x = OVERLAP_X if use_overlap else 0
    overlap_y = OVERLAP_Y if use_overlap else 0

    # 整数二维积分图，快速算任意 tile 的墨量（判空白）
    integ = np.zeros((H + 1, W + 1), dtype=np.int64)
    integ[1:, 1:] = np.cumsum(np.cumsum(dark, axis=0), axis=1)

    def ink_frac(x0, y0, x1, y1):
        s = (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
        area = max(1, (x1 - x0) * (y1 - y0))
        return s / area

    tiles, blank, tile_boxes = [], [], []
    for r in range(len(row_cuts) - 1):
        row_imgs, row_blank, row_boxes = [], [], []
        for c in range(len(col_cuts) - 1):
            x0, y0, x1, y1 = col_cuts[c], row_cuts[r], col_cuts[c + 1], row_cuts[r + 1]
            is_blank = ink_frac(x0, y0, x1, y1) < blank_ink
            ex0 = max(0, x0 - (overlap_x if c > 0 else 0))
            ex1 = min(W, x1 + (overlap_x if c + 1 < len(col_cuts) - 1 else 0))
            ey0 = max(0, y0 - (overlap_y if r > 0 else 0))
            ey1 = min(H, y1 + (overlap_y if r + 1 < len(row_cuts) - 1 else 0))
            row_blank.append(is_blank)
            row_boxes.append((ex0, ey0, ex1, ey1))
            row_imgs.append(None if is_blank else im.crop((ex0, ey0, ex1, ey1)))
        tiles.append(row_imgs)
        blank.append(row_blank)
        tile_boxes.append(row_boxes)

    # 左右并排子表：横线断裂中缝数 + 1 = 并排栏数（多数表为 1，不拆）
    panel_n = _panel_seams(g) + 1

    # 密集判据：每 cell 平均像素的边长 = √(去margin面积 / (行数×列数))。
    # 边长小 = 小字密集 → API 在原分辨率会读崩(行幻觉/列漂移)。用二维(行×列)而非
    # 一维列间距：能抓"列稀但行密"(945104ed)。<UP_EDGE 则上采样到目标 UP_TARGET。
    # slicer 行列估计实测误差仅 2~4%(密集表亦然)，故此判据可信。
    nrow = sum(row_cells); ncol = sum(col_cells)
    edge = (H * W / max(1, nrow * ncol)) ** 0.5
    _upcap = float(os.environ.get("C2_UPCAP", "2.0"))   # 上采样倍数上限=2:再高的放大把 tile 撑得过宽
    # 反而让 OCR 漏列(015bd47c cap2 减漏列)，且 2 倍已够密集小字读清;C2_UPCAP 可调。
    upsample = round(min(_upcap, UP_TARGET / edge), 2) if edge < UP_EDGE else 1.0

    meta = {"row_cuts": row_cuts, "col_cuts": col_cuts,
            "row_cells": row_cells, "col_cells": col_cells,
            "blank": blank, "grid": grid_ok, "col_framed": col_framed,
            "tile_boxes": tile_boxes, "overlap_x": overlap_x, "overlap_y": overlap_y,
            "panel_n": panel_n, "cell_edge": round(edge, 1), "upsample": upsample,
            "text_blocks": text_blocks, "fullwidth": fullwidth}
    return tiles, meta


if __name__ == "__main__":
    import glob
    import os
    from config import TRAIN_TABLE_DIR
    files = sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                   key=os.path.getsize)
    for md in [files[len(files) // 2], files[5], glob.glob(
            os.path.join(TRAIN_TABLE_DIR, "mds", "f64061da*.md"))[0]]:
        uuid = os.path.basename(md)[:-3]
        im = Image.open(os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg"))
        tiles, meta = slice_table(im)
        nr, nc = len(meta["row_cells"]), len(meta["col_cells"])
        nblank = sum(sum(row) for row in meta["blank"])
        total_cells = sum(meta["row_cells"]) * sum(meta["col_cells"])
        print(f"[{uuid[:8]}] size={im.size} 网格={nr}x{nc}={nr*nc}块 "
              f"空白块={nblank} | 骨架={sum(meta['row_cells'])}行x"
              f"{sum(meta['col_cells'])}列={total_cells}格 "
              f"行分布={meta['row_cells']} 列分布={meta['col_cells']}")
