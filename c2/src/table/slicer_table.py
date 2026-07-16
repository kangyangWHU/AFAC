# -*- coding: utf-8 -*-
"""TABLE 切片器：严格在网格线处切分，并预判空白块。

设计依据（已实测）：
  - 网格线清晰（列线~125px 间距，行线密集）→ 在网格线处下刀，每个 tile 含
    **整数行、整数列**，避免半行 → 根治"差1行"与"展平幻觉"。
  - API 不能 prompt → 空白/稀疏区会幻觉。故**预判空白 tile（墨量极低）→ 标记跳过**，
    由 stitch 端按已知行列数填空 `<td></td>`，既免幻觉又省调用。
  - 暴露每个 tile 的**已知行列数**作为重组骨架，让 stitch 不必猜测 API 的行分割。

tile 规模由 ≤MAX_TILE_ROWS 行 × ≤MAX_TILE_COLS 列 + 像素 ≤TILE_MAX 直接限定。
现仅服务列错位表的自由读回退(run_table.ocr_table)；主路是 grid_ocr.slice_grid。
overlap 已退役(实测净负:密集表列乱/行翻倍,945104ed 0.230→0.758)。
"""
import numpy as np
from PIL import Image

from table.geom import _runlen_lines, _panel_seams, _gap_lines, _boundaries, column_cuts
from table.tiles import MAX_TILE_ROWS, MAX_TILE_COLS, upsample_for
from common.config import BIN_INK, BIN_FAINT

Image.MAX_IMAGE_PIXELS = None
TILE_MAX = 1500             # tile 像素硬上限
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


def _group_boundaries(bnd, max_px, max_cells):
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


def slice_table(im):
    """密度自适应 + 严格网格线 2D 切分。

    返回:
      tiles : 2D 列表 tiles[r][c] = PIL.Image 或 None（空白块，跳过 API）
      meta  : {
        'row_cuts','col_cuts'     : 切点像素位置（落在网格线上）
        'row_cells','col_cells'   : 每个 tile-row/col 跨的单元格行/列数（重组骨架）
        'blank'                   : 2D bool，True=空白块
      }
    """
    g = np.asarray(im.convert("L"))
    H, W = g.shape
    dark = (g < BIN_INK)
    dark180 = (g < BIN_FAINT)                       # 较松二值化，专给框线检测（救浅灰线）

    # 列分流(column_cuts,与 Stage I 剥标题共用):墨柱找框线,不够回退白缝(Otsu宽度门杀数字白河)。
    col_lines, col_framed = column_cuts(dark, dark180)
    # 行：与列对称用墨柱（横向最长墨段）。关键纠正——文字行**不会连成一条线**：
    # 一行 `8504 8504 …` 的字/格之间有缝，最长横墨段只有一个数字宽(中位~2px)，而横框线
    # 全宽贯穿(几千px)，120px 阈值有 ~60× 余量，不会把文字行误判成线。横墨柱比密度法更准
    # （行估对 59→71/100），且连 15px 挤死的行都能救（015bd47c 101%）。无框→回退低墨缝。
    runl_rows = _runlen_lines(dark180.T, min_run=120)
    gap_rows = _gap_lines(dark.mean(axis=1))
    if len(runl_rows) > 1 and len(runl_rows) >= 0.5 * len(gap_rows):
        row_lines = runl_rows                # 有横框线：切在线上
    else:
        row_lines = gap_rows                 # 只有边框/残横线时不被劫持 → 回退横白缝

    col_bnd = _boundaries(col_lines, W)
    row_bnd = _boundaries(row_lines, H)

    textink = _textink_cols(dark)
    # tile 直接限行列：每 tile ≤MAX_TILE_ROWS 行 × ≤MAX_TILE_COLS 列。
    # 全宽模式自动门控:表宽刚过 limit(1~1.4×)——拆列会把表切成不均衡的小块,8a4 那种"重复
    # 标签列被孤立成小tile"会触发 API 复读幻觉/丢列。改成不拆列、全宽短行(8行),下游走
    # stitch 单表组装(列重建/稀疏补位照常)。稀疏表(94352240)全宽也无害,只需宽度门控。
    ctm, mr = TILE_MAX, MAX_TILE_ROWS
    if TILE_MAX < W <= 1.4 * TILE_MAX:   # 全宽模式
        ctm, mr = W, 8
    row_cuts, row_cells = _group_boundaries(row_bnd, TILE_MAX, max_cells=mr)
    # 列 tile 切分单独用列上限（仅切列，不影响上面已定的行带划分）。
    # 安全落刀：列切点 snap 到 ±25px 内文字墨最少处，避免把单元格/数字从中间切开。
    col_cuts, col_cells = _group_boundaries(col_bnd, ctm, max_cells=MAX_TILE_COLS)
    col_cuts = _snap_cuts(col_cuts, textink, win=25)
    # 安全落刀（行，对称于列）：snap 到横向文字墨最少处，避免把一行文字从中间横切。
    # _textink_cols 作用在 dark.T 上 = 每原始行的横向短墨段(笔画)占比。
    row_cuts = _snap_cuts(row_cuts, _textink_cols(dark.T), win=25)

    # 整数二维积分图，快速算任意 tile 的墨量（判空白）
    integ = np.zeros((H + 1, W + 1), dtype=np.int64)
    integ[1:, 1:] = np.cumsum(np.cumsum(dark, axis=0), axis=1)

    def ink_frac(x0, y0, x1, y1):
        s = (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
        area = max(1, (x1 - x0) * (y1 - y0))
        return s / area

    tiles, blank = [], []
    for r in range(len(row_cuts) - 1):
        row_imgs, row_blank = [], []
        for c in range(len(col_cuts) - 1):
            x0, y0, x1, y1 = col_cuts[c], row_cuts[r], col_cuts[c + 1], row_cuts[r + 1]
            is_blank = ink_frac(x0, y0, x1, y1) < BLANK_INK
            row_blank.append(is_blank)
            row_imgs.append(None if is_blank else im.crop((x0, y0, x1, y1)))
        tiles.append(row_imgs)
        blank.append(row_blank)

    # 左右并排子表：横线断裂中缝数 + 1 = 并排栏数（多数表为 1，不拆）
    panel_n = _panel_seams(g) + 1

    # 密集判据：每 cell 平均像素的边长 = √(去margin面积 / (行数×列数))。
    # 边长小 = 小字密集 → API 在原分辨率会读崩(行幻觉/列漂移)。用二维(行×列)而非
    # 一维列间距：能抓"列稀但行密"(945104ed)。<UP_EDGE 则上采样到目标 UP_TARGET。
    # slicer 行列估计实测误差仅 2~4%(密集表亦然)，故此判据可信。
    nrow = sum(row_cells); ncol = sum(col_cells)
    edge = (H * W / max(1, nrow * ncol)) ** 0.5
    upsample = upsample_for(edge)

    meta = {"row_cuts": row_cuts, "col_cuts": col_cuts,
            "row_cells": row_cells, "col_cells": col_cells,
            "blank": blank, "col_framed": col_framed,
            "panel_n": panel_n, "upsample": upsample}
    return tiles, meta
