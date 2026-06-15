# -*- coding: utf-8 -*-
"""TABLE 切片器：严格在网格线处切分，并预判空白块。

设计依据（已实测）：
  - 网格线清晰（列线~125px 间距，行线密集）→ 在网格线处下刀，每个 tile 含
    **整数行、整数列**，避免半行 → 根治"差1行"与"展平幻觉"。
  - API 不能 prompt → 空白/稀疏区会幻觉。故**预判空白 tile（墨量极低）→ 标记跳过**，
    由 stitch 端按已知行列数填空 `<td></td>`，既免幻觉又省调用。
  - 暴露每个 tile 的**已知行列数**作为重组骨架，让 stitch 不必猜测 API 的行分割。

tile 受两条约束：像素 ≤TILE_MAX（防内部降采样糊字）、单元格数 ≤CELL_BUDGET
（防 API ~12k 字符输出截断）。
"""
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

TILE_MAX = 1500             # tile 像素硬上限
MIN_COL_PX = 40             # 真实表格列的最小像素宽。无框表回退到空白缝检测时，会把
# 单元格内的字间空隙(中位~14px)误判成列线→列数虚高数倍(实测某表 468 列 vs 真实 109)→
# max_cols 虚高→每 tile 行预算被压到 3→tile 数爆炸(1843)。用"1500px 内最多容纳
# TILE_MAX/MIN_COL_PX 个真实列"作物理上限：≥40px 的真列不会被误删(物理放不下更多),
# 只剔除 <40px 的字间幻影列。仅用于行预算估计，不改变实际切列与重建。
CELL_BUDGET = 300           # 单 tile 最大单元格数（防输出截断）。
# 实测：API 输出受 token 预算限制，输出超 ~7100 字符即截断(未闭合<table>、丢行)。
# 含表 tile 每格约 16 字符(p99=25)。CELL_BUDGET=600 时 10.1% 的 tile 截断、大表 TEDS 被拖低。
# 截断从 ~431 cell 起；400 已 0 截断。取 300 留余量(300×16≈4.8k,300×25≈7.5k)确保不截断。
# 之前不敢降是怕 tile 变多触发共享 key 限流——该约束已解除(改用限流重试)。
BLANK_INK = 0.0015          # tile 平均墨量低于此 → 判为空白块（仅跳真正空白；0.003 的faint表头不再误跳）


def _grid_lines(dark_frac, hi=0.6):
    """暗占比剖面里找网格线位置（贯穿线占比高）。合并相邻。"""
    idx = np.where(dark_frac > hi)[0]
    if len(idx) == 0:
        return []
    lines, s, p = [], idx[0], idx[0]
    for x in idx[1:]:
        if x - p > 3:
            lines.append((s + p) // 2)
            s = x
        p = x
    lines.append((s + p) // 2)
    return lines


def _gap_lines(dark_frac, lo=0.02):
    """无边框回退：低墨位置（单元格间空白）作为候选切点。"""
    idx = np.where(dark_frac < lo)[0]
    if len(idx) == 0:
        return []
    lines, run = [], [idx[0]]
    for x in idx[1:]:
        if x - run[-1] <= 2:
            run.append(x)
        else:
            lines.append(int(np.mean(run)))
            run = [x]
    lines.append(int(np.mean(run)))
    return lines


def _boundaries(lines, total):
    """整理成完整单元格边界序列（含 0 与 total，升序去重）。"""
    return sorted(set([0, total] + [int(x) for x in lines if 0 < x < total]))


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


def slice_table(im, tile_max=TILE_MAX, cell_budget=CELL_BUDGET,
                blank_ink=BLANK_INK):
    """密度自适应 + 严格网格线 2D 切分。

    返回:
      tiles : 2D 列表 tiles[r][c] = PIL.Image 或 None（空白块，跳过 API）
      meta  : {
        'row_cuts','col_cuts'     : 切点像素位置（落在网格线上）
        'row_cells','col_cells'   : 每个 tile-row/col 跨的单元格行/列数（重组骨架）
        'blank'                   : 2D bool，True=空白块
        'grid'                    : 网格线是否可靠
      }
    """
    g = np.asarray(im.convert("L"))
    H, W = g.shape
    dark = (g < 128)
    col_lines = _grid_lines(dark.mean(axis=0)) or _gap_lines(dark.mean(axis=0))
    row_lines = _grid_lines(dark.mean(axis=1)) or _gap_lines(dark.mean(axis=1))
    grid_ok = len(col_lines) > 1 and len(row_lines) > 1

    col_bnd = _boundaries(col_lines, W)
    row_bnd = _boundaries(row_lines, H)

    # 先按像素分列组，得到每组最大列数 → 据此限制每 tile 行数，使 cell ≤ budget
    col_cuts, col_cells = _group_boundaries(col_bnd, tile_max)
    # 每列带的真实列数受物理上限约束：带宽/最小列宽。剔除字间幻影列，避免行预算虚低。
    def _real_cols(c):
        w = (col_cuts[c + 1] - col_cuts[c]) if c + 1 < len(col_cuts) else tile_max
        return min(col_cells[c], max(1, w // MIN_COL_PX))
    max_cols = max((_real_cols(c) for c in range(len(col_cells))), default=1)
    max_rows = max(1, cell_budget // max(1, max_cols))
    row_cuts, row_cells = _group_boundaries(row_bnd, tile_max, max_cells=max_rows)

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
            is_blank = ink_frac(x0, y0, x1, y1) < blank_ink
            row_blank.append(is_blank)
            row_imgs.append(None if is_blank else im.crop((x0, y0, x1, y1)))
        tiles.append(row_imgs)
        blank.append(row_blank)

    # 有框表：识别"子表分界"的 row_cut——该处竖线消失（两个带框子表之间的缝）。
    # 正常行切点处竖线贯穿(数量多)，子表缝处竖线≈0。据此标记需强制拆分的 band。
    split_bands = set()
    bordered = col_frac.max() > 0.4 if (col_frac := dark.mean(axis=0)) is not None else False
    if bordered and len(row_cuts) > 2:
        def vlines_at(y, half=14):
            seg = dark[max(0, y - half):min(H, y + half)].mean(axis=0)
            idx = np.where(seg > 0.5)[0]
            return 0 if len(idx) == 0 else 1 + int((np.diff(idx) > 3).sum())
        band_vl = [vlines_at((row_cuts[r] + row_cuts[r + 1]) // 2)
                   for r in range(len(row_cuts) - 1)]
        typ = np.median([v for v in band_vl if v > 3]) if any(v > 3 for v in band_vl) else 0
        if typ > 3:
            for r in range(1, len(row_cuts) - 1):
                if vlines_at(row_cuts[r]) < typ * 0.3:   # 该切点竖线骤降=子表缝
                    split_bands.add(r)

    meta = {"row_cuts": row_cuts, "col_cuts": col_cuts,
            "row_cells": row_cells, "col_cells": col_cells,
            "blank": blank, "grid": grid_ok, "split_bands": split_bands}
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
