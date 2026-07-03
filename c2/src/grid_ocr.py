# -*- coding: utf-8 -*-
"""Stage II — 骨架切割 OCR:行列估计已准(几何层),tile 严格按估计边界切、结果严格按骨架组装。

原则(与几何层的分工):
- **一律以估计为准**:tile 切点=估计的行列边界(切在缝/线上,天然不劈单元格);OCR 返回
  与骨架不符 = 幻觉/漏读,中间 tile 强制对齐骨架(多裁少补);仅首/末 tile 容 ±1 行
  (开边表末行/表头两行等已知几何偏差),取 OCR 实读。
- **稀疏区补零**:多表头/三角表的稀疏区(除表头外基本空白)不靠 OCR 猜——空 tile(墨≈0)
  不调 API,直接按骨架补空 cell;半空 tile 的短行也按骨架列数补 "" 到位。
- 列错位表(rows_misaligned)是唯一例外:骨架不可信,回退整段自由读(交上层 ocr_table)。
"""
import numpy as np
from PIL import Image

import api_client as api
from config import MAX_CONCURRENCY, API_USER_IDS, BIN_INK, BIN_FAINT
from geom import row_bnds, col_bnds, rows_misaligned
from slicer_table import MAX_TILE_ROWS, MAX_TILE_COLS, UP_EDGE, UP_TARGET
from stitch_table import parse_tile

_BLANK_TILE_INK = 0.001   # tile 墨率<此=空白,不调 API,按骨架补空 cell
_EDGE_PAD = 3             # tile 四周留白px(边界在缝中心,±3 不吃邻格墨)
_UP_CAP = 2.0             # 上采样上限(沿用 slicer 实测结论)


def _chunk(bnds, max_cells):
    """把边界序列切成 tile 带:每带 ≤max_cells 个单元格,**均匀分配**(避免 15/15/6 碎尾,
    band 大小一致 API 读得更稳)。返回 [(lo_idx, hi_idx), ...](骨架索引)。"""
    n = len(bnds) - 1
    nb = max(1, -(-n // max_cells))          # ceil
    size = -(-n // nb)
    out, i = [], 0
    while i < n:
        j = min(n, i + size)
        out.append((i, j))
        i = j
    return out


_CELL_BUDGET = MAX_TILE_ROWS * MAX_TILE_COLS   # 375:单 tile 单元格预算。行列上限互相让渡——
#   矮子表(行少)列上限放宽(11行→34列,宽表不必切碎),窄表(列少)行上限放宽,减调用且保上下文


def slice_grid(im):
    """按几何骨架切 tile。返回 (tiles, meta);tiles[r][c]=PIL.Image|None(空白)。

    meta: rows/cols(骨架数), rb/cb(边界px), row_bands/col_bands(骨架索引带),
          upsample, misaligned(True=骨架不可信,调用方回退自由读)。"""
    g = np.asarray(im.convert("L"))
    dark, dark180 = g < BIN_INK, g < BIN_FAINT
    rb, _rf = row_bnds(dark, dark180)
    cb, _cf = col_bnds(dark, dark180)
    R, C = len(rb) - 1, len(cb) - 1
    meta = {"rows": R, "cols": C, "rb": rb, "cb": cb, "misaligned": False}
    if R < 1 or C < 1 or rows_misaligned(dark, dark180):
        meta["misaligned"] = True
        return None, meta
    # 密集判据(沿用 slicer):每 cell 平均边长小 → 上采样
    edge = (g.shape[0] * g.shape[1] / max(1, R * C)) ** 0.5
    meta["upsample"] = round(min(_UP_CAP, UP_TARGET / edge), 2) if edge < UP_EDGE else 1.0
    meta["cell_edge"] = round(edge, 1)
    # 行列上限按单元格预算互相让渡(矮子表列放宽/窄表行放宽)
    eff_r = min(R, MAX_TILE_ROWS)
    col_cap = max(MAX_TILE_COLS, _CELL_BUDGET // max(1, eff_r))
    eff_c = min(C, col_cap)
    row_cap = max(MAX_TILE_ROWS, _CELL_BUDGET // max(1, eff_c))
    row_bands = _chunk(rb, row_cap)
    col_bands = _chunk(cb, col_cap)
    meta["row_bands"], meta["col_bands"] = row_bands, col_bands
    H, W = g.shape
    tiles = []
    for (ri, rj) in row_bands:
        row = []
        for (ci, cj) in col_bands:
            y0, y1 = max(0, rb[ri] - _EDGE_PAD), min(H, rb[rj] + _EDGE_PAD)
            x0, x1 = max(0, cb[ci] - _EDGE_PAD), min(W, cb[cj] + _EDGE_PAD)
            ink = dark[y0:y1, x0:x1].mean()
            row.append(None if ink < _BLANK_TILE_INK else im.crop((x0, y0, x1, y1)))
        tiles.append(row)
    return tiles, meta


def _up(t, factor):
    if t is not None and factor and factor > 1:
        return t.resize((round(t.width * factor), round(t.height * factor)), Image.LANCZOS)
    return t


def _fit(cells_rows, nr, nc):
    """强制对齐骨架 (nr×nc):列补 ""/截断(稀疏区补零),行多裁少补空行。"""
    rows = [r[:nc] + [""] * max(0, nc - len(r)) for r in cells_rows[:nr]]
    return rows + [[""] * nc for _ in range(nr - len(rows))]


def ocr_seg(im, timeout=240):
    """单个 seg 的骨架 OCR。返回 (grid, ncalls, meta);grid=None 表示骨架不可信需回退。"""
    tiles, meta = slice_grid(im)
    if meta["misaligned"]:
        return None, 0, meta
    up = meta["upsample"]
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    flat = [(r, c) for r in range(len(tiles)) for c in range(len(tiles[r]))
            if tiles[r][c] is not None]
    from concurrent.futures import ThreadPoolExecutor
    def call(args):
        i, (r, c) = args
        return api.call_safe(_up(tiles[r][c], up), timeout=timeout,
                             user_id=API_USER_IDS[i % len(API_USER_IDS)])
    outs = {}
    if flat:
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, len(flat))) as ex:
            for (r, c), o in zip(flat, ex.map(call, list(enumerate(flat)))):
                outs[(r, c)] = o
    # 组装:先定每带行数(仅首/末带容 ±1:各 tile 实读行数取多数,与骨架差 1 时信实读——
    # 开边末行/表头两行等几何已知偏差),再逐 tile 强制对齐 → 拼成全表。
    grid = []
    nR = len(row_bands)
    for r, (ri, rj) in enumerate(row_bands):
        nr = rj - ri
        if r in (0, nR - 1):
            reads = [len(parse_tile(outs[(r, c)])) for c in range(len(col_bands))
                     if (r, c) in outs]
            if reads:
                from collections import Counter
                real = Counter(reads).most_common(1)[0][0]
                if abs(real - nr) == 1:
                    nr = real
        band_rows = [[] for _ in range(nr)]
        for c, (ci, cj) in enumerate(col_bands):
            nc = cj - ci
            cells = _fit(parse_tile(outs[(r, c)]) if (r, c) in outs else [], nr, nc)
            for k in range(nr):                   # 空白 tile → _fit([]) = 全空 cell(补零)
                band_rows[k].extend(cells[k])
        grid.extend(band_rows)
    return grid, len(flat), meta
