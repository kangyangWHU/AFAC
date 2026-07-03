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
from PIL import Image, ImageDraw

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
    rb, rf = row_bnds(dark, dark180)
    cb, cf = col_bnds(dark, dark180)
    R, C = len(rb) - 1, len(cb) - 1
    meta = {"rows": R, "cols": C, "rb": rb, "cb": cb, "misaligned": False,
            "col_framed": bool(cf)}
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
    # 像素长宽比约束:API 拒收 >200:1(400)。矮表单tile可到 241:1(1de69d49 尾表 2行
    # 7000×29px)——列带宽 > 180×最矮行带高 时对半加密列带(切在列边界上,骨架拼接原生
    # 支持多tile;不垫白,内容无损)
    min_bh = min(rb[j] - rb[i] for i, j in row_bands)
    while col_bands:
        wmax = max(cb[j] - cb[i] for i, j in col_bands)
        if wmax <= 180 * max(1, min_bh) or all(j - i <= 1 for i, j in col_bands):
            break
        col_bands = _chunk(cb, max(1, -(-max(j - i for i, j in col_bands) // 2)))
    meta["row_bands"], meta["col_bands"] = row_bands, col_bands
    H, W = g.shape
    tiles = []
    for (ri, rj) in row_bands:
        row = []
        for (ci, cj) in col_bands:
            y0, y1 = max(0, rb[ri] - _EDGE_PAD), min(H, rb[rj] + _EDGE_PAD)
            x0, x1 = max(0, cb[ci] - _EDGE_PAD), min(W, cb[cj] + _EDGE_PAD)
            ink = dark[y0:y1, x0:x1].mean()
            if ink < _BLANK_TILE_INK:
                row.append(None)
                continue
            t = im.crop((x0, y0, x1, y1))
            if not rf:                             # **无框表画骨架行线进 tile**:VLM 看见
                t = t.convert("RGB")               # 显式行分隔→高格不再拆行(5年交)、行不
                dr = ImageDraw.Draw(t)             # 漏读(21→22);内部线 only,顶/底边不画。
                for i in range(ri + 1, rj):        # 有框表已有线,不叠画。
                    dr.line([(0, rb[i] - y0), (t.width, rb[i] - y0)],
                            fill=(0, 0, 0), width=2)
            row.append(t)
        tiles.append(row)
    return tiles, meta


def _up(t, factor):
    if t is not None and factor and factor > 1:
        return t.resize((round(t.width * factor), round(t.height * factor)), Image.LANCZOS)
    return t


def _parse_cap(raw):
    """解析 tile 输出 → (caption, rows)。tile 是从表内切出的,**不存在表外文字**——
    API 把跨列表头(colspan行,如"保单年度")当 caption 放在 <table> 前,必须回收。"""
    rows = parse_tile(raw) if raw else []
    cap = ""
    if raw and "<table" in raw.lower():
        head = raw[:raw.lower().index("<table")]
        cap = " ".join(head.split())
    return cap, rows


def ocr_seg(im, timeout=240):
    """单个 seg 的骨架 OCR。返回 (grid, ncalls, meta);grid=None 表示骨架不可信需回退。

    组装 = **骨架行级墨证据 × tile 读数逐行核销**(替代 tile 级补零/众数):
    · tile 的期望行 = 该 tile 列范围内**有文字墨**的骨架行(排除纯横线行——OCR 不输出
      空行/线行,右侧 tile 因 colspan 区顶部空白整体上移一行的错位由此消除,1674392a)
    · caption 回填:实读=期望-1 且有 caption → caption 是首个有墨行的内容(跨列表头)
    · 差额核销:实读<期望 → 信骨架补空(OCR漏行是常态);实读>期望 → 需同带≥2 tile
      佐证(或单tile带)才在带尾加一行,孤证=幻觉裁掉。"""
    tiles, meta = slice_grid(im)
    if meta["misaligned"]:
        return None, 0, meta
    up = meta["upsample"]
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    rb, cb = meta["rb"], meta["cb"]
    dark = np.asarray(im.convert("L")) < BIN_INK
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

    # 格级墨证据:cell_ink[i][j] = 骨架格(i,j)内部(收缩2px避开框线)文字墨≥3px。
    # 行对齐(哪些行有内容)和列摆放(哪些格有内容)共用——空白判定要求极低墨。
    R, C = meta["rows"], meta["cols"]
    cell_ink = np.zeros((R, C), dtype=bool)
    for i in range(R):
        y0, y1 = rb[i] + 2, rb[i + 1] - 2
        if y1 <= y0:
            y0, y1 = rb[i], rb[i + 1]
        for j in range(C):
            x0, x1 = cb[j] + 2, cb[j + 1] - 2
            if x1 <= x0:
                x0, x1 = cb[j], cb[j + 1]
            sub = dark[y0:y1, x0:x1]
            cnt = sub.sum(1)
            frac = sub.mean(1)
            line = frac > 0.5                      # 穿过格内的横线主体行
            lm = line.copy()                       # ±2px 膨胀:线的反锯齿灰边(frac
            for s in (1, 2):                       #  0.2~0.5,几十px)也一并排除,否则
                lm[:-s] |= line[s:]                #  保单年度底线的灰边使 row0 被误判
                lm[s:] |= line[:-s]                #  有字,数据行错进表头(1674392a)
            cell_ink[i, j] = bool(((cnt >= 3) & ~lm).any())

    # 先解析全部 tile
    parsed = {}
    for r in range(len(row_bands)):
        for c in range(len(col_bands)):
            parsed[(r, c)] = _parse_cap(outs.get((r, c)))

    # **列校准**(行列职责不对称:行估计=真值,列在稀疏区/标签区可能少):非空 tile 的行
    # 格数众数若一致 = 骨架列数+k(k>0),且 ≥2 个行带的 tile 同票(或该列带只有 1 个非空
    # tile) → 采纳 nc+k(5fdf46b0 三标签列被并 1 列,429 行每行一致多读 2 格=最强信号;
    # 稀疏空 tile 不投票)。
    from collections import Counter
    band_nc = []
    for c, (ci, cj) in enumerate(col_bands):
        nc = cj - ci
        votes = []
        if not meta.get("col_framed"):             # **只校准无框表**:有框列=框线,本就精确;
            for r in range(len(row_bands)):        # 且有框 tile 切点在框线上,±3px pad 带进
                _, rws = parsed[(r, c)]            # 框线+邻列残影,VLM 每行一致幻觉出一个
                if not rws:                        # 边缘格(9c7857f3 五个带全被投成+1,
                    continue                       # 数据中间散布假空格)——一致性骗过佐证
                cnts = [len(x) for x in rws if any(s.strip() for s in x)]
                if cnts:
                    votes.append(Counter(cnts).most_common(1)[0][0])
        if votes:
            top, n = Counter(votes).most_common(1)[0]
            if top > nc and (n >= 2 or len(votes) == 1):
                nc = top
        band_nc.append(nc)

    grid = []
    for r, (ri, rj) in enumerate(row_bands):
        band_idx = list(range(ri, rj))
        aligned = {}
        for c, (ci, cj) in enumerate(col_bands):
            cap, rows = parsed[(r, c)]
            E = [i for i in band_idx if cell_ink[i, ci:cj].any()]
            if cap and len(rows) == len(E) - 1:
                rows = [[cap]] + rows              # caption = 首个有墨行(跨列表头)
            while len(rows) > len(E):              # 实读超期望时先丢**全空行**:±3px pad
                empt = [k for k, x in enumerate(rows)          # 带进邻带行的底边残影,VLM
                        if not any(s.strip() for s in x)]      # 输出全空行,按序对齐会把
                if not empt:                                   # 整tile内容挤移一行(9c7857f3
                    break                                      # 多tile整块错位,0.78病根)。
                rows.pop(empt[0])                              # 全空行零信息,丢之必无害
            if r == 0 and len(rows) >= len(E) + 1 and len(rows) >= 2:
                a, b = rows[0], rows[1]            # 斜线表头:一个高格斜线分写两行,OCR
                ov = [t for t in range(min(len(a), len(b)))   # 拆成两行且仅第0格重叠
                      if a[t].strip() and b[t].strip()]       # → 合并,格0='下\上'
                if ov == [0]:                                 # (GT 口径:保单年度末\投保年龄)
                    m = [b[0] + "\\" + a[0]] + [x if x.strip() else y
                         for x, y in zip(a[1:] + [""] * (len(b) - len(a)),
                                         b[1:] + [""] * (len(a) - len(b)))]
                    rows = [m] + rows[2:]
            if len(rows) == len(E) + 1:
                # 拆行合并(行线/墨测试,非打分):拆出的两行本是同一条骨架行——相邻互补对
                # 合并后的非空格数须**恰等**该骨架行墨格数;真表头两行合并后对不上。
                # 通过者唯一才合并,否则保守裁尾。行是真值,读数必须与骨架一致。
                hits = []
                for k in range(len(rows) - 1):
                    a, b = rows[k], rows[k + 1]
                    L = max(len(a), len(b))
                    aa = a + [""] * (L - len(a))
                    bb = b + [""] * (L - len(b))
                    if not (any(x.strip() for x in aa) and any(x.strip() for x in bb)):
                        continue
                    if any(x.strip() and y.strip() for x, y in zip(aa, bb)):
                        continue
                    merged = [x if x.strip() else y for x, y in zip(aa, bb)]
                    nz = sum(1 for x in merged if x.strip())
                    if k < len(E) and nz == int(cell_ink[E[k], ci:cj].sum()):
                        hits.append((k, merged))
                if len(hits) == 1:
                    k, merged = hits[0]
                    rows = rows[:k] + [merged] + rows[k + 2:]
            aligned[c] = (E, rows)
        for i in band_idx:                         # 行强制一致:骨架行数即真值,多裁少补
            rowcells = []
            for c, (ci, cj) in enumerate(col_bands):
                E, rows = aligned[c]
                cells = rows[E.index(i)] if i in E and E.index(i) < len(rows) else []
                nc = band_nc[c]
                rowcells += list(cells[:nc]) + [""] * max(0, nc - len(cells))
            grid.append(rowcells)
    return grid, len(flat), meta
