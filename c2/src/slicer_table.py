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


def _otsu_split(widths):
    """对一组缝宽做 Otsu 双峰分离，返回阈值：宽度 ≥ 阈值 = 列缝。

    原理（关键）：**同一张表内，文字内缝(字/位间隙)永远比列缝窄**——所以缝宽分布
    是双峰：窄峰=文字白河、宽峰=列缝，中间有谷。在本表自己的宽度直方图上找这个谷
    （Otsu 最大化类间方差），就能逐表自适应地把两者切开，无需任何固定/相对常数。
    宽度跨度可达数百倍，故在 log2 空间做。近似单峰（无明显窄/宽之分）→ 返回 0（全留）。
    """
    ws = np.asarray(widths, dtype=float)
    if len(ws) < 3 or ws.max() / max(1.0, ws.min()) < 2.0:
        return 0.0
    lw = np.log2(ws)
    edges = np.linspace(lw.min(), lw.max(), 50)
    hist, _ = np.histogram(lw, bins=edges)
    ctr = (edges[:-1] + edges[1:]) / 2
    cum = np.cumsum(hist)
    cumv = np.cumsum(hist * ctr)
    tot, totv = cum[-1], cumv[-1]
    best_t, best_var = 0.0, -1.0
    for i in range(1, len(hist)):
        w0 = cum[i - 1]
        w1 = tot - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cumv[i - 1] / w0
        m1 = (totv - cumv[i - 1]) / w1
        v = w0 * w1 * (m0 - m1) ** 2          # 类间方差
        if v > best_var:
            best_var, best_t = v, edges[i]
    return 2.0 ** best_t


def _gap_lines(dark_frac, lo=0.02, floor=3, width_gate=False):
    """无边框回退：单元格间的低墨"白缝"作为候选切点。

    `width_gate`（**仅列方向开**）：右对齐数字各位之间会形成**全高、近零墨的"白河"**
    ——与真列缝在墨量上完全无法区分（卡到 0 也分不开），导致 `lo` 阈值单用时列数虚高
    数倍（100 张列检出中位 411%、最高 846%、49/100 严重过分割）。唯一可区分量是**缝宽**，
    故按 `_otsu_split` 逐表分离窄(文字)/宽(列缝)两峰只留宽峰，逐表自适应、无固定常数。
    **行方向必须关**：行缝大小相近，Otsu 会把窄行缝当文字、只留分节大缝 → 行塌成几条
    （实测行检出 102%→13%）。colspan/JPEG 杂点用 lo 容差兜（不要求严格 ==0，否则跨列
    表头处真缝被删）。floor 先滤 1~2px 抗锯齿噪点。
    """
    idx = np.where(dark_frac < lo)[0]
    if len(idx) == 0:
        return []
    runs, run = [], [idx[0]]                 # 先聚成连续低墨段（缝），段内允许 ≤2px 断点
    for x in idx[1:]:
        if x - run[-1] <= 2:
            run.append(x)
        else:
            runs.append(run)
            run = [x]
    runs.append(run)
    if not width_gate:                       # 行方向：原样返回每段中心，不做宽度过滤
        return [int(np.mean(r)) for r in runs]
    runs = [r for r in runs if r[-1] - r[0] + 1 >= floor] or runs
    thr = _otsu_split([r[-1] - r[0] + 1 for r in runs])
    keep = [r for r in runs if (r[-1] - r[0] + 1) >= thr] or runs
    return [int(np.mean(r)) for r in keep]


def _runlen_lines(dark, min_run=120):
    """墨柱法：竖直方向**最长连续墨段 ≥ min_run** 的列 = 框线。

    关键（实测）：框线 vs 文字的真正区分量不是墨深、不是跳变次数，而是**竖直连续墨段
    长度**——框线是长墨柱(几百 px)，文字是碎段(中位 1px、最长 ~10px)，分离度 ~880×，
    且不依赖墨色深浅（浅灰线只要成段就远超文字）。
    用**绝对阈值而非 0.3×全高**：① 文字竖段最长 ~10px，120px 有 12× 余量，不会误判；
    ② 一次性解决 margin（表只占图高 19~29% → 相对阈值过不了）和子表堆叠（线只贯穿子表
    那一段）两个问题，无需先裁 bbox 或分子表（实测 9 张漏检表救回 7 张）。
    输入 dark 建议用较松二值化(g<180)，把抗锯齿的浅框线也算进来。
    """
    H, W = dark.shape
    run = np.zeros(W, dtype=np.int32)
    best = np.zeros(W, dtype=np.int32)
    for y in range(H):
        run = (run + 1) * dark[y]
        best = np.maximum(best, run)
    isl = best >= min_run
    lines, i = [], 0
    while i < W:
        if isl[i]:
            j = i
            while j < W and isl[j]:
                j += 1
            lines.append((i + j) // 2)
            i = j
        else:
            i += 1
    return lines


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


def _panel_seams(g):
    """检测「左右并排子表」：横线在某竖直位置**断裂**（横墨跨多栏，但最长连续段只到
    单栏宽）= N 个带框窄表并排印刷。返回中缝数（并排栏数 = 中缝数 + 1）。

    判据（实测，全 100 张仅命中 8c8c784c/bd843d61 两张真并排、零误拆）：
      - 先取「横线行」(最长横墨段 ≥0.25W)；无横线行 → borderless，竖线只是列分隔
        → 不拆（3fa0851c 7竖线6列也判 0）。
      - 横线行里长段(≥0.1W)覆盖的 x 若**贯穿无缺口** → 单表/纵向堆叠(横向不拆)；
        若有内部缺口(>0.03W) → 缝处横线断裂 = 并排，缝数即额外栏数。
    与列类型周期(_panel_period)不同：这是直接的几何结构信号，不会把无横线单表(3fa0851c)
    误判成并排。纵向堆叠子表横线贯穿全宽 → 此处不拆，交由上下维度处理。"""
    W = g.shape[1]
    step = max(1, W // 1500)
    g2 = g[::step, ::step]
    Hs, Ws = g2.shape
    dark = (g2 < 170)
    thr = 0.25 * Ws
    minrun = 0.1 * Ws
    covx = np.zeros(Ws, dtype=bool)
    found = False
    for y in range(Hs):
        row = dark[y]
        if row.sum() < thr:                  # 横线至少 thr 个墨：快速预筛
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        runs = ends - starts
        if runs.size == 0 or runs.max() < thr:
            continue
        found = True
        for s, e, L in zip(starts, ends, runs):
            if L >= minrun:
                covx[s:e] = True
    if not found:
        return 0
    xs = np.where(covx)[0]
    xlo, xhi = xs[0], xs[-1]
    seams = 0
    i = xlo
    while i <= xhi:
        if not covx[i]:
            j = i
            while j <= xhi and not covx[j]:
                j += 1
            if j - i > 0.03 * Ws:
                seams += 1
            i = j
        else:
            i += 1
    return seams


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
    # 先裁掉表外 margin（空白边）：否则 TILE_MAX 会在大片空白里硬切出多余空白条带
    # （有些表只占图高 19~29%，余下全是 margin → 多切好几刀）。裁到内容 bbox 留 8px 边。
    g0 = np.asarray(im.convert("L"))
    d0 = (g0 < 128)
    ys = np.where(d0.mean(axis=1) > 0.002)[0]
    xs = np.where(d0.mean(axis=0) > 0.002)[0]
    if len(ys) > 0 and len(xs) > 0:
        y0 = max(0, ys[0] - 8); y1 = min(g0.shape[0], ys[-1] + 9)
        x0 = max(0, xs[0] - 8); x1 = min(g0.shape[1], xs[-1] + 9)
        if (x1 - x0) < g0.shape[1] or (y1 - y0) < g0.shape[0]:
            im = im.crop((x0, y0, x1, y1))

    g = np.asarray(im.convert("L"))
    H, W = g.shape
    dark = (g < 128)
    dark180 = (g < 180)                       # 较松二值化，专给框线检测（救浅灰线）

    # 列分流：先用墨柱(绝对≥120px)找框线；找够了=有框→切在线上；不够→无框→白缝。
    # 墨柱对 margin/子表/浅线都稳，文字又冒充不了长墨柱；白缝旁的 Otsu 宽度门杀数字白河。
    runl_cols = _runlen_lines(dark180, min_run=120)
    gap_cols = _gap_lines(dark.mean(axis=0), width_gate=True)
    if len(runl_cols) > 1 and len(runl_cols) >= 0.5 * len(gap_cols):
        col_lines = runl_cols                # 有框：墨柱线就是真单元格边界
    else:
        col_lines = gap_cols                 # 无框：白缝
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
    grid_ok = len(col_lines) > 1 and len(row_lines) > 1

    col_bnd = _boundaries(col_lines, W)
    row_bnd = _boundaries(row_lines, H)

    # 先按像素分列组，得到每组最大列数 → 据此限制每 tile 行数，使 cell ≤ budget
    col_cuts, col_cells = _group_boundaries(col_bnd, tile_max)
    # 安全落刀：把每个 tile 列切点 snap 到 ±25px 内文字墨最少处，避免把单元格/数字从中间
    # 切开（白缝字间割裂趋零；墨柱误检文字列的切割 16.6%→0.4%）。±25px 不改变骨架列数。
    textink = _textink_cols(dark)
    col_cuts = _snap_cuts(col_cuts, textink, win=25)
    # 每列带的真实列数受物理上限约束：带宽/最小列宽。剔除字间幻影列，避免行预算虚低。
    def _real_cols(c):
        w = (col_cuts[c + 1] - col_cuts[c]) if c + 1 < len(col_cuts) else tile_max
        return min(col_cells[c], max(1, w // MIN_COL_PX))
    max_cols = max((_real_cols(c) for c in range(len(col_cells))), default=1)
    max_rows = max(1, cell_budget // max(1, max_cols))
    row_cuts, row_cells = _group_boundaries(row_bnd, tile_max, max_cells=max_rows)
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

    # 左右并排子表：横线断裂中缝数 + 1 = 并排栏数（多数表为 1，不拆）
    panel_n = _panel_seams(g) + 1

    meta = {"row_cuts": row_cuts, "col_cuts": col_cuts,
            "row_cells": row_cells, "col_cells": col_cells,
            "blank": blank, "grid": grid_ok, "split_bands": split_bands,
            "tile_boxes": tile_boxes, "overlap_x": overlap_x, "overlap_y": overlap_y,
            "panel_n": panel_n}
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
