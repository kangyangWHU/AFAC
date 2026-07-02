# -*- coding: utf-8 -*-
"""共享几何原语：投影分段 / 框线检测 / 并排缝 / 表外文字剥离。

Stage I(crop)与 Stage II(slicer/stitch)都依赖这些低层几何,放中立模块避免跨阶段互相 import。
"""
import numpy as np

from config import BIN_FAINT, BIN_LINE


def _content_segs(proj, gap, thr=0.002):
    """投影 proj 上按 >thr 的连续内容分段，相邻段间隔 ≤gap 合并。返回 [(s,e),...]。"""
    idx = np.where(proj > thr)[0]
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for x in idx[1:]:
        if x - p <= gap:
            p = x
        else:
            out.append((s, p)); s, p = x, x
    out.append((s, p))
    return out


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


def _boundaries(lines, total):
    """整理成完整单元格边界序列（含 0 与 total，升序去重）。"""
    return sorted(set([0, total] + [int(x) for x in lines if 0 < x < total]))


def column_cuts(dark, dark180):
    """列切线(单一列检测,Stage I/II 共用)。返回 (col_lines, framed)。

    有框:墨柱线(_runlen_lines)。**框线阈值相对表高** min(120,0.6H)——救矮表(框线<120px
    被漏检误判无框→塌缩,如 00332e7f 框线109px)。
    无框:白缝。col_frac<0.01(lo=0.01)聚缝,保留宽≥10px(floor10 去数字白河/中等假缝),
    **不做 Otsu**——Otsu 双峰会被超宽 margin 缝骗、把真列缝误当窄峰滤掉致塌缩;floor10 保守
    (真列缝实测最小 12px,10 不误杀),简单稳。"""
    H = dark.shape[0]
    runl = _runlen_lines(dark180, min_run=min(120, int(0.6 * H)))
    colf = dark.mean(axis=0)
    idx = np.where(colf < 0.01)[0]
    gap = []
    if len(idx):
        runs = [[idx[0]]]                    # 连续白列(≤2px 断)聚成缝
        for x in idx[1:]:
            (runs[-1].append(x) if x - runs[-1][-1] <= 2 else runs.append([x]))
        gap = [int(np.mean(r)) for r in runs if r[-1] - r[0] + 1 >= 10]   # floor=10, 无 Otsu
    if len(runl) > 1 and len(runl) >= 0.5 * len(gap):
        return runl, True                    # 有框:墨柱线=真单元格边界
    return gap, False                        # 无框:白缝(floor10)


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
    dark = (g2 < BIN_LINE)
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


def split_table_texts(im, hi_frac=0.05, wd_frac=0.40):
    """从图里分出主表 bbox 与表外孤立文字块（页眉/页脚/水印/页码）。

    判据：表外文字块 = **矮**(高<图高 hi_frac) + **窄**(宽<图宽 wd_frac) + 非最大墨块。
    三角形表底部稀疏段虽矮但**宽**(横跨表列)→不满足"窄"→留在主表，不会被误拆；
    页脚/水印/页码又矮又窄→分出来单独识别。无文字块时主表 bbox=内容 bbox。

    返回 (主表bbox(x0,y0,x1,y1) 或 None, [文字块bbox,...])，坐标均为传入 im 像素。
    """
    g = np.asarray(im.convert("L"))
    dark = (g < BIN_FAINT)                # 松二值化:框线/标题常是淡灰/彩色(灰度130~180),
    #                                       严二值化会漏(如 f64061da 首标题灰度177,严墨0.0008/松0.0055)
    H, W = g.shape
    blks = []
    for (y0, y1) in _content_segs(dark.mean(axis=1), int(H * 0.025)):
        for (x0, x1) in _content_segs(dark[y0:y1 + 1].mean(axis=0), int(W * 0.02)):
            blks.append((x0, y0, x1, y1, int(dark[y0:y1 + 1, x0:x1 + 1].sum())))
    if not blks:
        return None, []
    big = max(b[4] for b in blks)
    texts = [b for b in blks
             if (b[3] - b[1]) < H * hi_frac and (b[2] - b[0]) < W * wd_frac and b[4] < big]
    tab = [b for b in blks if b not in texts]
    tb = (min(b[0] for b in tab), min(b[1] for b in tab),
          max(b[2] for b in tab), max(b[3] for b in tab))
    texts.sort(key=lambda b: (b[1], b[0]))      # 先上后下、同行左到右
    return tb, [b[:4] for b in texts]
