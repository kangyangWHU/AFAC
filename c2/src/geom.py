# -*- coding: utf-8 -*-
"""共享几何原语：投影分段 / 框线检测 / 并排缝 / 表外文字剥离。

Stage I(crop)与 Stage II(slicer/stitch)都依赖这些低层几何,放中立模块避免跨阶段互相 import。
"""
import numpy as np

from config import BIN_FAINT, BIN_LINE

# ---- 框线/行列检测阈值(几何常量,与图像内容自适应量区分) ----
FRAME_MIN_RUN = 120     # 框线判定:竖直/横向连续墨段 ≥此px(文字碎段最长~10px,12×余量)
FRAME_GATE = 0.5        # 框线路门控:框线数 ≥ GATE×白缝数 才走框线路(否则残线劫持)
LINE_COVER = 0.5        # 一行/列的均墨 ≥此 = 贯穿框线(横线检测/去竖线列共用)
ROW_WHITE_ABS = 3       # 白行判定:绝对墨量<此=白(一条行号扫描线~6-10px,不会误白)
ROW_CLEAN_ABS = 2       # 净行:墨<此(0~1px)。真行缝内必有净行;字符内假白缝(细笔画中段
#                         恰 2px,笔画贯穿)到不了 0——白段须含净行才算真缝(790bec64 劈数字);
#                         全局降白阈到 2 反而害 4 张(真缝中 2px 噪点行墨化→两行粘连)
ROW_NOISE_FRAC = 0.0005  # 白行噪声地板:允许 0.05%×宽 的二值化噪点(逐像素累积随宽缩放)
ROW_GAP_MIN = 3         # 真行缝最小高度px:字符内假白缝(细笔画扫描线)仅1~2px
FRAME_HFRAC = 0.6       # 矮表框线阈值:min(FRAME_MIN_RUN, HFRAC×表高)(救框线<120px矮表)
DATA_SPAN_MIN = 0.40    # 真数据行:墨迹横向跨度 ≥此×表宽(数据常不填满,0.4 仍挡集中标题/注释)
DATA_RUN_MIN = 3        # 真数据行:列向独立墨段 ≥此(多列);标题/注释=1~2 连续块
COL_WHITE_FRAC = 0.01   # 白列判定:列均墨 <此 = 候选列缝
COL_GAP_MIN = 10        # 真列缝最小宽度px(数字白河/中等假缝更窄;真列缝实测最小12px)


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
    runl = _runlen_lines(dark180, min_run=min(FRAME_MIN_RUN, int(FRAME_HFRAC * H)))
    keep = dark180.mean(axis=1) <= LINE_COVER      # 去横框线行(对称 row_bnds 去竖线列):
    colf = (dark[keep] if keep.any() else dark).mean(axis=0)   # 横线横贯全宽,不去则每列
    idx = np.where(colf < COL_WHITE_FRAC)[0]       # 都被线墨填满,白缝全灭(32799493 gap=0
    #                                                → FRAME_GATE 空虚通过,4根局部竖线误判有框)
    gap = []
    if len(idx):
        runs = [[idx[0]]]                    # 连续白列(≤2px 断)聚成缝
        for x in idx[1:]:
            (runs[-1].append(x) if x - runs[-1][-1] <= 2 else runs.append([x]))
        gap = [int(np.mean(r)) for r in runs if r[-1] - r[0] + 1 >= COL_GAP_MIN]  # 无 Otsu
    if len(runl) > 1 and len(runl) >= FRAME_GATE * len(gap):
        return runl, True                    # 有框:墨柱线=真单元格边界
    return gap, False                        # 无框:白缝(COL_GAP_MIN)


def col_bnds(dark, dark180):
    """列边界(对称于 row_bnds)。返回 (bnds, framed)；列数 = len(bnds) - 1。

    有框:竖框线**就是**列边界——竖线数 = 列数 + 1,左右边框已含在内,不补 0/W
    (补了在边框外各造一条幽灵列,+2:实测有框列精确 0%→93%)。
    无框:白缝(column_cuts 的 gap 路)本身是缝中心、非边框,首尾需补 0/W。
    单侧开边(边框缺一侧,如 d9a99684 -1 列)不补——与行的决策一致,交 OCR 定。"""
    lines, framed = column_cuts(dark, dark180)
    if framed and len(lines) > 1:
        return [int(x) for x in lines], True
    return _boundaries(lines, dark.shape[1]), False


def row_bnds(dark, dark180):
    """行边界(单一行检测)。返回 (bnds, framed)；行数 = len(bnds) - 1。

    有框:横框线**就是**行边界——框线数 = 行数 + 1,顶/底线已含在内,**不再补 0/H**
    (旧 _boundaries 补 0/H 会在顶线上方/底线下方各造一条 1~2px 幽灵行,+2)。
    无框:白行判据用**绝对墨量** < max(3, 0.0005W),不用相对占比(2%×全宽把稀疏行号行
    稀释成"白",阶梯表下半几十行漏光,如 94352240 est54/gt107);0.0005W 是噪声地板
    (二值化噪点逐像素累积、随宽度缩放),比行号行一条扫描线的墨(~6-10px)低。
    算行墨前**先去掉贯穿竖线列**(列均墨>0.5)——列线墨会把真白行染成非白。
    行 = 非白 run,内部边界 = 白 run 中心,首尾补 0/H(无框顶底无框线)。"""
    H, W = dark.shape
    runl = _runlen_lines(dark180.T, min_run=FRAME_MIN_RUN)
    gapr = _gap_lines(dark.mean(axis=1), width_gate=False)
    if len(runl) > 1 and len(runl) >= FRAME_GATE * len(gapr):
        # 不做"开边补界":个别开放边版式(行上画线、末行不封底,如 9c7857f3)会少估 1 行,
        # 但补界判据在"数据行/残条/表底注"间无稳判(三版实测 75/71/69%),不值复杂度——
        # 末行是否存在交 OCR 定(裁块本含末行,只是行数估计少 1)。less is more。
        return [int(y) for y in runl], True
    keep = dark180.mean(axis=0) <= LINE_COVER        # 去贯穿竖线列
    ink = dark[:, keep].sum(axis=1) if keep.any() else dark.sum(axis=1)
    white = ink < max(ROW_WHITE_ABS, int(ROW_NOISE_FRAC * W))
    hline = dark180.mean(axis=1) > LINE_COVER       # 残余横线行=分隔符非内容,按白并入缝
    white |= hline                                   # (横线墨≥thr 会自成4px假行→缝里两条线)
    y = 0                                            # 墨段高<ROW_GAP_MIN=噪点,按白(对称:
    while y < H:                                     #   真行缝≥3px,真字行同样≥3px——空白带
        if not white[y]:                             #   中 1 行 3px 噪墨自成假行,0ed86277)
            s = y
            while y < H and not white[y]:
                y += 1
            if y - s < ROW_GAP_MIN:
                white[s:y] = True
        else:
            y += 1
    bnds, runs, y = [0], [], 0
    while y < H:
        if white[y]:
            s = y
            while y < H and white[y]:
                y += 1
            # 内部白带且高≥ROW_GAP_MIN,且 [含横线行 或 含净行] → 边界取中心。
            # 高不足=字符内假白缝(090853cd +22);横线=分隔证据(字符内绝不含横线,
            # 且线周常有2~7px脏噪、无净行,cc1ea3a3);无横线时须含净行(min<CLEAN)——
            # 细笔画中段2px假缝到不了净(790bec64 劈数字)
            if s > 0 and y < H and y - s >= ROW_GAP_MIN \
                    and (hline[s:y].any() or int(ink[s:y].min()) < ROW_CLEAN_ABS):
                bnds.append((s + y) // 2)
                runs.append((s, y))                  # 记边界所在白段(近邻对杀弱用)
        else:
            y += 1
    # 近邻边界对:间距 < 0.5×行距中位 → 物理上不可能都真(表内行距均匀),必有一条穿字——
    # 斜线字形内的真 0 墨 dip 连净行检查都过(如"7"斜杠尾被隔成假行,790bec64/aef3bf0c)。
    # 杀弱者:白段含横线行的赢(分隔铁证);否则白段更高的赢;平手杀后者。迭代至无近邻对。
    # pitch 取表自身边界间距中位——上百个间隔混 1~2 个假的中位不动,表内自适应无外部常数。
    changed = True
    while changed and len(bnds) >= 3:
        changed = False
        pitch = float(np.median(np.diff(bnds + [H])))
        for j in range(1, len(bnds) - 1):
            if bnds[j + 1] - bnds[j] < 0.5 * pitch:
                a, b = runs[j - 1], runs[j]          # 近邻两条内部边界所在的白段
                sa = (2 if hline[a[0]:a[1]].any() else 0) + (a[1] - a[0]) / max(1.0, pitch)
                sb = (2 if hline[b[0]:b[1]].any() else 0) + (b[1] - b[0]) / max(1.0, pitch)
                k = j if sa >= sb else j - 1         # 弱者(平手杀后者)
                del bnds[k + 1], runs[k]
                changed = True
                break
    bnds.append(H)
    return bnds, False


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
    # _content_segs 的 (s,e) 含端点,bbox 按排他约定 +1——否则下游 im.crop() 把内容最后
    # 一行/列裁掉 1px(998ada4c 1px 底框线恰在最后一行,被裁 → 行数-1)
    tb = (min(b[0] for b in tab), min(b[1] for b in tab),
          max(b[2] for b in tab) + 1, max(b[3] for b in tab) + 1)
    texts.sort(key=lambda b: (b[1], b[0]))      # 先上后下、同行左到右
    return tb, [(b[0], b[1], b[2] + 1, b[3] + 1) for b in texts]
