# -*- coding: utf-8 -*-
"""多子表几何切分：把含多个子表的整图切成单子表图，再各自走 run_one、按序拼接。

动机：table_teds 按出现顺序逐表配对 GT 与 pred；旧 stitch 把 N 个子表读成 1 个大
table 时只配上 GT[0]、其余全 0 分（db515 原 0.107、ce5799/f64061/ec745 直接 0）。
先几何切成 N 个独立子表、各自 run_one → 每个小表 TEDS 高 → 逐表对齐，分数大涨
（db515 0.107→0.989、1de69 0.600→0.808）。

三层架构（subtables 把整图切成有序块，调用方 run_one_split 逐块处理）：
A. 裁表外小文字块：split_table_texts 剥出页眉/页脚/水印/页码 → kind='text'。
B. 裁子表：在主表区按"子表缝"切段；前导矮段(标题)→ kind='text'，子表主体→ 'table'。
C. text 块单独 API 识别成纯文本、table 块走 run_one，按阅读顺序拼接（A/B 的小块共用
   同一套"裁剪→识别→拼接"）。

子表缝判据（_row_bounds）：连续白线(整行墨<0.3%×W)成"白带"，切点须同时满足
① 白带高 ≥2×行缝基准(子表缝纵向远高于行间缝，实测真子表缝 2.5~10 倍、单表内空行≈1
倍)；② 缝内无框线竖线贯穿(≤2 条)——按"框线逻辑找完全空白"，挡住稀疏表底部那种"墨量
是白行、但边框列线仍连续穿过"的假缝(cabe16d5)。这两条把单表误拆降到 0，省掉了魔数
"段数<4"。空段(墨<1%)丢弃；panel 横线断裂切左右并排(_panel_seam_xs，零误拆)。
"""
import numpy as np

from slicer_table import split_table_texts


def _panel_seam_xs(g):
    """左右并排子表的竖直中缝 x 坐标：靠"横向长墨线(行)被竖直白带打断"判定。
    与 slicer._panel_seams 同源，但返回缝的 x 位置（用于切分）而非缝数。零误拆。"""
    W = g.shape[1]
    step = max(1, W // 1500)
    g2 = g[::step, ::step]
    Hs, Ws = g2.shape
    dark = (g2 < 170)
    thr = 0.25 * Ws
    minrun = 0.1 * Ws
    covx = np.zeros(Ws, bool)
    found = False
    for y in range(Hs):
        row = dark[y]
        if row.sum() < thr:
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        runs = e - s
        if runs.size == 0 or runs.max() < thr:
            continue
        found = True
        for a, b, L in zip(s, e, runs):
            if L >= minrun:
                covx[a:b] = True
    if not found:
        return []
    xs = np.where(covx)[0]
    xlo, xhi = xs[0], xs[-1]
    seams = []
    i = xlo
    while i <= xhi:
        if not covx[i]:
            j = i
            while j <= xhi and not covx[j]:
                j += 1
            if j - i > 0.03 * Ws:
                seams.append((i + j) // 2 * step)
            i = j
        else:
            i += 1
    return seams


_SEAM_K = 2.0
_SEAM_VMAX = 2


def _row_bounds(dark, k=_SEAM_K, vmax=_SEAM_VMAX):
    """单栏内的上下子表切点：**子表缝 = 高白带 且 无框线竖线贯穿**。

    判据一(缝高)：连续白带(整行墨<0.3%×W)高度 ≥ k×行缝基准(本栏白带高度中位)。子表
    边界纵向远高于普通行间缝——实测误拆单表的"缝"=行缝(缝高/行缝≈1.0)、真子表缝是
    行缝 2.5~10 倍，k=2 落在空带 [1.1,2.5] 正中。**不用 width_gate Otsu**：单表行缝
    等高时 Otsu 退化成全留、把每条行缝当切点。

    判据二(无竖线)：缝内贯穿的竖直墨柱(框线/列线)数 ≤ vmax。稀疏表底部的空行边框、
    列线仍连续穿过——墨量上是白行、却不是真子表边界(cabe16d5 底部 9 条竖线贯穿=仍在
    表内)，而真子表缝处框线断开(db515/1de69 = 0 条)。这条按"框线逻辑找完全空白"，替
    掉了原来的魔数"段数<4"。"""
    H, W = dark.shape
    white = dark.sum(1) < max(2, 0.003 * W)          # 连续白线（纯白行）
    bands = []                                        # 连续白带 (起,止,高)
    y = 0
    while y < H:
        if white[y]:
            s = y
            while y < H and white[y]:
                y += 1
            bands.append((s, y, y - s))
        else:
            y += 1
    if not bands:
        return [0, H]
    row_gap = float(np.median([h for _, _, h in bands]))   # 典型行缝高
    cuts = []
    for s, e, h in bands:
        if h < k * row_gap or not (20 < (s + e) // 2 < H - 20):
            continue
        if int((dark[s:e].mean(0) > 0.8).sum()) > vmax:    # 框线竖线贯穿 → 表内，非缝
            continue
        cuts.append((s + e) // 2)
    return [0] + sorted(cuts) + [H]


def subtables(im, pad=4):
    """把整图切成**有序块**列表 `[(kind, bbox), ...]`（阅读顺序，先上后下）。

    kind='text' : 小文字块——① `split_table_texts` 剥出的表外页眉/页脚/水印/页码；
                  ② 子表上方被切出的矮标题段(高<0.4×中位)。两者都交由调用方"裁剪→
                  单独 API 识别→拼成纯文本"，**共用同一套小块识别**，不污染表结构。
    kind='table': 子表主体，交由调用方走 run_one。

    单表时返回单个 ('table', 整图 或 主表bbox)；调用方按 table 块数 ≤1 退回 run_one。

    把矮标题段标成 'text'(而非旧的几何"并回表体")：table 块数完全一样(矮段本就不算
    table)，但标题被独立识别成文本、不会被 stitch 包成空 <table> 或读进表格第一行。
    末尾矮段不并(可能是真的末尾小子表，1de69 第5表 92px)——只认**前导**矮段为标题。"""
    tb, texts = split_table_texts(im)
    if tb is None:
        return [('table', (0, 0, im.width, im.height))]
    x0, y0, x1, y1 = tb
    items = [(by0, 'text', (bx0, by0, bx1, by1))      # ① 表外小文字块
             for (bx0, by0, bx1, by1) in texts]
    g = np.asarray(im.crop(tb).convert("L"))
    dark = (g < 128)
    H, W = g.shape
    colb = [0] + _panel_seam_xs(g) + [W]
    for c in range(len(colb) - 1):                    # ② 主表内切子表
        cx0, cx1 = colb[c], colb[c + 1]
        rb = _row_bounds(dark[:, cx0:cx1])
        segs = [(ra, rbb) for ra, rbb in zip(rb[:-1], rb[1:])
                if rbb - ra >= 20 and dark[ra:rbb, cx0:cx1].mean() >= 0.01]
        if not segs:
            continue
        med = float(np.median([b - a for a, b in segs]))
        lead = 0                                       # 前导矮段=标题→text
        while lead < len(segs) - 1 and (segs[lead][1] - segs[lead][0]) < 0.4 * med:
            lead += 1
        for i, (ra, rbb) in enumerate(segs):
            bb = (x0 + cx0, y0 + max(0, ra - pad), x0 + cx1, y0 + min(H, rbb + pad))
            items.append((y0 + ra, 'text' if i < lead else 'table', bb))
    items.sort(key=lambda t: t[0])                     # 阅读顺序
    return [(k, bb) for _, k, bb in items] or [('table', tb)]
