# -*- coding: utf-8 -*-
"""多子表几何切分：把含多个子表的整图切成单子表图，再各自走 run_one、按序拼接。

动机：table_teds 按出现顺序逐表配对 GT 与 pred；旧 stitch 把 N 个子表读成 1 个大
table 时只配上 GT[0]、其余全 0 分（db515 原 0.107、ce5799/f64061/ec745 直接 0）。
先几何切成 N 个独立子表、各自 run_one → 每个小表 TEDS 高 → 逐表对齐，分数大涨
（db515 0.107→0.989、1de69 0.600→0.808）。

三层架构（subtables 把整图切成有序块，调用方 run_one_split 逐块处理）：
A. 裁表外小文字块：split_table_texts 剥出页眉/页脚/水印/页码 → kind='text'。
B. 裁子表段：在主表区按"子表缝"切段，全标 kind='seg'（**不在几何层预判标题/表格**）。
C. text 块单独识别成纯文本；seg 块走 run_one，**按识别出的 td 数判**:多格(≥10)=真子表
   (保留 table)、单格/几格=标题(转纯文本)。按阅读顺序拼接。靠识别而非几何段高,矮的真
   子表(ec745 147px、td=306)不会被误当标题。

子表缝判据（_row_bounds）：连续白线(整行墨<0.3%×W)成"白带"，切点须同时满足
① 白带高 ≥2×行缝基准(子表缝纵向远高于行间缝，实测真子表缝 2.5~10 倍、单表内空行≈1
倍)；② 缝内无框线竖线贯穿(≤2 条)——按"框线逻辑找完全空白"，挡住稀疏表底部那种"墨量
是白行、但边框列线仍连续穿过"的假缝(cabe16d5)。这两条把单表误拆降到 0，省掉了魔数
"段数<4"。空段(墨<1%)丢弃；panel 横线断裂切左右并排(_panel_seam_xs，零误拆)。
"""
import numpy as np

from slicer_table import split_table_texts, _runlen_lines


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


def _vline_break_ys(dark180):
    """竖线断裂找横缝：框线竖线(纵向长墨柱)在子表之间**全部断开**的 y。对称于 panel 的
    横线断裂找左右(这里纵向墨柱被横向白带打断找上下)。仅有框表(竖线≥3)。救"白带被稀疏
    行打断、缝高不足"的有框密集表(88684b6b/471413c1)。

    两条收紧(避免顶部抬头标题被切、切到文字):
    ① 只取**中间**的断裂带(3%<s<97%H)——排除顶部抬头标题/表尾,它们不是子表间。
    ② 切点取断裂带**顶部 s**(上子表框结束处)——标题归下子表、不切到标题文字。
    注:"只剩第1列"(88c6dbb4 GT1)和"真子表标题区"(1c9ac6d2 GT3)的断裂带 cov 都=2、几何
    无法区分,故仍按<20%判断裂——宁可 88c6dbb4 多切一刀(td 判后 table 仍=1、无害),也不能
    漏切 1c9ac6d2(漏切=table 数错、TEDS 崩)。"""
    H = dark180.shape[0]
    vlines = _runlen_lines(dark180, min_run=120)
    if len(vlines) < 3:
        return []
    cov = np.zeros(H)
    for x in vlines:
        cov += dark180[:, x]
    brk = cov < len(vlines) * 0.2                   # 贯穿竖线<20%=断裂
    ys = []
    y = 0
    while y < H:
        if brk[y]:
            s = y
            while y < H and brk[y]:
                y += 1
            if y - s >= 3 and H * 0.03 < s < H * 0.97:   # ② 中间的带; ③ 切点取顶部 s
                ys.append(s)
        else:
            y += 1
    return ys


def _row_bounds(dark, dark180=None, k=_SEAM_K, vmax=_SEAM_VMAX):
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
    if dark180 is not None:                                # 判据三:竖线断裂(有框表)
        cuts += _vline_break_ys(dark180)
    cuts = sorted(set(cuts))
    merged = []                                            # 相邻(<120px)缝合并=同一缝
    for c in cuts:
        if not merged or c - merged[-1] >= 120:
            merged.append(c)
    return [0] + merged + [H]


def subtables(im, pad=4):
    """把整图切成**有序块**列表 `[(kind, bbox), ...]`（阅读顺序，先上后下）。

    kind='text' : `split_table_texts` 剥出的表外页眉/页脚/水印/页码，调用方单独识别成纯文本。
    kind='seg'  : 主表区切出的子表段——**不在几何层预判它是标题还是表格**。调用方对每段
                  run_one,按识别出的 td 数判定:多格(≥阈值)=真子表(table)、单格/几格=标题
                  (转纯文本)。实测真子表 td≥72、标题 td≤5,中间空带极宽,靠识别比靠"段高
                  <0.4×中位"那种几何猜可靠——后者会把矮的真子表(ec745 147px)误当标题。

    单表(只 1 个 seg)由调用方退回 run_one 整图。"""
    tb, texts = split_table_texts(im)
    if tb is None:
        return [('seg', (0, 0, im.width, im.height))]
    x0, y0, x1, y1 = tb
    items = [(by0, 'text', (bx0, by0, bx1, by1))      # ① 表外小文字块
             for (bx0, by0, bx1, by1) in texts]
    g = np.asarray(im.crop(tb).convert("L"))
    dark = (g < 128)
    dark180 = (g < 180)                               # 松二值化:给竖线断裂检测(救浅灰框线)
    H, W = g.shape
    colb = [0] + _panel_seam_xs(g) + [W]
    for c in range(len(colb) - 1):                    # ② 主表内切子表段(不预判类型)
        cx0, cx1 = colb[c], colb[c + 1]
        rb = _row_bounds(dark[:, cx0:cx1], dark180[:, cx0:cx1])
        for ra, rbb in zip(rb[:-1], rb[1:]):
            # 丢极小段(<20px)和空白段(墨<0.3%)。阈值 0.3% 卡在"空白带(墨<0.2%)"和
            # "稀疏标题/说明(墨0.3~1%)"之间:空白照丢(无框单表不切碎)、标题保留(不丢字)。
            if rbb - ra < 20 or dark[ra:rbb, cx0:cx1].mean() < 0.003:
                continue
            bb = (x0 + cx0, y0 + max(0, ra - pad), x0 + cx1, y0 + min(H, rbb + pad))
            items.append((y0 + ra, 'seg', bb))
    items.sort(key=lambda t: t[0])                     # 阅读顺序
    return [(k, bb) for _, k, bb in items] or [('seg', tb)]
