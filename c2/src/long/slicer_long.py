# -*- coding: utf-8 -*-
"""LONG（面条图）切片器：1D 水平投影找行间空白，在空白处下刀。

目的：在文字行之间（空白带）切，**不把一行字劈成两半**。
这直接缓解裸拼接缝处的残行问题（实测 readScore 偏低的主因）。

LONG 宽固定 ~1500（≤ API 1500px 原生上限），故只切高度、不切宽度。
返回切片列表与切点 y 坐标。
"""
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _ink_profile(im, dark_thresh=200):
    """每一行的"墨量"=该行暗像素数。空白带对应墨量≈0。"""
    g = np.asarray(im.convert("L"))
    return (g < dark_thresh).sum(axis=1)          # shape: (H,)


def _find_gap(ink, center, search):
    """在 [center-search, center+search] 内找墨量最小（最空白）的 y。"""
    lo = max(0, center - search)
    hi = min(len(ink), center + search)
    if hi <= lo:
        return center
    seg = ink[lo:hi]
    # 优先选完全空白(墨量0)且最靠近 center 的行；否则取最小墨量行
    zeros = np.where(seg == 0)[0]
    if len(zeros):
        # 离 center 最近的空白行
        idx = zeros[np.argmin(np.abs(zeros + lo - center))]
    else:
        idx = int(np.argmin(seg))
    return lo + int(idx)


RULE_THRESH = 245        # 框线阈值:表格框线常是浅灰(#ddd 级),200 的正文阈值检不到
_BAND_WIN = 600          # 竖线检测窗口高;表格只占文档一小段时,全图尺度会被稀释
MIN_RULES = 3            # 最少竖线条数:左边框+列线+右边框。2 条会把两栏排版误判成表格


def _n_rules(xs):
    """把相邻像素列合并成"条":一条线通常占 2 个像素列,按列数会高估一倍。"""
    if not xs:
        return 0
    xs = sorted(xs)
    return 1 + sum(1 for a, b in zip(xs, xs[1:]) if b - a > 2)


def table_bands(im, rule_thresh=RULE_THRESH, win=_BAND_WIN, min_run=400):
    """返回表格所在的 y 区间 [(y0,y1), ...]。

    两步:
      1. 粗筛出"疑似框线列"—— 某 600px 窗口内近乎贯通的列。窗口只用来发现 x,
         不用来定 y 边界:表格若跨在窗口边界上,占比会被稀释到阈值以下,
         用窗口定 y 会漏掉表格上沿,而切点恰恰常落在上沿附近。
      2. 用这些列**自身的纵向连续墨段**定 y 边界 —— 竖线画到哪,表格就到哪。

    竖线按**线**而不是按像素列计数(一条线通常占 2 个相邻像素列),且**逐带**核验:
    整篇取并集会被"目录用一根分栏线 + 别处有真表格"蒙混过去 —— 实测两栏目录整块
    被判成表格。表格至少有左边框+列线+右边框三条,故每个带独立要求 ≥MIN_RULES 条。
    """
    g = np.asarray(im.convert("L")) < rule_thresh
    h = g.shape[0]

    xs = set()
    for y0 in range(0, h, win // 2):              # 半窗重叠,少漏
        sub = g[y0:y0 + win]
        if sub.shape[0] < win // 2:
            continue
        xs.update(np.where(sub.mean(0) > 0.9)[0].tolist())
    if _n_rules(xs) < MIN_RULES:
        return []

    runs = []                                     # (y0, y1, x) —— 带上 x 才能逐带数线
    for x in sorted(xs):
        col = g[:, x]
        idx = np.where(col)[0]
        if not len(idx):
            continue
        brk = np.where(np.diff(idx) > 1)[0]
        for a, b in zip(np.r_[0, brk + 1], np.r_[brk, len(idx) - 1]):
            y0, y1 = int(idx[a]), int(idx[b])
            if y1 - y0 >= min_run:                # 短墨段是文字笔画,不是框线
                runs.append((y0, y1, x))

    runs.sort()
    bands = []                                    # [y0, y1, {x,...}]
    for y0, y1, x in runs:
        if bands and y0 <= bands[-1][1]:
            bands[-1][1] = max(bands[-1][1], y1)
            bands[-1][2].add(x)
        else:
            bands.append([y0, y1, {x}])
    return [(a, b) for a, b, xs_ in bands if _n_rules(xs_) >= MIN_RULES]


def _avoid_bands(c, bands, ink, prev_cut, search, min_h, target_h):
    """切点落在表格带内 → 退到该带上边界之上的最近空白带。

    **只对放得下的表让路**。比一整条还高的表(疾病清单动辄上万 px)无论怎么切都会
    被切开,让路买不到任何完整性,却会把切点顶到很上面、造出一条很短的条带并让
    后续切点连锁位移 —— 实测 266d63c2 就是这样把表格内容挤成了正文
    (read 99.0 → 65.1)。这种表直接维持原切点,交给 stitch 去接相邻条带的 <table>。
    """
    for y0, y1 in bands:
        if not (y0 < c < y1):
            continue
        if y1 - y0 > target_h:
            return c
        moved = _find_gap(ink, y0 - search // 2, search // 2)
        if prev_cut + min_h < moved < y0:
            return moved
        return c
    return c


def slice_long(im, target_h=5000, search=500, dark_thresh=200, min_h=800):
    """把面条图切成若干横条。

    target_h : 目标条高（实测甜点 ~5000）
    search   : 在目标切点附近 ±search 内找空白带下刀
    min_h    : 末段过短则并入上一条
    返回 (strips:[PIL], cuts:[int])，cuts 含 0 与 H。

    空白带判据分不清「段落间空白」和「表格行间空白」—— 表格横线上下同样是空白,
    于是切点专挑表格横线下刀。表被劈开后,上半段失去列网格,API 认不出那是表格,
    会把表头当普通文本读出去。故先算表格禁切区,再把落入其中的切点顶到表上方。
    """
    w, h = im.size
    if h <= target_h:
        return [im], [0, h]

    ink = _ink_profile(im, dark_thresh)
    bands = table_bands(im)
    cuts = [0]
    y = target_h
    while y < h - min_h:
        c = _find_gap(ink, y, search)
        if bands:
            c = _avoid_bands(c, bands, ink, cuts[-1], search, min_h, target_h)
        if c <= cuts[-1] + min_h:                 # 防止切点回退/过近
            c = min(h, cuts[-1] + target_h)
        cuts.append(c)
        y = c + target_h
    cuts.append(h)

    strips = [im.crop((0, cuts[i], w, cuts[i + 1]))
              for i in range(len(cuts) - 1)]
    return strips, cuts
