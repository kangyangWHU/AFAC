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


def slice_long(im, target_h=5000, search=500, dark_thresh=200, min_h=800):
    """把面条图切成若干横条。

    target_h : 目标条高（实测甜点 ~5000）
    search   : 在目标切点附近 ±search 内找空白带下刀
    min_h    : 末段过短则并入上一条
    返回 (strips:[PIL], cuts:[int])，cuts 含 0 与 H。
    """
    w, h = im.size
    if h <= target_h:
        return [im], [0, h]

    ink = _ink_profile(im, dark_thresh)
    cuts = [0]
    y = target_h
    while y < h - min_h:
        c = _find_gap(ink, y, search)
        if c <= cuts[-1] + min_h:                 # 防止切点回退/过近
            c = min(h, cuts[-1] + target_h)
        cuts.append(c)
        y = c + target_h
    cuts.append(h)

    strips = [im.crop((0, cuts[i], w, cuts[i + 1]))
              for i in range(len(cuts) - 1)]
    return strips, cuts


if __name__ == "__main__":
    import glob
    import os
    from config import TRAIN_LONG_DIR
    f = sorted(glob.glob(os.path.join(TRAIN_LONG_DIR, "images", "*.jpg")))[0]
    im = Image.open(f)
    for H in (3000, 5000, 8000):
        strips, cuts = slice_long(im, target_h=H)
        hs = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
        print(f"H={H}: {len(strips)} 条, 条高 min={min(hs)} max={max(hs)} "
              f"(原高 {im.size[1]})")
