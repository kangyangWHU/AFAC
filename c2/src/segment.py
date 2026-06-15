# -*- coding: utf-8 -*-
"""子表切分：几何找候选分界 + API 确认标题。

多子表图（GT 用多个 <table>+标题分隔，占训练集 ~46%）被合并成一个大 <table>
是 TABLE 主要失分点。稳健方案 = **候选（几何，高召回）+ 确认（API 读字，防误切）**：

  1) 候选分界（几何）：
     - 有框表：数每横带的**竖线**，竖线骤降处=候选分界（你的洞见）；
     - 无框表：宽度覆盖度的**空白带**=候选分界。
  2) 确认（API）：每个候选裁一窄带送 API，**读出的是标题文字才确认**（短+无表+中文为主）。
     → 单表内部的"假分界"（合并格/section）读到的是数据/无内容 → 被否决 → **不误切单表**。
  3) 在确认处切 → 每子区跑现成单表流水线。

成本：仅候选数次 API 调用/表（多子表 ~表数次，单表 ~0）。
"""
import re
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# 几何：找候选分界 y（高召回，可含假阳性，交给 API 确认）
# ---------------------------------------------------------------------------
def is_bordered(im, hi=0.4):
    arr = np.asarray(im.convert("L"))
    return (arr < 128).mean(axis=0).max() > hi


def _vlines_in_band(dark, y0, y1, hi=0.55):
    colfrac = dark[y0:y1].mean(axis=0)
    idx = np.where(colfrac > hi)[0]
    return 0 if len(idx) == 0 else 1 + int((np.diff(idx) > 3).sum())


def _row_coverage(im, nbins=40):
    g = np.asarray(im.convert("L"))
    H, W = g.shape
    dark = (g < 150)
    edges = np.linspace(0, W, nbins + 1).astype(int)
    cov = np.zeros(H)
    for b in range(nbins):
        cov += dark[:, edges[b]:edges[b + 1]].any(axis=1)
    return cov / nbins


def _blocks_to_gaps(blocks, H):
    """相邻内容块之间的间隙中点 = 候选分界 y。"""
    return [(a[1] + b[0]) // 2 for a, b in zip(blocks[:-1], blocks[1:])]


def _runs(flags, ys, stripe, min_h):
    blocks = []
    i = 0
    while i < len(flags):
        if flags[i]:
            s = ys[i]
            while i < len(flags) and flags[i]:
                i += 1
            e = ys[i - 1] + stripe
            if e - s >= min_h:
                blocks.append((s, e))
        else:
            i += 1
    return blocks


def candidate_separators(im, stripe=20, min_block_h=100):
    """返回候选分界 y 列表（高召回）。有框走竖线、无框走覆盖度。"""
    arr = np.asarray(im.convert("L"))
    dark = (arr < 128)
    H = arr.shape[0]
    ys = list(range(0, H, stripe))
    if is_bordered(im):
        counts = [_vlines_in_band(dark, y, min(H, y + stripe)) for y in ys]
        vals = [c for c in counts if c > 3]
        typ = np.median(vals) if vals else 0
        thr = max(3, typ * 0.25)
        is_tab = [c >= thr for c in counts]
    else:
        cov = _row_coverage(im)
        # 行带平均覆盖度 > 阈值 = 内容/表区
        is_tab = [cov[y:min(H, y + stripe)].mean() > 0.12 for y in ys]
    blocks = _runs(is_tab, ys, stripe, min_block_h)
    if len(blocks) <= 1:
        return []
    return _blocks_to_gaps(blocks, H)


# ---------------------------------------------------------------------------
# 确认：API 读窄带，判是否标题
# ---------------------------------------------------------------------------
_CJK = re.compile(r"[一-鿿]")
_DIGIT = re.compile(r"\d")


def is_caption_text(text):
    if not text:
        return False
    t = re.sub(r"```\w*", "", text).strip()
    if "<table" in t.lower():
        return False
    if len(t) > 220:
        return False
    cjk, dig = len(_CJK.findall(t)), len(_DIGIT.findall(t))
    return cjk >= 3 and cjk >= dig          # 中文为主、数字不占多数


def confirm_separators(im, cand_ys, call_fn, half=45):
    """对每个候选 y 裁窄带 [y-half, y+half] 调 API，返回确认为标题的 y 列表。"""
    W, H = im.size
    confirmed = []
    for y in cand_ys:
        crop = im.crop((0, max(0, y - half), W, min(H, y + half)))
        if is_caption_text(call_fn(crop)):
            confirmed.append(y)
    return confirmed


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def segment(im, call_fn):
    """返回子区 [(y0,y1), ...]；单表/无确认分界 → [(0,H)]。"""
    H = im.size[1]
    cands = candidate_separators(im)
    if not cands:
        return [(0, H)]
    cuts = sorted(confirm_separators(im, cands, call_fn))
    if not cuts:
        return [(0, H)]
    bounds = [0] + cuts + [H]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


if __name__ == "__main__":
    import os
    import glob
    from config import TRAIN_TABLE_DIR
    from preprocess import prep
    # 仅看候选（不调 API）：多子表应有候选，单表候选少
    for uuid, truth in [("1ae225a3", 1), ("a300a942", 2), ("f64061da", 6),
                        ("897df5dc", 2), ("0878213b", 1), ("e56db050", 1),
                        ("90a4388e", 2)]:
        f = glob.glob(os.path.join(TRAIN_TABLE_DIR, "images", uuid + "*.jpg"))[0]
        im = prep(Image.open(f))
        c = candidate_separators(im)
        print(f"{uuid}(真值{truth},{'框' if is_bordered(im) else '无框'}): 候选分界={len(c)} y={c[:8]}")
