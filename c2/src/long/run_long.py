# -*- coding: utf-8 -*-
"""LONG 端到端 runner + A/B 评测。

对比两种策略，量化智能切点的收益：
  - naive : 等高裸切 + 直接拼（对照）
  - smart : 行间空白带切 + 接缝去重（本方案）

用训练集（带 GT）跑若干图，打印三指标。所有 API 调用走缓存。
"""
import os
import glob
import argparse

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common.config import TRAIN_LONG_DIR
from common.preprocess import prep
from table.tiles import call_tiles
from long.slicer_long import slice_long
from long.stitch_long import merge_strips
from long.heading_norm import (relevel_strips, toc_bullets_to_headings,
                               roman_to_unicode, subscript_to_latex)
from metrics.evaluate import text_edit_loss, read_order_loss


def run_smart(im, target_h=5000, timeout=240):
    strips, _ = slice_long(im, target_h=target_h)
    outs = call_tiles(strips, timeout=timeout, retry_rounds=4)
    outs = [toc_bullets_to_headings(o) for o in outs]   # 目录列表项→标题(在定级前)
    outs = relevel_strips(outs)
    md = merge_strips(outs)
    # GT 规范:罗马数字→unicode、下标→LaTeX(仅 LONG 需要;TABLE 无此内容)
    md = subscript_to_latex(roman_to_unicode(md))
    return md, len(strips)


def run_naive(im, target_h=5000, timeout=240):
    """等高裸切 + 裸拼（对照）。"""
    w, h = im.size
    k = max(1, -(-h // target_h))
    strips = [im.crop((0, i * target_h, w, min(h, (i + 1) * target_h)))
              for i in range(k)]
    outs = call_tiles(strips, timeout=timeout, retry_rounds=4)
    return "\n".join(outs), k


def _score(pred, gt):
    te = text_edit_loss(pred, gt, include_tables=True)
    ro = read_order_loss(pred, gt)
    return {"text": round((1 - te) * 100, 2),
            "read": round((1 - ro) * 100, 2),
            "len": len(pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--target_h", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(TRAIN_LONG_DIR, "mds", "*.md")),
                   key=os.path.getsize)
    # 取中段若干，避开极端
    mid = len(files) // 2
    files = files[mid: mid + args.n]

    agg = {"naive": {"text": [], "read": []},
           "smart": {"text": [], "read": []}}
    for md in files:
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(TRAIN_LONG_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img):
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img))

        pn, kn = run_naive(im, args.target_h, args.timeout)
        ps, ks = run_smart(im, args.target_h, args.timeout)
        sn, ss = _score(pn, gt), _score(ps, gt)
        agg["naive"]["text"].append(sn["text"]); agg["naive"]["read"].append(sn["read"])
        agg["smart"]["text"].append(ss["text"]); agg["smart"]["read"].append(ss["read"])
        print(f"[{uuid[:8]}] gt_len={len(gt)}")
        print(f"   naive(k={kn}): text={sn['text']} read={sn['read']} len={sn['len']}")
        print(f"   smart(k={ks}): text={ss['text']} read={ss['read']} len={ss['len']}")

    def avg(v):
        return round(sum(v) / len(v), 2) if v else None
    print("\n===== 均值 =====")
    for m in ("naive", "smart"):
        print(f"  {m}: text={avg(agg[m]['text'])}  read={avg(agg[m]['read'])}")


if __name__ == "__main__":
    main()
