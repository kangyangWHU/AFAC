# -*- coding: utf-8 -*-
"""LONG 端到端 runner + 训练集评测。

run_smart = 行间空白带切 + 接缝去重(本方案)。用训练集(带 GT)跑若干图,
打印 text/read 指标。所有 API 调用走缓存。
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
from long.table_fix import fix_tables, split_tables
from long.heading_norm import (relevel_strips, toc_bullets_to_headings,
                               infer_missing_headings, flatten_math,
                               enforce_enum_consistency,
                               flatten_scripts,
                               roman_to_unicode)
from metrics.evaluate import text_edit_loss, read_order_loss


def run_smart(im, target_h=5000, timeout=240):
    strips, _ = slice_long(im, target_h=target_h)
    outs = call_tiles(strips, timeout=timeout, retry_rounds=4)
    outs = [toc_bullets_to_headings(o) for o in outs]   # 目录列表项→标题(在定级前)
    outs = relevel_strips(outs)
    md = merge_strips(outs)
    md = split_tables(fix_tables(md))       # 横幅行 colspan 归一 → 再据此切开子表
    md = infer_missing_headings(md)         # 编号序列空洞→补回漏标标题
    md = enforce_enum_consistency(md)       # 编号序列多数派表决:全标题或全正文(一致性优先)
    # GT 规范:罗马数字→unicode;上/下标一律压平成普通字符,不产出 LaTeX。
    # GT 全量 200 篇里 unicode 上下标出现 0 次(×10⁹/L 写成 "109/L"),故压平只会更贴近。
    # 注:本地 text_edit 走 NFKC,看不出压平的差别(⁹ 与 9 等价),线上若不归一化才有收益。
    md = flatten_scripts(roman_to_unicode(md))
    md = flatten_math(md)                   # API 的 $$\text{}$$ 公式 → GT 风格纯文本($$ 在 GT 出现 0 次)
    return md, len(strips)


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

    agg = {"text": [], "read": []}
    for md in files:
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(TRAIN_LONG_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img):
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img))
        ps, ks = run_smart(im, args.target_h, args.timeout)
        ss = _score(ps, gt)
        agg["text"].append(ss["text"]); agg["read"].append(ss["read"])
        print(f"[{uuid[:8]}] gt_len={len(gt)} smart(k={ks}): "
              f"text={ss['text']} read={ss['read']} len={ss['len']}")

    def avg(v):
        return round(sum(v) / len(v), 2) if v else None
    print(f"\n均值: text={avg(agg['text'])}  read={avg(agg['read'])}")


if __name__ == "__main__":
    main()
