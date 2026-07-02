# -*- coding: utf-8 -*-
"""行列估计准确度诊断(纯几何,不调 API)。

对每张图: crop 出 seg → 用**我们自己的几何估计**(geom.column_cuts 列 + 对称行检测,
即 Stage I 那套,非 slice_table 二次加工)算骨架 row_bnd × col_bnd →
① 画出估计的行列线到 out/grids/<uid>.jpg ② 对比 GT 的 tr/td 算准确度。
"""
import os
import re
import glob
import json

import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from config import TRAIN_TABLE_DIR, OUT_DIR, BIN_INK, BIN_FAINT
from preprocess import prep
from crop import crop
from geom import column_cuts, row_bnds, _boundaries


def grid_lines(sub):
    """我们的几何行列估计(与 Stage I 同源): column_cuts 定列线、row_bnds 定行边界。
    返回 (row_bnd, col_bnd) 局部像素边界。"""
    g = np.asarray(sub.convert("L"))
    W = g.shape[1]
    dark, dark180 = g < BIN_INK, g < BIN_FAINT
    col_lines, _ = column_cuts(dark, dark180)
    rb, _framed = row_bnds(dark, dark180)
    return rb, _boundaries(col_lines, W)


def gt_tables(gt):
    """解析 GT 每个 <table> 的 (行数, 列数)。列数=某行最多 td 数。"""
    out = []
    for tb in re.findall(r"<table.*?</table>", gt, re.S | re.I):
        trs = re.findall(r"<tr.*?</tr>", tb, re.S | re.I)
        if not trs:
            continue
        cols = max(len(re.findall(r"<t[dh][ >]", tr, re.I)) for tr in trs)
        out.append((len(trs), cols))
    return out


def main():
    os.makedirs(os.path.join(OUT_DIR, "grids"), exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                               key=os.path.getsize) if not f.endswith("_pred.md")]
    rows = []
    for md in files:
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img):
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img))
        segs = [bb for k, bb in crop(im) if k == "seg"]
        ov = im.convert("RGB")
        dr = ImageDraw.Draw(ov)
        lw = max(2, ov.width // 1500)   # 线宽随图放大,避免大图线太细
        est = []                       # 每 seg (rows, cols)
        for (x0, y0, x1, y1) in segs:
            rb, cb = grid_lines(im.crop((x0, y0, x1, y1)))
            est.append((len(rb) - 1, len(cb) - 1))
            for y in rb:               # 行线(蓝)
                dr.line([(x0, y0 + y), (x1, y0 + y)], fill=(0, 90, 255), width=lw)
            for x in cb:               # 列线(红)
                dr.line([(x0 + x, y0), (x0 + x, y1)], fill=(255, 0, 0), width=lw)
        gts = gt_tables(gt)
        est_cells = sum(r * c for r, c in est)
        gt_cells = sum(r * c for r, c in gts)
        gt_td = len(re.findall(r"<t[dh][ >]", gt, re.I))
        # 表数匹配时逐表算行/列命中
        row_ok = col_ok = 0
        if len(est) == len(gts):
            for (er, ec), (gr, gc) in zip(sorted(est), sorted(gts)):
                row_ok += (er == gr); col_ok += (ec == gc)
        rows.append({
            "uuid": uuid, "n_seg": len(est), "n_gt_tab": len(gts),
            "est": est, "gt": gts,
            "est_cells": est_cells, "gt_cells": gt_cells, "gt_td": gt_td,
            "cell_ratio": round(est_cells / max(1, gt_cells), 3),
            "tab_match": len(est) == len(gts),
            "row_ok": row_ok, "col_ok": col_ok,
        })
        ov.save(os.path.join(OUT_DIR, "grids", uuid + ".jpg"), quality=92)  # 原分辨率,清晰
        print(f"{uuid[:8]} seg={len(est)}/gt_tab={len(gts)} est={est} gt={gts} "
              f"cells {est_cells}/{gt_cells} ratio={est_cells/max(1,gt_cells):.2f}", flush=True)

    tabm = [r for r in rows if r["tab_match"]]
    tot_row = sum(len(r["est"]) for r in tabm)
    row_hit = sum(r["row_ok"] for r in tabm)
    col_hit = sum(r["col_ok"] for r in tabm)
    ratios = [r["cell_ratio"] for r in rows]
    summary = {
        "n": len(rows),
        "tab_match_imgs": len(tabm),
        "row_acc": round(row_hit / max(1, tot_row), 3),
        "col_acc": round(col_hit / max(1, tot_row), 3),
        "cell_ratio_median": round(sorted(ratios)[len(ratios) // 2], 3),
        "cell_ratio_mean": round(sum(ratios) / len(ratios), 3),
    }
    json.dump({"summary": summary, "rows": rows},
              open(os.path.join(OUT_DIR, "grid_acc.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== 行列估计准确度 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"行列线图: {OUT_DIR}/grids/  明细: {OUT_DIR}/grid_acc.json")


if __name__ == "__main__":
    main()
