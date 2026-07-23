# -*- coding: utf-8 -*-
"""行列估计诊断(纯几何,不调 API)。

对每张图: crop 出 seg → slice_grid 骨架(与管线同源)→
① 画骨架到 <out>/<uid>.jpg: 黄框=seg、绿线=行边界、红线=列边界、
  粗蓝线=tile 切分带、misaligned 段标注文字
② 有 GT(mds)时对比 tr/td 算准确度(默认训练集;--images 指定测试集则跳过)。
"""
import os
import re
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None

from common.config import TRAIN_TABLE_DIR, OUT_DIR
from common.preprocess import prep
from table.crop import crop
from table.grid_ocr import slice_grid


def gt_tables(gt):
    """解析 GT 每个 <table> 的 (行数, 列数)。列数=各行 **colspan 求和** 的最大值——
    数原始 td 标签会把 <td colspan=46> 当 1 列(d9a99684 表头行 4+colspan46 数成 51,
    真实网格宽 50)。"""
    out = []
    for tb in re.findall(r"<table.*?</table>", gt, re.S | re.I):
        # 按 <tr 开标签切行——GT 存在未闭合 <tr>(d9a99684 表头行),要求 </tr> 会把
        # 两行并一行:行数少 1、colspan 求和翻倍(4+46+46=96)
        trs = re.split(r"<tr[^>]*>", tb, flags=re.I)[1:]
        if not trs:
            continue
        cols = 0
        for tr in trs:
            w = 0
            for td in re.findall(r"<t[dh][^>]*>", tr, re.I):
                m = re.search(r'colspan\s*=\s*"?(\d+)', td, re.I)
                w += int(m.group(1)) if m else 1
            cols = max(cols, w)
        out.append((len(trs), cols))
    return out


def draw_seg(dr, im, bb, lw):
    """单 seg 骨架绘制,返回 (est_rows, est_cols)。"""
    x0, y0, x1, y1 = bb
    _, meta = slice_grid(im.crop(bb))
    dr.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 200, 0), width=3 * lw)
    if meta["misaligned"]:
        dr.text((x0 + 8, y0 + 8), "misaligned", fill=(255, 0, 0))
        return meta["rows"], meta["cols"]
    rb, cb = meta["rb"], meta["cb"]
    for y in rb:                       # 行线(绿)
        dr.line([(x0, y0 + y), (x1, y0 + y)], fill=(0, 180, 0), width=lw)
    for x in cb:                       # 列线(红)
        dr.line([(x0 + x, y0), (x0 + x, y1)], fill=(255, 0, 0), width=lw)
    for (ri, rj) in meta["row_bands"]:      # tile 带(粗蓝)
        for y in (rb[ri], rb[rj]):
            dr.line([(x0, y0 + y), (x1, y0 + y)], fill=(0, 90, 255), width=3 * lw)
    for (ci, cj) in meta["col_bands"]:
        for x in (cb[ci], cb[cj]):
            dr.line([(x0 + x, y0), (x0 + x, y1)], fill=(0, 90, 255), width=3 * lw)
    return meta["rows"], meta["cols"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None,
                    help="图片目录(默认训练集 images+mds 并算 GT 准确度)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "grids"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.images:                     # 测试集模式:无 GT,只画
        pairs = [(p, None) for p in sorted(glob.glob(os.path.join(args.images, "*.jpg")))]
    else:
        pairs = []
        for md in [f for f in sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                                     key=os.path.getsize) if not f.endswith("_pred.md")]:
            img = os.path.join(TRAIN_TABLE_DIR, "images",
                               os.path.basename(md)[:-3] + ".jpg")
            if os.path.exists(img):
                pairs.append((img, md))

    rows = []
    for img, md in pairs:
        uuid = os.path.splitext(os.path.basename(img))[0]
        im = prep(Image.open(img))
        segs = [bb for k, bb in crop(im) if k not in ("text", "title")]
        ov = im.convert("RGB")
        dr = ImageDraw.Draw(ov)
        lw = max(2, ov.width // 1500)   # 线宽随图放大,避免大图线太细
        est = [draw_seg(dr, im, bb, lw) for bb in segs]
        # 测试集模式加 _grid 后缀,与 grids_B 里的原图/_pred 并存不冲突
        fname = uuid + ("_grid.jpg" if args.images else ".jpg")
        ov.save(os.path.join(args.out, fname), quality=92)
        if md is None:
            print(f"{uuid[:8]} seg={len(est)} est={est}", flush=True)
            continue
        gt = open(md, encoding="utf-8").read()
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
        print(f"{uuid[:8]} seg={len(est)}/gt_tab={len(gts)} est={est} gt={gts} "
              f"cells {est_cells}/{gt_cells} ratio={est_cells/max(1,gt_cells):.2f}", flush=True)

    if not rows:                        # 测试集模式:无 GT,只画图
        print(f"行列线图: {args.out}/")
        return
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
    print(f"行列线图: {args.out}/  明细: {OUT_DIR}/grid_acc.json")


if __name__ == "__main__":
    main()
