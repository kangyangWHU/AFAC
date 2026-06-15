# -*- coding: utf-8 -*-
"""TABLE 失败成因诊断：逐张表归类，产出按桶排序的 backlog。

对每张训练表（带 GT）跑现成流水线（API 走缓存），与 GT 比对，按主因归类：
  ok           : TEDS≥0.7
  col_explode  : pred 列 > 1.4×GT（列爆炸）
  col_under    : pred 列 < 0.7×GT（列偏少）
  row_under    : pred 行 < 0.7×GT（行偏少）
  multi_sub    : GT 多子表且未对齐
  cell_misalign: 结构大致对但 TEDS 低（单元格/行错位）
输出 out/table_diagnosis.csv，并打印按桶汇总。
"""
import os
import re
import csv
import glob
from collections import Counter

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from config import TRAIN_TABLE_DIR, OUT_DIR
from preprocess import prep
from slicer_table import slice_table
from stitch_table import stitch_multi
from run_table import _call_grid
from evaluate import table_teds, text_edit_loss


def _mode_col(s, tag):
    rows = re.findall(r"<tr[^>]*>.*?</tr>", s, re.S)
    cc = [len(re.findall(rf"<{tag}", r)) for r in rows]
    return (Counter(cc).most_common(1)[0][0] if cc else 0), len(rows)


def _bordered(im):
    import numpy as np
    arr = np.asarray(im.convert("L"))
    return bool((arr < 128).mean(axis=0).max() > 0.4)


def classify(teds, gc, pc, gr, pr, nsub):
    if teds >= 0.7:
        return "ok"
    if gc and pc > 1.4 * gc:
        return "col_explode"
    if gc and pc < 0.7 * gc:
        return "col_under"
    if gr and pr < 0.7 * gr:
        return "row_under"
    if nsub >= 2:
        return "multi_sub"
    return "cell_misalign"


def main():
    rows = []
    files = sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")))
    for i, md in enumerate(files):
        uuid = os.path.basename(md)[:-3]
        img = glob.glob(os.path.join(TRAIN_TABLE_DIR, "images", uuid + "*.jpg"))
        if not img:
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img[0]))
        tiles, meta = slice_table(im)
        outs = _call_grid(tiles)
        pred = stitch_multi(outs, meta)
        teds = table_teds(pred, gt) or 0.0
        te = text_edit_loss(pred, gt)
        gc, gr = _mode_col(gt, "t[dh]")
        pc, pr = _mode_col(pred, "td")
        nsub = gt.count("<table")
        cat = classify(teds, gc, pc, gr, pr, nsub)
        rows.append({"uuid": uuid[:8], "teds": round(teds, 3),
                     "text": round((1 - te) * 100, 1), "gt_col": gc, "pred_col": pc,
                     "gt_row": gr, "pred_row": pr, "n_sub": nsub,
                     "bordered": _bordered(im), "gt_len": len(gt), "cat": cat})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)}")

    path = os.path.join(OUT_DIR, "table_diagnosis.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 汇总
    cats = Counter(r["cat"] for r in rows)
    teds_by_cat = {}
    for r in rows:
        teds_by_cat.setdefault(r["cat"], []).append(r["teds"])
    print("\n===== 按成因分桶（含均 TEDS、潜在涨幅）=====")
    order = ["multi_sub", "col_explode", "col_under", "row_under",
             "cell_misalign", "ok"]
    for cat in order:
        if cat not in cats:
            continue
        ts = teds_by_cat[cat]
        avg = sum(ts) / len(ts)
        # 修到 0.8 的潜在 TEDS 涨幅（占 100 张表）
        gain = sum(max(0, 0.8 - t) for t in ts) / 100 * 100
        print(f"  {cat:14s}: {cats[cat]:2d} 张  均TEDS={avg:.2f}  "
              f"修到0.8潜在TEDS涨幅≈{gain:.1f}分")
    print(f"\n  当前均 TEDS = {sum(r['teds'] for r in rows)/len(rows)*100:.1f}")
    print(f"  明细: {path}")
    # 打印每桶样例
    print("\n===== 各桶样例（按 TEDS 升序）=====")
    for cat in order:
        if cat == "ok":
            continue
        sub = sorted([r for r in rows if r["cat"] == cat], key=lambda x: x["teds"])
        print(f"\n  [{cat}]")
        for r in sub[:8]:
            print(f"    {r['uuid']} TEDS={r['teds']:.2f} text={r['text']} "
                  f"GT {r['gt_row']}x{r['gt_col']} → PRED {r['pred_row']}x{r['pred_col']} "
                  f"子表={r['n_sub']} {'框' if r['bordered'] else '无框'}")


if __name__ == "__main__":
    main()
