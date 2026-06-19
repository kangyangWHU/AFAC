# -*- coding: utf-8 -*-
"""全量 TABLE 评测（离线，仅读缓存）→ per-table TEDS + 诊断指标 → JSON。

复现 0.7634：CACHE_ONLY=True，全部走已落盘缓存，不调 API、不随机、可复现。
输出 out/teds_table_full.json（按 TEDS 升序，便于看短板）。
"""
import os
import glob
import json

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import api_client as api
api.CACHE_ONLY = True                       # 离线评测：只读缓存

import config as cfg
cfg.MAX_CONCURRENCY = 16
from config import TRAIN_TABLE_DIR, OUT_DIR
from preprocess import prep
from run_table import run_one
from evaluate import table_teds, text_edit_loss


def _gt_stats(gt):
    """GT 中表格规模：<table> 个数、<tr> 行数、<td>/<th> 单元格数。"""
    import re
    ntab = len(re.findall(r"<table", gt, re.I))
    ntr = len(re.findall(r"<tr", gt, re.I))
    ntd = len(re.findall(r"<t[dh][ >]", gt, re.I))
    return ntab, ntr, ntd


def main():
    files = sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                   key=os.path.getsize)
    rows = []
    for i, md in enumerate(files):
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img):
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img))
        W, H = im.size
        pred, ncalls, meta = run_one(im)
        teds = table_teds(pred, gt)
        teds = teds if teds is not None else 0.0
        te = text_edit_loss(pred, gt, include_tables=True)
        nr = sum(meta["row_cells"])
        nc = sum(meta["col_cells"])
        ntab, ntr, ntd = _gt_stats(gt)
        rows.append({
            "uuid": uuid,
            "teds": round(teds, 4),
            "text_score": round((1 - te) * 100, 1),
            "gt_len": len(gt),
            "pred_len": len(pred),
            "len_ratio": round(len(pred) / max(1, len(gt)), 3),
            "img_w": W, "img_h": H, "aspect": round(W / max(1, H), 2),
            "grid_ok": bool(meta.get("grid")),
            "upsample": meta.get("upsample", 1.0),
            "cell_edge": meta.get("cell_edge"),
            "tile_grid": f"{len(meta['row_cells'])}x{len(meta['col_cells'])}",
            "skeleton": f"{nr}x{nc}", "skel_cells": nr * nc,
            "ncalls": ncalls,
            "split_bands": len(meta.get("split_bands") or []),
            "panel_n": meta.get("panel_n", 0),
            "gt_tables": ntab, "gt_tr": ntr, "gt_td": ntd,
        })
        print(f"[{i+1:>3}/{len(files)}] {uuid[:8]} TEDS={teds:.4f} "
              f"txt={(1-te)*100:5.1f} grid={meta.get('grid')} up={meta.get('upsample')} "
              f"{len(meta['row_cells'])}x{len(meta['col_cells'])}->{nr}x{nc} "
              f"gt(tab={ntab},tr={ntr},td={ntd}) lenR={len(pred)/max(1,len(gt)):.2f}")

    rows.sort(key=lambda r: r["teds"])
    mean = sum(r["teds"] for r in rows) / len(rows)
    out = {"n": len(rows), "mean_teds": round(mean, 4), "rows": rows}
    path = os.path.join(OUT_DIR, "teds_table_full.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n===== 全量 {len(rows)} 张 均值 TEDS = {mean:.4f} =====")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
