# -*- coding: utf-8 -*-
"""全量 TABLE 评测（新三段式 parse_table = crop→ocr→merge）。

真跑 API 建缓存 + 算 TEDS/textScore，输出 out/teds_new_full.json（按 TEDS 升序）。
与旧 eval_table_full.py(ocr_table 整图) 对照，验证新 Stage I(标题peel/收紧/子表切分)效果。
"""
import os
import re
import glob
import json
import time

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from config import TRAIN_TABLE_DIR, OUT_DIR
from preprocess import prep
from run_table import parse_table
from evaluate import table_teds, text_edit_loss


def _tr_td(s):
    """HTML 行列规模：<tr> 行数、<td>/<th> 单元格数。"""
    return (len(re.findall(r"<tr", s or "", re.I)),
            len(re.findall(r"<t[dh][ >]", s or "", re.I)))


def main():
    files = [f for f in sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                               key=os.path.getsize) if not f.endswith("_pred.md")]
    rows = []
    t_start = time.time()
    for i, md in enumerate(files):
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img):
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img))
        t0 = time.time()
        pred, ncalls, meta = parse_table(im)
        teds = table_teds(pred, gt) or 0.0
        te = text_edit_loss(pred, gt, include_tables=True)
        gtr, gtd = _tr_td(gt)          # GT 行/单元格
        ptr, ptd = _tr_td(pred)        # 预测行/单元格(行列还原准确度)
        rows.append({
            "uuid": uuid, "teds": round(teds, 4),
            "text_score": round((1 - te) * 100, 1),
            "subs": meta.get("subs"), "ncalls": ncalls,
            "gt_tr": gtr, "gt_td": gtd, "pred_tr": ptr, "pred_td": ptd,
            "td_ratio": round(ptd / max(1, gtd), 3),
            "gt_len": len(gt), "pred_len": len(pred),
        })
        print(f"[{i+1:>3}/{len(files)}] {uuid[:8]} TEDS={teds:.4f} "
              f"txt={(1-te)*100:5.1f} tr={ptr}/{gtr} td={ptd}/{gtd} "
              f"subs={meta.get('subs')} calls={ncalls} {time.time()-t0:.0f}s", flush=True)

    rows.sort(key=lambda r: r["teds"])
    mean = sum(r["teds"] for r in rows) / len(rows)
    meantxt = sum(r["text_score"] for r in rows) / len(rows)
    out = {"n": len(rows), "mean_teds": round(mean, 4),
           "mean_text": round(meantxt, 1), "rows": rows}
    path = os.path.join(OUT_DIR, "teds_new_full.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n===== 新路全量 {len(rows)} 张 均值 TEDS={mean:.4f} text={meantxt:.1f} "
          f"总耗时{(time.time()-t_start)/60:.1f}min =====")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
