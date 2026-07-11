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

from common.config import TRAIN_TABLE_DIR, OUT_DIR
from common.preprocess import prep
from table.run_table import parse_table
from metrics.evaluate import table_teds, text_edit_loss


def _tr_td(s):
    """HTML 行列规模：<tr> 行数、<td>/<th> 单元格数。"""
    return (len(re.findall(r"<tr", s or "", re.I)),
            len(re.findall(r"<t[dh][ >]", s or "", re.I)))


def main():
    files = [f for f in sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                               key=os.path.getsize) if not f.endswith("_pred.md")]
    rows = []
    t_start = time.time()
    from multiprocessing import Pool
    with Pool(12) as pool:                       # CPU批处理进程池(TEDS/编辑距离/PIL全是
        results = pool.imap_unordered(_eval_one, files)   # CPU,GIL下线程池会串行)
        for i, r in enumerate(results):
            if r is None:
                continue
            rows.append(r)
            print(f"[{len(rows):3}/{len(files)}] {r['uuid'][:8]} TEDS={r['teds']:.4f} "
                  f"txt={r['text_score']:.1f} tr={r['pred_tr']}/{r['gt_tr']} "
                  f"td={r['pred_td']}/{r['gt_td']} calls={r['ncalls']} {r['secs']:.0f}s",
                  flush=True)
    _summary(rows, t_start)


def _eval_one(md):
    uuid = os.path.basename(md)[:-3]
    img = os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg")
    if not os.path.exists(img):
        return None
    gt = open(md, encoding="utf-8").read()
    im = prep(Image.open(img))
    t0 = time.time()
    try:
        pred, ncalls, meta = parse_table(im)
        teds = table_teds(pred, gt) or 0.0
        te = text_edit_loss(pred, gt, include_tables=True)
        gtr, gtd = _tr_td(gt)          # GT 行/单元格
        ptr, ptd = _tr_td(pred)        # 预测行/单元格(行列还原准确度)
        return {
            "uuid": uuid, "teds": round(teds, 4),
            "text_score": round((1 - te) * 100, 1),
            "subs": meta.get("subs"), "ncalls": ncalls,
            "gt_tr": gtr, "gt_td": gtd, "pred_tr": ptr, "pred_td": ptd,
            "td_ratio": round(ptd / max(1, gtd), 3),
            "gt_len": len(gt), "pred_len": len(pred),
            "secs": time.time() - t0,
        }
    except Exception as e:
        print(f"  {uuid[:8]} 失败: {e}", flush=True)
        return {"uuid": uuid, "teds": 0.0, "text_score": 0.0, "subs": 0, "ncalls": 0,
                "gt_tr": 0, "gt_td": 0, "pred_tr": 0, "pred_td": 0,
                "td_ratio": 0, "gt_len": len(gt), "pred_len": 0, "secs": time.time() - t0}


def _summary(rows, t_start):
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
