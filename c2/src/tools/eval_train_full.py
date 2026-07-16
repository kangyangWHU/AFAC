# -*- coding: utf-8 -*-
"""全量训练集评测：用最新 process_image 跑 LONG+TABLE 各 100 张，出三指标。

训练集有 GT 无缓存 → 真跑 API 建缓存。Pool 图级并行(绕 GIL)。
输出 out/train_full.json（按 overall_2term 升序）+ 打印 long/table 均值。
用法(在 src 目录)： python -m tools.eval_train_full
"""
import os
import glob
import json
from multiprocessing import Pool
from functools import partial

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common.config import TRAIN_LONG_DIR, TRAIN_TABLE_DIR, OUT_DIR
from main import process_image
from metrics.evaluate import evaluate_one


def _one(item, timeout):
    kind, img, md = item
    uuid = os.path.basename(img)[:-4]
    gt = open(md, encoding="utf-8").read()
    try:
        pred, _, ncalls = process_image(img, kind, timeout=timeout)
    except Exception as e:
        return {"kind": kind, "uuid": uuid, "err": str(e)}
    r = evaluate_one(pred, gt)
    return {"kind": kind, "uuid": uuid, "ncalls": ncalls,
            "gt_len": len(gt), "pred_len": len(pred), **r}


def main(timeout=240, workers=6):
    tasks = []
    for kind, d in [("long", TRAIN_LONG_DIR), ("table", TRAIN_TABLE_DIR)]:
        for md in sorted(glob.glob(os.path.join(d, "mds", "*.md"))):
            if md.endswith("_pred.md"):          # mds/ 里混有预测残留,不是 GT
                continue
            uuid = os.path.basename(md)[:-3]
            img = os.path.join(d, "images", uuid + ".jpg")
            if os.path.exists(img):
                tasks.append((kind, img, md))
    print(f"全量训练集 {len(tasks)} 张 → out/train_full.json", flush=True)

    rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(partial(_one, timeout=timeout), tasks)):
            if "err" in r:
                print(f"  [{i+1}/{len(tasks)}] {r['kind']} {r['uuid'][:8]} 失败: {r['err']}", flush=True)
                continue
            rows.append(r)
            teds = f"{r['teds_score']:.1f}" if r["teds_score"] is not None else "—"
            print(f"  [{i+1}/{len(tasks)}] {r['kind']:5} {r['uuid'][:8]} "
                  f"overall2={r['overall_2term']:5.1f} text={r['text_score']:5.1f} "
                  f"read={r['read_score']:5.1f} teds={teds}", flush=True)

    rows.sort(key=lambda x: x["overall_2term"])
    means = {}
    for kind in ("long", "table"):
        sub = [x for x in rows if x["kind"] == kind]
        if sub:
            means[kind] = {
                "n": len(sub),
                "overall_2term": round(sum(x["overall_2term"] for x in sub) / len(sub), 2),
                "text_score": round(sum(x["text_score"] for x in sub) / len(sub), 2),
                "read_score": round(sum(x["read_score"] for x in sub) / len(sub), 2),
                "teds_score": round(sum(x["teds_score"] for x in sub if x["teds_score"] is not None)
                                    / max(1, sum(1 for x in sub if x["teds_score"] is not None)), 2),
            }
    path = os.path.join(OUT_DIR, "train_full.json")
    json.dump({"means": means, "rows": rows}, open(path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== 均值 =====")
    for kind, m in means.items():
        print(f"  {kind}: overall2={m['overall_2term']} text={m['text_score']} "
              f"read={m['read_score']} teds={m['teds_score']} (n={m['n']})")
    print(f"已保存 {path}")


if __name__ == "__main__":
    main()
