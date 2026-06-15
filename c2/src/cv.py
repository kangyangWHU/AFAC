# -*- coding: utf-8 -*-
"""训练集 K 折交叉验证 —— 估算线上 Overall（同分布，带方差）。

本任务无模型训练（"参数"是切块/重组配置），故 K 折的意义是：把 200 张带 GT 的
训练图分成 K 折，逐图跑完整流水线 + 本地三指标，给出 **每折 Overall + 全局 均值±标准差**，
作为线上分的稳健估计（含置信带）。

口径说明（官方对"无表格 LONG"如何计 TEDS 未公开，故同时给两种）：
  - overall_3term：无表样本 TEDS 记 100（偏乐观）
  - overall_2term：无表样本只平均 文本+阅读流（偏保守）
另按 LONG / TABLE 分组报告，便于定位短板。

缓存：API 调用按 tile 内容哈希缓存，**参数不变则命中缓存**，重跑很快。
用法：
  python cv.py --folds 5 --concurrency 32            # 全 200 张
  python cv.py --folds 5 --concurrency 32 --limit 5  # 每类每折各取5，快速冒烟
"""
import os
import json
import glob
import time
import argparse

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import config as cfg
from config import TRAIN_LONG_DIR, TRAIN_TABLE_DIR, OUT_DIR
from evaluate import evaluate_one


def _collect(dir_, kind):
    """收集 (img_path, gt, kind)，按文件名排序保证折分配确定。"""
    items = []
    for md in sorted(glob.glob(os.path.join(dir_, "mds", "*.md"))):
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(dir_, "images", uuid + ".jpg")
        if os.path.exists(img):
            items.append((img, open(md, encoding="utf-8").read(), kind))
    return items


def _fold_of(idx, k):
    return idx % k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="每类每折最多取几张（冒烟用）")
    ap.add_argument("--target_h", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    cfg.MAX_CONCURRENCY = args.concurrency           # 运行时覆盖并发
    from main import process_image                   # 在并发覆盖后再导入

    longs = _collect(TRAIN_LONG_DIR, "long")
    tables = _collect(TRAIN_TABLE_DIR, "table")

    # 分层折分配：long、table 各自 idx%K，保证每折两类均衡
    tagged = []
    for i, it in enumerate(longs):
        tagged.append((_fold_of(i, args.folds), *it))
    for i, it in enumerate(tables):
        tagged.append((_fold_of(i, args.folds), *it))

    if args.limit:                                   # 冒烟：每(折,类)各取 limit 张
        from collections import defaultdict
        cnt = defaultdict(int)
        keep = []
        for fold, img, gt, kind in tagged:
            if cnt[(fold, kind)] < args.limit:
                keep.append((fold, img, gt, kind)); cnt[(fold, kind)] += 1
        tagged = keep

    print(f"共 {len(tagged)} 张, {args.folds} 折, 并发 {args.concurrency}")
    t0 = time.time()
    rows = []
    for n, (fold, img, gt, kind) in enumerate(tagged):
        try:
            pred, _, _ = process_image(img, args.target_h, args.timeout)
            r = evaluate_one(pred, gt)
        except Exception as e:
            print(f"  [{n+1}] {os.path.basename(img)[:8]} 失败: {e}")
            continue
        r.update(fold=fold, kind=kind, name=os.path.basename(img))
        rows.append(r)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(tagged)} 累计{time.time()-t0:.0f}s")

    # ---- 聚合 ----
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def std(vals):
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

    # 每折的 overall（两种口径）
    per_fold_3, per_fold_2 = [], []
    for f in range(args.folds):
        fr = [r for r in rows if r["fold"] == f]
        if not fr:
            continue
        per_fold_3.append(mean([r["overall_3term"] for r in fr]))
        per_fold_2.append(mean([r["overall_2term"] for r in fr]))

    def block(rs, title):
        print(f"\n=== {title} (n={len(rs)}) ===")
        print(f"  text={mean([r['text_score'] for r in rs]):.1f} "
              f"read={mean([r['read_score'] for r in rs]):.1f} "
              f"teds={mean([r['teds_score'] for r in rs])}")
        print(f"  overall_3term={mean([r['overall_3term'] for r in rs]):.1f} "
              f"overall_2term={mean([r['overall_2term'] for r in rs]):.1f}")

    print("\n" + "=" * 50)
    block(rows, "全部")
    block([r for r in rows if r["kind"] == "long"], "LONG")
    block([r for r in rows if r["kind"] == "table"], "TABLE")
    print(f"\n=== {args.folds} 折 Overall ===")
    print(f"  overall_3term 每折: {[round(x,1) for x in per_fold_3]}")
    print(f"    均值±std = {mean(per_fold_3):.1f} ± {std(per_fold_3):.1f}")
    print(f"  overall_2term 每折: {[round(x,1) for x in per_fold_2]}")
    print(f"    均值±std = {mean(per_fold_2):.1f} ± {std(per_fold_2):.1f}")
    print(f"\n总用时 {time.time()-t0:.0f}s")

    out = os.path.join(OUT_DIR, "cv_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "per_fold_3term": per_fold_3,
                   "per_fold_2term": per_fold_2}, f, ensure_ascii=False, indent=2)
    print("明细已存:", out)


if __name__ == "__main__":
    main()
