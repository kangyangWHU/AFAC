# -*- coding: utf-8 -*-
"""端到端入口：图片目录 → submission.csv。

流程：对每张图 分类(LONG/TABLE) → 路由到对应流水线 → 生成 Markdown/HTML。
LONG 走 空白带切+接缝去重；TABLE 走 网格切+空白跳过+2D重组。
Pool 图级并行,每张图内部再按 MAX_CONCURRENCY 并发调 API;单图异常不影响整批。

用法：
  python main.py --images DIR1 [DIR2 ...] --out submission.csv      # 推理出提交
  python main.py --a_test                                          # 跑 A 榜两目录
"""
import os
import csv
import glob
import time
import argparse
import traceback

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common.config import A_LONG_DIR, A_TABLE_DIR, OUT_DIR, RUN_TARGETS
from common.preprocess import prep
from long.run_long import run_smart
from table.run_table import parse_table as run_table_one   # 三段式:crop → ocr → merge


def process_image(path, kind, target_h=5000, timeout=240):
    """单图 → (markdown, kind, n_calls)。kind 由调用方按目录给定。异常时返回空串。"""
    im = prep(Image.open(path))
    if kind == "long":
        md, ncalls = run_smart(im, target_h=target_h, timeout=timeout)
    else:
        md, ncalls, _ = run_table_one(im, timeout=timeout)
    return md, kind, ncalls


def _iter_images(targets):
    """targets: [(目录, kind)] → 依次产出 (图片路径, kind)。"""
    for d, kind in targets:
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for f in sorted(glob.glob(os.path.join(d, ext))):
                yield f, kind


def _run_one(item, target_h, timeout):
    path, kind = item
    name = os.path.basename(path)
    try:
        md, kind, ncalls = process_image(path, kind, target_h, timeout)
    except Exception as e:
        print(f"  {name} 失败: {e}", flush=True)
        traceback.print_exc()
        md, kind, ncalls = "", "err", 0
    return name, md, kind, ncalls


def run_batch(targets, out_csv, target_h=5000, timeout=240, limit=None):
    """处理一批 (目录, kind)，写 submission.csv（file_name, ground_truth）。"""
    files = list(_iter_images(targets))
    if limit:
        files = files[:limit]
    print(f"待处理图片 {len(files)} 张 → {out_csv}")

    t0 = time.time()
    from multiprocessing import Pool
    from functools import partial
    done = {}
    with Pool(6) as pool:                        # 图级进程并行(CPU段绕GIL);tile级API
        results = pool.imap_unordered(           # 并发由各流水线内部线程池管(table:ocr_seg / long:call_tiles)
            partial(_run_one, target_h=target_h, timeout=timeout), files)
        for i, (name, md, kind, ncalls) in enumerate(results):
            done[name] = md
            print(f"  [{i+1}/{len(files)}] {name} kind={kind} calls={ncalls} "
                  f"len={len(md)} 累计{time.time()-t0:.0f}s", flush=True)
    rows = [(os.path.basename(p), done.get(os.path.basename(p), ""))
            for p, _ in files]

    # 写 CSV：UTF-8，全引用，换行/逗号自动转义
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["file_name", "ground_truth"])
        w.writerows(rows)
    print(f"完成：{len(rows)} 行，用时 {time.time()-t0:.0f}s → {out_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", help="图片目录（可多个，需配 --kind）")
    ap.add_argument("--kind", choices=["long", "table"], help="--images 的类别")
    ap.add_argument("--a_test", action="store_true", help="跑 A 榜两目录")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "submission.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target_h", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    if args.a_test:
        targets = [(os.path.join(A_LONG_DIR, "images"), "long"),
                   (os.path.join(A_TABLE_DIR, "images"), "table")]
        run_batch(targets, args.out, args.target_h, args.timeout, args.limit)
    elif args.images:
        if not args.kind:
            ap.error("--images 需配 --kind long|table")
        targets = [(d, args.kind) for d in args.images]
        run_batch(targets, args.out, args.target_h, args.timeout, args.limit)
    elif RUN_TARGETS:
        run_batch(RUN_TARGETS, args.out, args.target_h, args.timeout, args.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
