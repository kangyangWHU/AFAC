# -*- coding: utf-8 -*-
"""端到端入口：图片目录 → submission.csv。

流程：对每张图 分类(LONG/TABLE) → 路由到对应流水线 → 生成 Markdown/HTML。
LONG 走 空白带切+接缝去重；TABLE 走 网格切+空白跳过+2D重组。
图片级串行（每张图内部已 16 路并发调 API），单图异常不影响整批。

用法：
  python main.py --images DIR1 [DIR2 ...] --out submission.csv      # 推理出提交
  python main.py --a_test                                          # 跑 A 榜两目录
  python main.py --train_eval --n 8                                # 训练集自评(带GT)
"""
import os
import csv
import glob
import time
import argparse
import traceback

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from config import (A_LONG_DIR, A_TABLE_DIR, TRAIN_LONG_DIR, TRAIN_TABLE_DIR,
                    OUT_DIR)
from preprocess import prep
from classify import classify
from run_long import run_smart
from run_table import run_one as run_table_one
from heading_norm import roman_to_unicode, subscript_to_latex


def process_image(path, target_h=5000, timeout=240):
    """单图 → (markdown, kind, n_calls)。异常时返回空串。"""
    im = prep(Image.open(path))
    kind = classify(im)
    if kind == "long":
        md, ncalls = run_smart(im, target_h=target_h, timeout=timeout)
    else:
        md, ncalls, _ = run_table_one(im, timeout=timeout)
    # 不转半角:VLM 与 GT 都用中文全角标点(，。：；),转半角反而拉低分。
    # 仅保留 GT 确用的规范:罗马数字→unicode、下标→LaTeX。
    md = subscript_to_latex(roman_to_unicode(md))
    return md, kind, ncalls


def _iter_images(dirs):
    for d in dirs:
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for f in sorted(glob.glob(os.path.join(d, ext))):
                yield f


def run_batch(image_dirs, out_csv, target_h=5000, timeout=240, limit=None):
    """处理一批目录里的图片，写 submission.csv（file_name, ground_truth）。"""
    files = list(_iter_images(image_dirs))
    if limit:
        files = files[:limit]
    print(f"待处理图片 {len(files)} 张 → {out_csv}")

    rows = []
    t0 = time.time()
    for i, path in enumerate(files):
        name = os.path.basename(path)
        try:
            md, kind, ncalls = process_image(path, target_h, timeout)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {name} 失败: {e}")
            traceback.print_exc()
            md, kind, ncalls = "", "err", 0
        rows.append((name, md))
        print(f"  [{i+1}/{len(files)}] {name} kind={kind} calls={ncalls} "
              f"len={len(md)} 累计{time.time()-t0:.0f}s")

    # 写 CSV：UTF-8，全引用，换行/逗号自动转义
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["file_name", "ground_truth"])
        w.writerows(rows)
    print(f"完成：{len(rows)} 行，用时 {time.time()-t0:.0f}s → {out_csv}")
    return rows


def train_eval(n=8, target_h=5000, timeout=240):
    """训练集自评：各取 n 张 LONG/TABLE，跑流水线并与 GT 比三指标。"""
    from evaluate import evaluate_one

    def pick(dir_, n):
        mds = sorted(glob.glob(os.path.join(dir_, "mds", "*.md")),
                     key=os.path.getsize)
        mid = len(mds) // 2
        return mds[mid: mid + n]

    pairs = []
    for tag, d in [("long", TRAIN_LONG_DIR), ("table", TRAIN_TABLE_DIR)]:
        for md in pick(d, n):
            uuid = os.path.basename(md)[:-3]
            img = os.path.join(d, "images", uuid + ".jpg")
            if not os.path.exists(img):
                continue
            gt = open(md, encoding="utf-8").read()
            try:
                pred, kind, _ = process_image(img, target_h, timeout)
            except Exception as e:
                print(f"  {os.path.basename(img)[:8]} 失败: {e}")
                continue
            r = evaluate_one(pred, gt)
            print(f"  [{kind}] {uuid[:8]} overall2={r['overall_2term']:.1f} "
                  f"text={r['text_score']:.1f} read={r['read_score']:.1f} "
                  f"teds={r['teds_score']}")
            pairs.append((tag, r))

    for tag in ("long", "table"):
        sub = [r for t, r in pairs if t == tag]
        if not sub:
            continue
        o2 = sum(r["overall_2term"] for r in sub) / len(sub)
        print(f"\n  === {tag} 均值 overall_2term = {o2:.1f} (n={len(sub)}) ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", help="图片目录（可多个）")
    ap.add_argument("--a_test", action="store_true", help="跑 A 榜两目录")
    ap.add_argument("--train_eval", action="store_true", help="训练集自评(带GT)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "submission.csv"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target_h", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    if args.train_eval:
        train_eval(args.n, args.target_h, args.timeout)
    elif args.a_test:
        dirs = [os.path.join(A_LONG_DIR, "images"),
                os.path.join(A_TABLE_DIR, "images")]
        run_batch(dirs, args.out, args.target_h, args.timeout, args.limit)
    elif args.images:
        run_batch(args.images, args.out, args.target_h, args.timeout, args.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
