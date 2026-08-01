# -*- coding: utf-8 -*-
"""端到端入口：图片目录 → submission.csv（一键复现提交结果）。

流程：每张图按所在目录定 kind(long/table) → 路由到对应流水线 → 生成 Markdown。
LONG 走 空白带切+接缝去重+标题定级+几何定级；TABLE 走 网格切+空白跳过+2D重组+残差重读。
Pool 图级并行,每张图内部再按 MAX_CONCURRENCY 并发调 API;单图异常不影响整批。

一次调用即产出**完整**提交(long 半边 + table 半边合并、按 file_name 全局排序),
不需要任何后续拼接脚本 —— 这是官方复审要求的唯一入口。

用法：
  # 完整提交(推荐,等价于 ../run.sh)
  python main.py --long_dir <B榜long>/images --table_dir <B榜table>/images \
      --out ../out/submission.csv

  # 只跑一半(调试/分半归因用)
  python main.py --long_dir DIR --out ../out/long_only.csv
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

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common.config import OUT_DIR, POOL_PROCS
from common.preprocess import prep
from long.run_long import run_smart
from long.geom_heading import correct as geom_correct   # 几何标题定级(long 路最后一步)
from table.run_table import parse_table as run_table_one   # 三段式:crop → ocr → merge


def _local_ocr():
    """本地逐行识别函数 ocr(pil)->str,供几何定级取标题字号用。

    惰性初始化:rapidocr 引擎在每个工作进程里各建一份(fork 后不共享)。
    """
    from table.cell_ocr import _engine
    eng = _engine()

    def ocr(pil):
        r = eng(np.asarray(pil.convert("RGB")),
                use_det=False, use_cls=False, use_rec=True)
        return r.txts[0] if r and getattr(r, "txts", None) else ""
    return ocr


def process_image(path, kind, target_h=5000, timeout=240):
    """单图 → (markdown, kind, n_calls)。kind 由调用方按目录给定。异常时返回空串。"""
    im = prep(Image.open(path))
    if kind == "long":
        md, ncalls = run_smart(im, target_h=target_h, timeout=timeout)
        # 几何标题定级:回到原图量标题字号与淡横线,重定 # 层级。
        # 必须在 run_smart 之后 —— 前者按条带出文本,层级仍是局部判断;
        # 这一步用整图几何做全局裁决,是 long 路的最后一步。
        md, _ = geom_correct(md, im, _local_ocr(), cache_key=os.path.basename(path))
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


def run_batch(targets, out_csv, target_h=5000, timeout=240, limit=None,
              procs=POOL_PROCS):
    """处理一批 (目录, kind)，写 submission.csv（file_name, ground_truth）。

    行顺序按 file_name 全局排序 —— 与提交文件一致,且不受进程完成顺序影响,
    同一份输入重跑得到同样的行序,便于逐字节比对复现。
    """
    files = list(_iter_images(targets))
    if limit:
        files = files[:limit]
    print(f"待处理图片 {len(files)} 张 → {out_csv}")

    t0 = time.time()
    from multiprocessing import Pool
    from functools import partial
    done = {}
    with Pool(procs) as pool:                    # 图级进程并行(CPU段绕GIL);tile级API
        results = pool.imap_unordered(           # 并发由各流水线内部线程池管(table:ocr_seg / long:call_tiles)
            partial(_run_one, target_h=target_h, timeout=timeout), files)
        for i, (name, md, kind, ncalls) in enumerate(results):
            done[name] = md
            print(f"  [{i+1}/{len(files)}] {name} kind={kind} calls={ncalls} "
                  f"len={len(md)} 累计{time.time()-t0:.0f}s", flush=True)
    rows = [(n, done[n]) for n in sorted(done)]

    empty = [n for n, md in rows if not md.strip()]
    if empty:                                    # 空结果=该图整条流水线失败,提交前必须查
        print(f"[warn] {len(empty)} 份结果为空: {', '.join(empty)}", flush=True)

    # 写 CSV：UTF-8，全引用，换行/逗号自动转义
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["file_name", "ground_truth"])
        w.writerows(rows)
    print(f"完成：{len(rows)} 行，用时 {time.time()-t0:.0f}s → {out_csv}")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="AFAC 赛题二 端到端解析：图片目录 → submission.csv")
    ap.add_argument("--long_dir", help="LONG(面条图)图片目录,如 <数据集>/finix_huge_long_rest_B/images")
    ap.add_argument("--table_dir", help="TABLE(大表图)图片目录,如 <数据集>/finix_huge_table_rest_B/images")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "submission.csv"),
                    help="输出 CSV 路径")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 张(冒烟测试)")
    ap.add_argument("--procs", type=int, default=POOL_PROCS, help="图级并行进程数")
    ap.add_argument("--target_h", type=int, default=5000, help="LONG 切条目标高度(px)")
    ap.add_argument("--timeout", type=int, default=240, help="单图流水线超时(秒)")
    args = ap.parse_args()

    targets = [(d, k) for d, k in ((args.long_dir, "long"),
                                   (args.table_dir, "table")) if d]
    if not targets:
        ap.error("至少给一个 --long_dir / --table_dir")
    for d, _ in targets:
        if not os.path.isdir(d):
            ap.error(f"目录不存在: {d}")

    run_batch(targets, args.out, args.target_h, args.timeout, args.limit,
              args.procs)


if __name__ == "__main__":
    main()
