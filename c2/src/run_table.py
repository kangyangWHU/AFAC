# -*- coding: utf-8 -*-
"""TABLE 端到端 runner + TEDS 评测。

流程：预处理 → 网格切片 → 并发调用 → 2D 重组 → 与 GT 比 TEDS / 文本编辑距离。
所有 API 调用走缓存。
"""
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import api_client as api
from config import TRAIN_TABLE_DIR, API_USER_IDS
from preprocess import prep
from slicer_table import slice_table
from stitch_table import stitch_table, parse_tile, rows_to_html
from evaluate import table_teds, text_edit_loss


def _is_truncated(o):
    """tile 输出被 ~12k 上限截断：有 <table 却无 </table>。"""
    return bool(o) and "<table" in o.lower() and "</table>" not in o.lower()


def _split_call_merge(img, timeout, depth=0):
    """把（截断的）tile 竖切两半各自重读，合并行；半块仍截断则递归（≤2 层）。"""
    w, h = img.size
    top = img.crop((0, 0, w, h // 2))
    bot = img.crop((0, h // 2, w, h))
    ot = api.call_safe(top, timeout=timeout)
    ob = api.call_safe(bot, timeout=timeout)
    if depth < 2 and _is_truncated(ot) and h // 2 > 200:
        ot = _split_call_merge(top, timeout, depth + 1)
    if depth < 2 and _is_truncated(ob) and h // 2 > 200:
        ob = _split_call_merge(bot, timeout, depth + 1)
    merged = parse_tile(ot) + parse_tile(ob)      # 上半行 + 下半行
    return rows_to_html(merged) if merged else (ot or ob)


def _refine_truncated(tiles, outs, timeout=240):
    """对截断的 tile 竖切重读，恢复被截掉的底部行（row_under 主因）。并发处理。"""
    from config import MAX_CONCURRENCY
    todo = [(r, c) for r in range(len(outs)) for c in range(len(outs[r]))
            if tiles[r][c] is not None and _is_truncated(outs[r][c])]
    if not todo:
        return outs
    workers = min(MAX_CONCURRENCY, max(1, len(todo)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fixed = list(ex.map(
            lambda rc: _split_call_merge(tiles[rc[0]][rc[1]], timeout), todo))
    for (r, c), o in zip(todo, fixed):
        outs[r][c] = o
    return outs


def _call_grid(tiles, timeout=240):
    """并发调用 2D tiles，保持 [r][c] 结构。None（空白块）不调 API。"""
    from config import MAX_CONCURRENCY
    flat = [(r, c) for r in range(len(tiles))
            for c in range(len(tiles[r])) if tiles[r][c] is not None]
    workers = min(MAX_CONCURRENCY, max(1, len(flat)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(
            lambda x: api.call_safe(tiles[x[1][0]][x[1][1]], timeout=timeout,
                                    user_id=API_USER_IDS[x[0] % len(API_USER_IDS)]),
            list(enumerate(flat))))
    grid = [[None] * len(tiles[r]) for r in range(len(tiles))]
    for (r, c), o in zip(flat, outs):
        grid[r][c] = o
    return grid


def run_one(im, timeout=240):
    tiles, meta = slice_table(im)
    outs = _call_grid(tiles, timeout)
    # 注：截断块重读(_refine_truncated)实测过慢(分裂×慢API→~3h/全集)，违背预算，已停用。
    pred = stitch_table(outs, meta)
    ncalls = sum(1 for row in tiles for t in row if t is not None)
    return pred, ncalls, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--pick", choices=["median", "small", "spread"],
                    default="median")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                   key=os.path.getsize)
    if args.pick == "median":
        mid = len(files) // 2
        sel = files[mid: mid + args.n]
    elif args.pick == "small":
        sel = files[:args.n]
    else:  # spread
        step = max(1, len(files) // args.n)
        sel = files[::step][:args.n]

    teds_list = []
    for md in sel:
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img):
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img))
        pred, ncalls, meta = run_one(im, args.timeout)
        teds = table_teds(pred, gt)
        te = text_edit_loss(pred, gt, include_tables=True)
        teds_list.append(teds if teds is not None else 0.0)
        nr, nc = len(meta["row_cuts"]) - 1, len(meta["col_cuts"]) - 1
        print(f"[{uuid[:8]}] gt_len={len(gt):>7} 网格={nr}x{nc}={ncalls}块 "
              f"grid={meta['grid']} | TEDS={teds:.4f} textScore={(1-te)*100:.1f} "
              f"pred_len={len(pred)}")

    if teds_list:
        print(f"\n===== 均值 TEDS = {sum(teds_list)/len(teds_list):.4f} "
              f"(×100 = {sum(teds_list)/len(teds_list)*100:.1f}) =====")


if __name__ == "__main__":
    main()
