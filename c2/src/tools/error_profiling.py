# -*- coding: utf-8 -*-
"""精度/错误画像：用带 GT 的训练图，实测「精度 vs 切块尺寸 vs 延迟」权衡曲线。

两条核心实验：
  A) TABLE：整图降采样(1块) vs 行带分块(K块) → 比 TEDS。
     回答：密集表最少切几块能保住精度？是否 ≤11（3h 预算上限）？
  B) LONG：不同条高(非重叠裸拼) → 比文本编辑距离 / 阅读流。
     回答：精度与延迟平衡的条高是多少？

附带：把 API 输出与 GT 逐项对比，给出错误类型直觉（漏/重/数字/层级）。

所有 API 调用走缓存（api_client），重复运行不再花钱/花时间。
结果存 out/profiling_*.json。
"""
import os
import re
import json
import time
import glob
import argparse

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import common.api_client as api
from common.config import TRAIN_TABLE_DIR, TRAIN_LONG_DIR, OUT_DIR
from metrics.evaluate import (text_edit_loss, table_teds, read_order_loss,
                      normalize_text)

_TR_RE = re.compile(r"<tr[^>]*>.*?</tr>", re.I | re.S)
_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.I | re.S)


# ---------------------------------------------------------------------------
# 工具：表格 HTML 合并（行带分块后把各块的 <tr> 顺序拼成一个 <table>）
# ---------------------------------------------------------------------------
def merge_table_bands(htmls):
    """从多个分块输出里按顺序收集所有 <tr>，重组为单个 <table>。"""
    trs = []
    for h in htmls:
        # 只取 <table> 内部，避免块里夹带的说明文字
        m = _TABLE_RE.search(h or "")
        body = m.group(0) if m else (h or "")
        trs.extend(_TR_RE.findall(body))
    return "<table>\n" + "\n".join(trs) + "\n</table>"


def _row_bands(im, k, overlap=0):
    """把图按高度等分成 k 个全宽行带（可选重叠像素）。返回 PIL 列表。"""
    w, h = im.size
    band = h // k
    out = []
    for i in range(k):
        top = max(0, i * band - overlap)
        bot = min(h, (i + 1) * band + overlap) if i < k - 1 else h
        out.append(im.crop((0, top, w, bot)))
    return out


def _call_imgs(imgs, timeout=240):
    """并发调用一组切片（5 userId 轮询，保序，走缓存）。
    返回 (outputs, wall_latency)。wall_latency 为并发墙钟，贴近实际复现耗时。
    """
    from concurrent.futures import ThreadPoolExecutor
    from common.config import API_USER_IDS
    t0 = time.time()
    workers = min(16, max(1, len(imgs)))          # 实测 16 路并发无限流
    with ThreadPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(
            lambda x: api.call(x[1], timeout=timeout,
                               user_id=API_USER_IDS[x[0] % len(API_USER_IDS)]),
            list(enumerate(imgs))))
    return outs, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------
# 实验 A：TABLE 切块数 × TEDS
# ---------------------------------------------------------------------------
def profile_table(n_imgs=3, bands=(4, 8, 12, 16), timeout=240):
    """对若干典型表，比较 整图 vs 行带分块 的 TEDS。"""
    # 选“中位密度”附近的表（避开 639k 巨表极端值），更代表多数情况
    files = glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md"))
    files = sorted(files, key=lambda f: os.path.getsize(f))
    mid = len(files) // 2
    files = files[mid: mid + n_imgs]

    results = []
    for md_path in files:
        uuid = os.path.splitext(os.path.basename(md_path))[0]
        img_path = os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt = open(md_path, encoding="utf-8").read()
        im = Image.open(img_path).convert("RGB")
        rec = {"uuid": uuid[:8], "img_size": list(im.size),
               "gt_len": len(gt), "by_bands": {}}
        print(f"\n[TABLE] {uuid[:8]} size={im.size} gt_len={len(gt)}")
        for k in bands:
            imgs = _row_bands(im, k) if k > 1 else [im]
            outs, lat = _call_imgs(imgs, timeout)
            pred = merge_table_bands(outs) if k > 1 else outs[0]
            teds = table_teds(pred, gt)
            te = text_edit_loss(pred, gt, include_tables=True)
            rec["by_bands"][k] = {
                "teds": round(teds, 4) if teds is not None else None,
                "text_edit": round(te, 4),
                "n_calls": k, "latency_s": lat,
                "pred_len": len(pred),
            }
            print(f"   bands={k:<2d} calls={k:<2d} lat={lat:>6}s "
                  f"TEDS={rec['by_bands'][k]['teds']} "
                  f"textEdit={rec['by_bands'][k]['text_edit']} "
                  f"pred_len={len(pred)}")
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# 实验 B：LONG 条高 × 文本/阅读流
# ---------------------------------------------------------------------------
def profile_long(n_imgs=2, heights=(3000, 5000, 8000), timeout=240):
    """对若干面条图，比较不同条高(非重叠裸拼)的文本编辑距离与阅读流。
    注：此处用裸拼（无去重），用于选条高；真正的接缝去重在 stitch_long 实现。
    """
    files = glob.glob(os.path.join(TRAIN_LONG_DIR, "mds", "*.md"))
    # 选中等长度，避免最长的拖慢实验
    files = sorted(files, key=lambda f: os.path.getsize(f))
    files = files[len(files) // 2: len(files) // 2 + n_imgs]

    results = []
    for md_path in files:
        uuid = os.path.splitext(os.path.basename(md_path))[0]
        img_path = os.path.join(TRAIN_LONG_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt = open(md_path, encoding="utf-8").read()
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        rec = {"uuid": uuid[:8], "img_size": [w, h],
               "gt_len": len(gt), "by_height": {}}
        print(f"\n[LONG] {uuid[:8]} size={im.size} gt_len={len(gt)}")
        for H in heights:
            k = max(1, -(-h // H))                 # ceil 切条数
            imgs = [im.crop((0, i * H, w, min(h, (i + 1) * H))) for i in range(k)]
            outs, lat = _call_imgs(imgs, timeout)
            pred = "\n".join(outs)                  # 裸拼
            te = text_edit_loss(pred, gt, include_tables=True)
            ro = read_order_loss(pred, gt)
            rec["by_height"][H] = {
                "n_calls": k, "latency_s": lat,
                "text_edit": round(te, 4), "read_order": round(ro, 4),
                "text_score": round((1 - te) * 100, 2),
                "read_score": round((1 - ro) * 100, 2),
                "pred_len": len(pred),
            }
            print(f"   H={H:<5d} calls={k:<2d} lat={lat:>6}s "
                  f"textScore={rec['by_height'][H]['text_score']} "
                  f"readScore={rec['by_height'][H]['read_score']} "
                  f"pred_len={len(pred)} (gt {len(gt)})")
        results.append(rec)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["table", "long", "both"], default="both")
    ap.add_argument("--n", type=int, default=2, help="每类取几张图")
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    out = {}
    if args.mode in ("table", "both"):
        out["table"] = profile_table(n_imgs=args.n, timeout=args.timeout)
    if args.mode in ("long", "both"):
        out["long"] = profile_long(n_imgs=args.n, timeout=args.timeout)

    path = os.path.join(OUT_DIR, "profiling_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已保存：", path)


if __name__ == "__main__":
    main()
