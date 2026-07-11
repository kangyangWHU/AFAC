# -*- coding: utf-8 -*-
"""标定 FinixDoc-VL 的尺寸/延迟边界，为切块参数提供依据。

回答两个问题：
  1) 单次能吃多大的图？（多大开始超时/报错/输出明显退化）
  2) 延迟随尺寸怎么变？（决定 3h 预算内的切块数量与并发策略）

分两类几何：
  - TABLE 几何：近正方形，按最长边 ladder 等比缩放后整图送测；
  - LONG 几何：宽固定 1500，按高度 ladder 切条送测。

每次调用 ~30s+，故用 5 个 userId 线程并发跑 ladder。结果存 out/calibrate_*.json。
"""
import os
import json
import time
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import common.api_client as api
from common.config import TRAIN_TABLE_DIR, TRAIN_LONG_DIR, OUT_DIR, API_USER_IDS


def _timed_call(pil_img, uid, timeout):
    """单次受控调用，返回测量结果（不抛异常）。"""
    t0 = time.time()
    rec = {"status": "", "latency_s": 0.0, "out_len": 0,
           "out_head": "", "error": ""}
    try:
        md = api.call(pil_img, timeout=timeout, retries=1,
                      use_cache=False, user_id=uid)
        rec["status"] = "ok"
        rec["out_len"] = len(md)
        rec["out_head"] = md[:120]
    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = ("%s: %s" % (type(e).__name__, e))[:200]
    rec["latency_s"] = round(time.time() - t0, 1)
    return rec


def _resize_longest(im, target_long):
    """等比缩放使最长边 = target_long。"""
    w, h = im.size
    scale = target_long / max(w, h)
    return im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                     Image.LANCZOS)


def _crop_strip(im, width, height):
    """从顶部裁一条 width×height（不足则取全宽）。"""
    w, h = im.size
    return im.crop((0, 0, min(width, w), min(height, h)))


def run_ladder(jobs, timeout):
    """jobs: List[(label, PIL.Image)]，并发执行，userId 轮询分配。"""
    results = {}
    with ThreadPoolExecutor(max_workers=len(API_USER_IDS)) as ex:
        fut = {}
        for i, (label, img) in enumerate(jobs):
            uid = API_USER_IDS[i % len(API_USER_IDS)]
            fut[ex.submit(_timed_call, img, uid, timeout)] = (label, img.size)
        for f in as_completed(fut):
            label, size = fut[f]
            r = f.result()
            r["size"] = list(size)
            results[label] = r
            print(f"  [{label:>16s}] size={size} -> {r['status']:4s} "
                  f"{r['latency_s']}s len={r['out_len']} {r.get('error') or ''}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=180,
                    help="单次超时（秒）")
    ap.add_argument("--mode", choices=["table", "long", "both"], default="both")
    args = ap.parse_args()

    out = {}

    if args.mode in ("table", "both"):
        # 取一张中等表格图，按最长边 ladder 缩放
        timg = sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "images", "*.jpg")))[3]
        im = Image.open(timg).convert("RGB")
        print(f"\n[TABLE] 基图 {os.path.basename(timg)} 原尺寸 {im.size}")
        ladder = [1500, 2500, 4000, 6000, 8000, 10000]
        jobs = [("table_L%d" % L, _resize_longest(im, L)) for L in ladder]
        out["table"] = {"base_image": os.path.basename(timg),
                        "base_size": list(im.size),
                        "ladder": run_ladder(jobs, args.timeout)}

    if args.mode in ("long", "both"):
        # 取一张面条图，固定宽 1500，按高度 ladder 切条
        limg = sorted(glob.glob(os.path.join(TRAIN_LONG_DIR, "images", "*.jpg")))[0]
        im = Image.open(limg).convert("RGB")
        print(f"\n[LONG] 基图 {os.path.basename(limg)} 原尺寸 {im.size}")
        heights = [1500, 2500, 3500, 5000, 7000, 10000]
        jobs = [("long_H%d" % H, _crop_strip(im, 1500, H)) for H in heights]
        out["long"] = {"base_image": os.path.basename(limg),
                       "base_size": list(im.size),
                       "ladder": run_ladder(jobs, args.timeout)}

    path = os.path.join(OUT_DIR, "calibrate_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n标定结果已保存：", path)

    # 简要建议
    print("\n===== 建议（基于成功且延迟可接受的最大档）=====")
    for kind in out:
        ok = [(lab, r) for lab, r in out[kind]["ladder"].items()
              if r["status"] == "ok"]
        if ok:
            ok.sort(key=lambda x: x[1]["size"][0] * x[1]["size"][1])
            big = ok[-1]
            print(f"  {kind}: 最大成功档 {big[0]} size={big[1]['size']} "
                  f"延迟 {big[1]['latency_s']}s")
        else:
            print(f"  {kind}: 无成功档，需降低尺寸或排查")


if __name__ == "__main__":
    main()
