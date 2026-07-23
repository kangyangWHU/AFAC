# -*- coding: utf-8 -*-
"""LONG 回归基线：训练集 100 张全缓存重放，逐文档记 text / read / 标题层级。

改动 heading_norm / slicer_long / 表格装配前后各跑一次，比对 out/long_baseline*.json，
用来判断一次改动是真收益还是过拟合。缓存 100% 命中时不打 API。

标题层级指标是本地补的:官方"标题对齐"看绝对层级(见 doc/答疑)，而 metrics/evaluate.py
只实现了 text/teds/read 三项。这里按归一化标题文本做序列对齐，再比层级:
  level_acc = 文本对上且层级也对的标题数 / max(pred 标题数, gt 标题数)
分母取 max 是为了同时惩罚漏标题和多标题(只除以 gt 数的话,乱加标题不扣分)。

用法(在 src 目录)：
  python -m tools.eval_long_baseline --tag before
  python -m tools.eval_long_baseline --tag after_C
"""
import os
import re
import glob
import json
import time
import difflib
import argparse
import unicodedata
from functools import partial
from multiprocessing import Pool

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from common.config import TRAIN_LONG_DIR, OUT_DIR
from main import process_image
from metrics.evaluate import evaluate_one

_H = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _headings(md):
    """[(level, 归一化文本)]。归一化只为对齐用,不影响评分口径。"""
    out = []
    for line in (md or "").splitlines():
        m = _H.match(line)
        if m and m.group(2):
            text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", m.group(2)))
            if text:
                out.append((len(m.group(1)), text))
    return out


def heading_stats(pred, gt):
    """返回 (level_acc, text_recall, n_pred, n_gt)。

    text_recall 单独看是为了把两类错误拆开:标题该不该是标题(召回) vs 层级对不对。
    否则一次改动同时动了两者时看不出是哪边在变。
    """
    hp, hg = _headings(pred), _headings(gt)
    if not hg:
        return (1.0 if not hp else 0.0), 1.0, len(hp), 0
    sm = difflib.SequenceMatcher(a=[t for _, t in hp], b=[t for _, t in hg], autojunk=False)
    matched = same = 0
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            matched += 1
            if hp[i + k][0] == hg[j + k][0]:
                same += 1
    denom = max(len(hp), len(hg))
    return same / denom, matched / len(hg), len(hp), len(hg)


def _one(item, timeout):
    img, md = item
    uuid = os.path.basename(img)[:-4]
    gt = open(md, encoding="utf-8").read()
    try:
        pred, _, ncalls = process_image(img, "long", timeout=timeout)
    except Exception as e:
        return {"uuid": uuid, "err": str(e)}
    r = evaluate_one(pred, gt)
    lv, rc, np_, ng = heading_stats(pred, gt)
    return {"uuid": uuid, "ncalls": ncalls,
            "text_score": r["text_score"], "read_score": r["read_score"],
            "overall_2term": r["overall_2term"],
            "teds_score": r["teds_score"],          # None = GT 无表格
            "level_acc": lv * 100, "head_recall": rc * 100,
            "n_head_pred": np_, "n_head_gt": ng}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="快照名,出 out/long_baseline_<tag>.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()

    tasks = []
    for md in sorted(glob.glob(os.path.join(TRAIN_LONG_DIR, "mds", "*.md"))):
        if md.endswith("_pred.md"):            # mds/ 里混有预测残留,不是 GT
            continue
        img = os.path.join(TRAIN_LONG_DIR, "images", os.path.basename(md)[:-3] + ".jpg")
        if os.path.exists(img):
            tasks.append((img, md))

    t0 = time.time()
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(partial(_one, timeout=a.timeout), tasks)):
            if "err" in r:
                print(f"  [{i+1}/{len(tasks)}] {r['uuid'][:8]} 失败: {r['err']}", flush=True)
                continue
            rows.append(r)
            print(f"  [{i+1}/{len(tasks)}] {r['uuid'][:8]} "
                  f"text={r['text_score']:5.1f} read={r['read_score']:5.1f} "
                  f"lv={r['level_acc']:5.1f} rc={r['head_recall']:5.1f} "
                  f"({r['n_head_pred']}/{r['n_head_gt']})", flush=True)

    keys = ("text_score", "read_score", "overall_2term", "level_acc", "head_recall")
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    withtb = [r for r in rows if r["teds_score"] is not None]
    mean["teds_score"] = (sum(r["teds_score"] for r in withtb) / len(withtb)
                          if withtb else None)          # 只在含表样本上平均
    mean["n_with_table"] = len(withtb)
    calls = sum(r["ncalls"] for r in rows)
    rows.sort(key=lambda x: x["level_acc"])
    out = os.path.join(OUT_DIR, f"long_baseline_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tag": a.tag, "n": len(rows), "mean": mean, "rows": rows},
                  f, ensure_ascii=False, indent=1)

    print(f"\n=== LONG 基线 [{a.tag}]  {len(rows)} 张  {time.time()-t0:.0f}s  API调用 {calls} ===")
    for k in keys:
        print(f"  {k:14} {mean[k]:6.2f}")
    if mean["teds_score"] is not None:
        print(f"  {'teds_score':14} {mean['teds_score']:6.2f}  (含表 {mean['n_with_table']} 张)")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
