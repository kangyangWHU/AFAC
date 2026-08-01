# -*- coding: utf-8 -*-
"""在 LONG 的 pipeline 文本产出(*_txt.csv)之上施加几何标题定级,写出最终 *_raw.csv。

这是 B 榜 long 的**最后一步驱动**:pipeline(run_long,走 API)先产出 *_txt.csv,
本脚本用 long.geom_heading.correct() 逐篇重定级标题(需原图算字号/淡横线),得 *_raw.csv。
correct() 内部按图像 hash 缓存逐行 OCR(cache_geo/),重跑只需数秒。

复现最终 long 提交:
  python -m tools.apply_geom_long \
      --txt ../out/_long_B_v9_txt.csv \
      --images "../data/AFACB榜评测数据集/finix_huge_long_rest_B/images" \
      --out ../out/_long_B_v9_raw.csv

与 build_long_submission 合并表格半即得整份提交。
"""
import os
import sys
import csv
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
csv.field_size_limit(sys.maxsize)

from common.preprocess import prep
from long.geom_heading import correct

_IMAGES = None          # 子进程共享(fork 继承);由 main 设定
_PREDS = None


def _work(n):
    from table.cell_ocr import _engine       # 惰性:每进程各自初始化 rapidocr
    eng = _engine()

    def ocr(pil):
        r = eng(np.asarray(pil.convert("RGB")), use_det=False, use_cls=False, use_rec=True)
        return r.txts[0] if r and getattr(r, "txts", None) else ""

    im = prep(Image.open(os.path.join(_IMAGES, n)))
    new, ch = correct(_PREDS[n], im, ocr, cache_key=n)
    return n, new, ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", required=True, help="pipeline 文本产出 *_txt.csv")
    ap.add_argument("--images", required=True, help="对应原图目录")
    ap.add_argument("--out", required=True, help="输出 *_raw.csv")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    global _IMAGES, _PREDS
    _IMAGES = a.images
    with open(a.txt, encoding="utf-8") as f:
        _PREDS = {r["file_name"]: r["ground_truth"] for r in csv.DictReader(f)}

    names = [n for n in sorted(os.listdir(_IMAGES)) if n in _PREDS]
    with Pool(a.workers) as p:
        res = p.map(_work, names)

    out = {n: new for n, new, _ in res}
    nch = sum(1 for _, _, ch in res if ch)
    ndrop = sum(len(ch) for _, _, ch in res)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["file_name", "ground_truth"])
        for n in sorted(_PREDS):
            w.writerow([n, out.get(n, _PREDS[n])])
    print(f"geom 施加 {len(names)} 篇;{ndrop} 处标题改动 across {nch} 篇 → {a.out}")


if __name__ == "__main__":
    main()
