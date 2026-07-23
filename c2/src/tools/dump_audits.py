# -*- coding: utf-8 -*-
"""audit 事件全量出图:按当前管线重放(缓存),把 ocr_seg meta['audit'] 逐条可视化。

分类:
  span/    压线行(本地逐格重读,跨列字可能切碎)→ tile原图+重构上下卡
  cols/    疑漏检列线 / 对齐缝复核未过 → 列区域裁剪,红线=骨架列边界
  lostrow/ 补空丢行 → 行条裁剪
  audits.txt 全部 audit 原文清单
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

from common.preprocess import prep
from table.crop import crop
from table.grid_ocr import ocr_seg, slice_grid
from tools.dump_issue_tiles import _render_table, _stack

TILE_RE = re.compile(r"tile\[(\d+)\]\[(\d+)\]")
COL_RE = re.compile(r"(?:骨架col|列带)(\d+)")
ROW_RE = re.compile(r"骨架行(\d+)")


def process(args):
    path, out = args
    uid = os.path.basename(path)[:8]
    lines = []
    try:
        im = prep(Image.open(path))
        for k, bb in crop(im):
            if k in ("text", "title"):
                continue
            seg = im.crop(bb)
            grid, _, meta = ocr_seg(seg)
            audits = meta.get("audit", [])
            if not audits:
                continue
            rb, cb = meta.get("rb"), meta.get("cb")
            for a in audits:
                lines.append(f"{uid} y{bb[1]} {a}")
                tag = f"{uid}_y{bb[1]}"
                if "压线行" in a and grid is not None:
                    m = TILE_RE.search(a)
                    if not m:
                        continue
                    r, c = int(m.group(1)), int(m.group(2))
                    ri, rj = meta["row_bands"][r]
                    ci, cj = meta["col_bands"][c]
                    t = seg.crop((cb[ci], rb[ri], cb[cj], rb[rj]))
                    sub = [row[ci:cj] for row in grid[ri:rj]]
                    card = _stack(t, _render_table(sub, "google-chrome"),
                                  f"{tag} {a}", label2="RECON(本地逐格)")
                    card.save(os.path.join(out, "span", f"{tag}_t{r}_{c}.jpg"),
                              quality=90)
                elif "补空丢行" in a and rb is not None:
                    # 复核未过/疑漏检列线 = 审计否决记录(列估计被维持原判),
                    # 不是问题,不出图,只进 audits.txt
                    m = ROW_RE.search(a)
                    if not m:
                        continue
                    i = int(m.group(1))
                    y0, y1 = rb[max(0, i - 1)], rb[min(len(rb) - 1, i + 2)]
                    t = seg.crop((0, y0, seg.width, y1))
                    t.thumbnail((1900, 600))
                    t.save(os.path.join(out, "lostrow", f"{tag}_r{i}.jpg"), quality=90)
    except Exception as e:
        lines.append(f"{uid} err {e!r}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=6)
    args = ap.parse_args()
    for sub in ("span", "cols", "lostrow"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    with Pool(args.pool) as pool:
        all_lines = [x for lst in pool.map(process, [(f, args.out) for f in files])
                     for x in lst]
    with open(os.path.join(args.out, "audits.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"audit事件 {len(all_lines)} 条 → {args.out}/ (span/cols/lostrow + audits.txt)")


if __name__ == "__main__":
    main()
