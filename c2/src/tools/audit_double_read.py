# -*- coding: utf-8 -*-
"""双读审计(零管线改动,本地小模型当第二证人):

#1 重读 tile(判定不合格):内容数字格占比(小模型安全区)+ 本地 vs 旧API缓存
   的数字 token 多重集重合率(量化两证人在数字位上的分歧)。
#2 合格 tile(API 通过占位恒等):抽样逐格本地重读 vs API 格值(数字位归一),
   **排除表首带**——专抓"占位对但值错"的隐形残错。输出分歧格清单。
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from common.preprocess import prep
from table.crop import crop
from table.grid_ocr import slice_grid, _ink_evidence, _read_tiles, _calibrate_cols
from table.cell_ocr import read_cells

NUM = re.compile(r"[\d.,%\-\s]+")


def _digits(s):
    return re.sub(r"\D", "", s or "")


def _w(row):
    last = max((k for k, s in enumerate(row) if s.strip()), default=-1)
    return last + 1


def audit_image(args):
    path, sample_per_tile = args
    uid = os.path.basename(path)[:8]
    res = {"file": uid, "reread": [], "ok_mismatch": [], "ok_cells": 0, "ok_agree": 0}
    try:
        im = prep(Image.open(path))
        for k, bb in crop(im):
            if k in ("text", "title"):
                continue
            seg = im.crop(bb)
            tiles, meta = slice_grid(seg)
            if meta["misaligned"]:
                continue
            rb, cb = meta["rb"], meta["cb"]
            ink, gray = _ink_evidence(seg, rb, cb, meta["rows"], meta["cols"])
            has = ink | gray
            parsed = _read_tiles(tiles, meta, 240)
            _calibrate_cols(parsed, meta)
            for r, (ri, rj) in enumerate(meta["row_bands"]):
                for c, (ci, cj) in enumerate(meta["col_bands"]):
                    if tiles[r][c] is None:
                        continue
                    _, rows = parsed[(r, c)]
                    E = [i for i in range(ri, rj) if has[i, ci:cj].any()]
                    nz = [x for x in rows if any(s.strip() for s in x)]

                    def occ(row, i):
                        for j in range(cj - ci):
                            if bool(j < len(row) and row[j].strip()) != bool(has[i, ci + j]):
                                return False
                        return not any(s.strip() for s in row[cj - ci:])
                    ok = (len(nz) == len(E)
                          and all(occ(x, i) for x, i in zip(nz, E)))
                    if not ok and E:
                        # #1: 本地重读 tile —— 数字占比 + 与API数字token重合率
                        cells = [(i, j) for i in E for j in range(ci, cj) if has[i, j]]
                        vals = read_cells(seg, rb, cb, cells)
                        vs = [v for v in vals.values() if v.strip()]
                        numeric = sum(1 for v in vs if NUM.fullmatch(v.strip()))
                        api_toks = Counter(_digits(t) for x in nz for t in x if _digits(t))
                        loc_toks = Counter(_digits(v) for v in vs if _digits(v))
                        inter = sum((api_toks & loc_toks).values())
                        union = sum((api_toks | loc_toks).values())
                        res["reread"].append({
                            "seg_y": int(bb[1]), "tile": [r, c],
                            "cells": len(vs), "num_frac": round(numeric / max(1, len(vs)), 3),
                            "digit_jaccard": round(inter / max(1, union), 3)})
                    elif ok and E and r > 0:
                        # #2: 合格 tile 抽样双读(排除表首带 r==0)
                        cand = [(i, j) for i in E for j in range(ci, cj) if has[i, j]]
                        step = max(1, len(cand) // sample_per_tile)
                        samp = cand[::step][:sample_per_tile]
                        vals = read_cells(seg, rb, cb, samp)
                        for (i, j) in samp:
                            k2 = E.index(i)
                            api_row = nz[k2] if k2 < len(nz) else []
                            av = api_row[j - ci] if j - ci < len(api_row) else ""
                            lv = vals.get((i, j), "")
                            if not _digits(av) and not _digits(lv):
                                continue
                            res["ok_cells"] += 1
                            if _digits(av) == _digits(lv):
                                res["ok_agree"] += 1
                            else:
                                res["ok_mismatch"].append({
                                    "seg_y": int(bb[1]), "tile": [r, c],
                                    "row": int(i), "col": int(j),
                                    "api": av[:20], "local": lv[:20]})
    except Exception as e:
        res["error"] = repr(e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=12, help="合格tile每块抽样格数")
    ap.add_argument("--pool", type=int, default=6)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    with Pool(args.pool) as pool:
        results = pool.map(audit_image, [(f, args.sample) for f in files])

    rr = [t for r in results for t in r["reread"]]
    mm = [m for r in results for m in r["ok_mismatch"]]
    cells = sum(r["ok_cells"] for r in results)
    agree = sum(r["ok_agree"] for r in results)
    print(f"#1 重读tile {len(rr)}个: 数字格占比中位 "
          f"{sorted(t['num_frac'] for t in rr)[len(rr)//2] if rr else 0:.3f}, "
          f"纯数字(≥95%)tile {sum(1 for t in rr if t['num_frac'] >= 0.95)}个, "
          f"数字token重合率中位 {sorted(t['digit_jaccard'] for t in rr)[len(rr)//2] if rr else 0:.3f}")
    low = [t for t in rr if t['num_frac'] < 0.8]
    print(f"   含文本较多(<80%数字)的重读tile: {len(low)}个")
    print(f"#2 合格tile双读(排除表首带): 抽样{cells}格, 一致{agree} "
          f"({agree / max(1, cells):.2%}), 分歧{len(mm)}格")
    for m in sorted(mm, key=lambda x: x['api'])[:15]:
        print(f"   {m}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"reread": rr, "ok_mismatch": mm,
                   "ok_cells": cells, "ok_agree": agree,
                   "images": results}, f, ensure_ascii=False, indent=1, default=int)
    print(f"明细 → {args.out}")


if __name__ == "__main__":
    main()
