# -*- coding: utf-8 -*-
"""严格零容差一致性审计(只统计,不修复,不改管线)。

对每个 tile 的**原始解析行**(绕过 EMPTY/FLAT/截断/塌缩等全部抢救与修复链)套判据:
  ① 非空行数 == |E|(墨迹有字行数)
  ② 每一行有效宽 ∈ [该行墨迹宽, nc](行级,零容差,不取众数)
任何一条不满足 → 不合格。跨列墨(colspan 疑似)单独标记,只 audit。
(v6 后管线判定即零容差,本工具用于管线外独立审计原始读数。)
API 走缓存重放;B 集缓存全热,无新调用。
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from common.preprocess import prep
from table.crop import crop
from table.grid_ocr import (slice_grid, _ink_evidence, _parse_cap, _w,
                            span_rows as _span_rows)
from table.tiles import call_tiles


def audit_image(path, dump=None):
    name = os.path.basename(path)
    uid = name.split("-")[0]
    res = {"file": name, "segs": 0, "misaligned": 0, "tiles": 0, "blank": 0,
           "fail": [], "span_tiles": 0}
    try:
        im = prep(Image.open(path))
        segs = [bb for k, bb in crop(im) if k not in ("text", "title")]
        for bb in segs:
            res["segs"] += 1
            seg = im.crop(bb)
            tiles, meta = slice_grid(seg)
            if meta["misaligned"]:
                res["misaligned"] += 1
                continue
            rb, cb = meta["rb"], meta["cb"]
            row_bands, col_bands = meta["row_bands"], meta["col_bands"]
            cell_ink, _ = _ink_evidence(seg, rb, cb, meta["rows"], meta["cols"])
            seg_gray = np.asarray(seg.convert("L"))
            flat = [(r, c) for r in range(len(tiles)) for c in range(len(tiles[r]))
                    if tiles[r][c] is not None]
            res["blank"] += sum(len(row) for row in tiles) - len(flat)
            outs = dict(zip(flat, call_tiles([tiles[r][c] for r, c in flat],
                                             timeout=240, upsample=meta["upsample"])))
            for (r, c) in flat:
                res["tiles"] += 1
                ri, rj = row_bands[r]
                ci, cj = col_bands[c]
                nc = cj - ci
                _, rows = _parse_cap(outs.get((r, c)))
                E = [i for i in range(ri, rj) if cell_ink[i, ci:cj].any()]
                inkws = [int(np.where(cell_ink[i, ci:cj])[0].max()) + 1 for i in E]
                span = _span_rows(seg_gray, rb, cb, ri, rj, ci, cj)
                if span:
                    res["span_tiles"] += 1
                    if dump:
                        from PIL import ImageDraw
                        t = seg.crop((cb[ci], rb[ri], cb[cj], rb[rj])).convert("RGB")
                        dr = ImageDraw.Draw(t)
                        for j in range(ci + 1, cj):
                            dr.line([(cb[j] - cb[ci], 0), (cb[j] - cb[ci], t.height)],
                                    fill=(255, 0, 0), width=2)
                        for i in span:
                            dr.rectangle([(0, rb[i] - rb[ri]),
                                          (t.width - 1, rb[i + 1] - rb[ri])],
                                         outline=(0, 150, 255), width=4)
                        t.save(os.path.join(dump, "span",
                               f"{uid}_y{bb[1]}_t{r}_{c}.jpg"), quality=90)
                nzrows = [x for x in rows if any(s.strip() for s in x)]
                reasons = []
                if len(nzrows) != len(E):
                    reasons.append(f"ROWS {len(nzrows)}!={len(E)}")
                bad_w = []
                for k, x in enumerate(nzrows):
                    wk = _w(x)
                    lo = inkws[k] if k < len(inkws) else 0
                    if not (lo <= wk <= nc):
                        bad_w.append((k, wk, lo))
                if bad_w:
                    k, wk, lo = bad_w[0]
                    reasons.append(f"W {len(bad_w)}行越界(如r{k}:w{wk}∉[{lo},{nc}])")
                if reasons:
                    collapsed = sum(1 for x in nzrows
                                    if sum(1 for s in x if s.strip()) == 1
                                    and len((next(s for s in x if s.strip())).split()) >= 2)
                    res["fail"].append({
                        "seg_y": bb[1], "tile": [r, c], "nc": nc, "E": len(E),
                        "nz": len(nzrows), "collapsed": collapsed,
                        "w_over": sum(1 for _k, w_, _lo in bad_w if w_ > nc),
                        "w_under": sum(1 for _k, w_, lo in bad_w if w_ < lo),
                        "reasons": reasons,
                        "span_rows": len(span)})
                    if dump:
                        from tools.dump_issue_tiles import _render_table, _stack
                        t = seg.crop((cb[ci], rb[ri], cb[cj], rb[rj]))
                        rrows = rows if rows else [["(空读)"]]
                        card = _stack(t, _render_table(rrows, "google-chrome"),
                                      f"{uid} seg@y{bb[1]} tile[{r}][{c}] "
                                      f"{';'.join(reasons)}",
                                      label2="RAW READ(原始解析行)")
                        tag = "RW" if len(reasons) == 2 else (
                            "R" if reasons[0].startswith("ROWS") else "W")
                        card.save(os.path.join(dump, "fail",
                                  f"{uid}_y{bb[1]}_t{r}_{c}_{tag}.jpg"), quality=90)
    except Exception as e:
        res["error"] = repr(e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=6)
    ap.add_argument("--dump", default=None, help="两类出图目录(fail/ 与 span/)")
    args = ap.parse_args()
    if args.dump:
        os.makedirs(os.path.join(args.dump, "fail"), exist_ok=True)
        os.makedirs(os.path.join(args.dump, "span"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    from functools import partial
    with Pool(args.pool) as pool:
        results = pool.map(partial(audit_image, dump=args.dump), files)

    tot = Counter()
    for r in results:
        tot["tiles"] += r["tiles"]
        tot["fail"] += len(r["fail"])
        tot["span"] += r["span_tiles"]
        tot["misaligned"] += r["misaligned"]
        tot["rows_bad"] += sum(1 for f in r["fail"] if any(
            x.startswith("ROWS") for x in f["reasons"]))
        tot["w_bad"] += sum(1 for f in r["fail"] if any(
            x.startswith("W") for x in f["reasons"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": dict(tot), "images": results},
                  f, ensure_ascii=False, indent=1, default=int)
    print(f"tile总数 {tot['tiles']}  不合格 {tot['fail']} "
          f"(行数不符{tot['rows_bad']} / 行宽越界{tot['w_bad']})")
    print(f"跨列墨tile {tot['span']}  misaligned段 {tot['misaligned']}")
    print(f"明细 → {args.out}")


if __name__ == "__main__":
    main()
