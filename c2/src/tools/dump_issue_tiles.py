# -*- coding: utf-8 -*-
"""问题 tile 审计卡片:tile 原图 + 该区域最终重构(装配后子表)上下拼一张图。

用法:
  python tools/dump_issue_tiles.py --image IMG --y 659 --tiles "1,1 1,2" --out DIR
ocr_seg 走 API 缓存重放,不产生新调用(除非缓存缺失)。
"""
import argparse
import html
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

from common.preprocess import prep
from table.crop import crop
from table.grid_ocr import slice_grid, ocr_seg

CSS = """
body { font: 13px/1.5 "Noto Sans CJK SC", sans-serif; margin: 8px; background: white; }
table { border-collapse: collapse; }
td { border: 1px solid #333; padding: 2px 6px; white-space: nowrap; }
"""


def _render_table(rows, chrome, width=6000, height=4000):
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c or '')}</td>" for c in r) + "</tr>"
        for r in rows)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "t.html"
        page.write_text(f'<meta charset="utf-8"><style>{CSS}</style>'
                        f"<table>{body}</table>", encoding="utf-8")
        png = Path(td) / "t.png"
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", f"--screenshot={png}",
             f"--window-size={width},{height}", page.as_uri()],
            check=True, capture_output=True, timeout=120)
        im = Image.open(png).convert("RGB")
        bbox = im.convert("L").point(lambda p: 255 - p).getbbox()
        if bbox:
            im = im.crop((0, 0, min(im.width, bbox[2] + 12),
                          min(im.height, bbox[3] + 12)))
        return im


def _stack(tile_im, recon_im, header, label2="RECON(装配后子表)"):
    """上 tile 下重构,左上角灰条标注。"""
    bar = 28
    w = max(tile_im.width, recon_im.width)
    h = bar + tile_im.height + bar + recon_im.height
    out = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, w, bar), fill="#444")
    d.text((8, 6), f"{header}  |  TILE", fill="white")
    out.paste(tile_im, (0, bar))
    y = bar + tile_im.height
    d.rectangle((0, y, w, y + bar), fill="#08519c")
    d.text((8, y + 6), label2, fill="white")
    out.paste(recon_im, (0, y + bar))
    return out


def band_widths(meta):
    """各列带在最终 grid 中的宽度:骨架宽 + 列校准 adopt 修正。"""
    widths = [cj - ci for ci, cj in meta["col_bands"]]
    for a in meta.get("adopt", []):
        m = re.match(r"列带(\d+) 列校准采纳 骨架\d+列→(\d+)列", a)
        if m:
            widths[int(m.group(1))] = int(m.group(2))
    return widths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--y", type=int, required=True, help="seg 在整图中的 y0")
    ap.add_argument("--tiles", required=True, help='"r,c r,c ..."')
    ap.add_argument("--out", required=True)
    ap.add_argument("--chrome", default="google-chrome")
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    uid = os.path.basename(args.image).split("-")[0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    im = prep(Image.open(args.image))
    segs = [bb for k, bb in crop(im) if k not in ("text", "title") and bb[1] == args.y]
    if not segs:
        sys.exit(f"seg@y{args.y} 未找到")
    seg = im.crop(segs[0])

    _, smeta = slice_grid(seg)
    grid, _, meta = ocr_seg(seg, timeout=args.timeout)
    if grid is None:
        sys.exit("ocr_seg 骨架不可信(misaligned),无重构")
    rb, cb = smeta["rb"], smeta["cb"]
    widths = band_widths(meta)

    for spec in args.tiles.split():
        r, c = map(int, spec.split(","))
        ri, rj = meta["row_bands"][r]
        ci, cj = meta["col_bands"][c]
        tile_im = seg.crop((cb[ci], rb[ri], cb[cj], rb[rj]))
        off = sum(widths[:c])
        sub = [row[off:off + widths[c]] for row in grid[ri:rj]]
        recon_im = _render_table(sub, args.chrome)
        hdr = (f"{uid} seg@y{args.y} tile[{r}][{c}]  骨架{rj - ri}行x{cj - ci}列 "
               f"重构{len(sub)}行x{widths[c]}列")
        _stack(tile_im, recon_im, hdr).save(
            out_dir / f"{uid}_y{args.y}_t{r}_{c}.jpg", quality=92)
        print(f"{uid}_y{args.y}_t{r}_{c}.jpg  骨架{rj-ri}x{cj-ci} 重构{len(sub)}x{widths[c]}",
              flush=True)


if __name__ == "__main__":
    main()
