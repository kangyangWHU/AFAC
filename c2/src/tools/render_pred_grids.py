# -*- coding: utf-8 -*-
"""Render submission predictions to images for side-by-side audit.

For each CSV row: copy the source image as <uid>.jpg and render the predicted
markdown/HTML to <uid>_pred.jpg via headless Chrome full-page screenshot.
"""
import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
csv.field_size_limit(sys.maxsize)

TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.I | re.S)

CSS = """
body { font: 13px/1.5 "Noto Sans CJK SC", sans-serif; margin: 16px; background: white; }
table { border-collapse: collapse; margin: 10px 0; }
td, th { border: 1px solid #333; padding: 2px 6px; white-space: nowrap; }
p { margin: 6px 0; white-space: pre-wrap; }
"""


def pred_to_html(doc):
    """表格保留原 HTML,表外文本转义成段落。"""
    parts, pos = [], 0
    for m in TABLE_RE.finditer(doc):
        parts.append(f"<p>{html.escape(doc[pos:m.start()])}</p>")
        parts.append(m.group(0))
        pos = m.end()
    parts.append(f"<p>{html.escape(doc[pos:])}</p>")
    return f'<meta charset="utf-8"><style>{CSS}</style>{"".join(parts)}'


def _trim_white(im, margin=20):
    """裁掉右/下大片空白(截图窗口远大于内容)。"""
    gray = im.convert("L")
    bbox = gray.point(lambda p: 255 - p).getbbox()
    if not bbox:
        return im
    w = min(im.width, bbox[2] + margin)
    h = min(im.height, bbox[3] + margin)
    return im.crop((0, 0, w, h))


def render_one(doc, out_jpg, chrome, width, height):
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "page.html"
        page.write_text(pred_to_html(doc), encoding="utf-8")
        png = Path(td) / "shot.png"
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", f"--screenshot={png}",
             f"--window-size={width},{height}", page.as_uri()],
            check=True, capture_output=True, timeout=120)
        im = _trim_white(Image.open(png).convert("RGB"))
        im.save(out_jpg, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chrome", default="google-chrome")
    ap.add_argument("--no_orig", action="store_true", help="只写 _pred.jpg,不复制原图")
    ap.add_argument("--width", type=int, default=7000)
    ap.add_argument("--height", type=int, default=8000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for i, row in enumerate(rows, 1):
        name = row["file_name"]
        uid = os.path.splitext(name)[0]
        src = Path(args.images) / name
        if not args.no_orig and src.exists():
            shutil.copy(src, out_dir / name)
        try:
            render_one(row.get("ground_truth", ""), out_dir / f"{uid}_pred.jpg",
                       args.chrome, args.width, args.height)
            print(f"[{i}/{len(rows)}] {uid}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(rows)}] {uid} 失败: {e}", flush=True)


if __name__ == "__main__":
    main()
