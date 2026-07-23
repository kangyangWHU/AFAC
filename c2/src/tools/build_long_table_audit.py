# -*- coding: utf-8 -*-
"""Build a local side-by-side audit page for LONG documents containing tables."""
import argparse
import csv
import html
import os
import re
import sys

csv.field_size_limit(sys.maxsize)
from pathlib import Path
from urllib.parse import quote


TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.I | re.S)
ROW_RE = re.compile(r"<tr\b", re.I)
CELL_RE = re.compile(r"<t[dh]\b", re.I)


def _context(doc, match, radius=300):
    before = doc[max(0, match.start() - radius):match.start()]
    after = doc[match.end():match.end() + radius]
    return before, after


def build(csv_path, image_dir, output_path):
    csv_path = Path(csv_path).resolve()
    image_dir = Path(image_dir).resolve()
    output_path = Path(output_path).resolve()
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    cards = []
    for row in rows:
        doc = row.get("ground_truth", "")
        matches = list(TABLE_RE.finditer(doc))
        if not matches:
            continue
        name = row["file_name"]
        image_path = image_dir / name
        image_url = quote(os.path.relpath(image_path, output_path.parent))
        sections = []
        for i, match in enumerate(matches, 1):
            table = match.group(0)
            before, after = _context(doc, match)
            sections.append(
                '<section class="table-block">'
                f'<h3>Table {i}: {len(ROW_RE.findall(table))} rows, '
                f'{len(CELL_RE.findall(table))} cells</h3>'
                f'<pre>{html.escape(before)}</pre>'
                f'<div class="rendered">{table}</div>'
                f'<pre>{html.escape(after)}</pre>'
                '</section>'
            )
        cards.append(
            '<article class="card">'
            f'<h2>{html.escape(name)}</h2>'
            '<div class="columns">'
            f'<div class="image-pane"><img src="{image_url}" alt="source image"></div>'
            f'<div class="prediction-pane">{"".join(sections)}</div>'
            '</div></article>'
        )

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>LONG table audit</title>
<style>
body {{ font: 14px/1.4 system-ui, sans-serif; margin: 20px; background: #f5f5f5; }}
.card {{ background: white; margin: 0 0 28px; padding: 16px; border: 1px solid #bbb; }}
.columns {{ display: grid; grid-template-columns: minmax(360px, 45%) 1fr; gap: 18px; }}
.image-pane {{ height: 850px; overflow: auto; border: 1px solid #aaa; background: #ddd; }}
.image-pane img {{ width: 100%; height: auto; display: block; }}
.prediction-pane {{ max-height: 850px; overflow: auto; }}
.table-block {{ margin-bottom: 28px; border-bottom: 2px solid #888; padding-bottom: 20px; }}
.rendered {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #777; padding: 3px 6px; }}
pre {{ white-space: pre-wrap; background: #f3f3f3; padding: 8px; max-height: 150px; overflow: auto; }}
</style>
<h1>LONG table audit — {len(cards)} documents</h1>
<p>Check table count, row continuity, missing/duplicate rows, cell merges, and surrounding text.</p>
{"".join(cards)}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    print(f"Wrote {output_path} ({len(cards)} documents)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build(args.csv, args.images, args.out)


if __name__ == "__main__":
    main()
