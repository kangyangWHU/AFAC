"""全库预解析(新 parse):PDF→PaddleOCR-VL 逐页 md;html/txt→直读。
输出 processed_vl/{domain}/{doc_id}/p{N}.md + manifest.json,断点续跑。

运行环境:conda env `paddleocr`(连 vLLM 服务 8118)。
用法:
  python -m script.preprocess_vl --all                 # 全库
  python -m script.preprocess_vl --domain financial_reports
  python -m script.preprocess_vl --domain insurance --limit 2
  python -m script.preprocess_vl --all --force          # 重跑(忽略已完成)
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.doc_index import build_doc_index            # noqa: E402
from agent.vl.parser import parse_pdf                  # noqa: E402

RAW = os.path.join(ROOT, "data", "public_dataset_upload", "raw")
OUT = os.path.join(ROOT, "processed_vl")


class _HTMLText(HTMLParser):
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "table"}
    _SKIP = {"script", "style", "head"}

    def __init__(self):
        super().__init__()
        self.buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1
        elif tag in self._BLOCK:
            self.buf.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.buf.append("\n")

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.buf.append(data.strip())

    def text(self) -> str:
        import re
        t = " ".join(x if x != "\n" else "\n" for x in self.buf)
        t = re.sub(r"[ \t]+", " ", t)
        return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def _html_pages(path: str) -> list[dict]:
    raw = open(path, encoding="utf-8", errors="ignore").read()
    txt = None
    try:                                                  # trafilatura 抽正文,去导航/页眉页脚
        import trafilatura
        txt = trafilatura.extract(raw, include_tables=True, include_comments=False,
                                  favor_recall=True)
    except Exception:
        txt = None
    if not txt:                                           # 回退:朴素标签剥离
        p = _HTMLText()
        p.feed(raw)
        txt = p.text()
    return [{"page_no": 1, "markdown": txt}]


def _txt_pages(path: str) -> list[dict]:
    return [{"page_no": 1, "markdown": open(path, encoding="utf-8", errors="ignore").read().strip()}]


def _done(out_dir: str) -> bool:
    return os.path.exists(os.path.join(out_dir, "manifest.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    index = build_doc_index(RAW)
    targets = sorted(index)
    if args.domain:
        targets = [d for d in targets if index[d]["domain"] == args.domain]
    if args.limit:
        targets = targets[:args.limit]

    total_pages = 0
    t0 = time.time()
    done = skipped = failed = 0
    for i, doc_id in enumerate(targets, 1):
        m = index[doc_id]
        dom, ext, path = m["domain"], m["ext"].lower(), m["path"]
        out_dir = os.path.join(OUT, dom, doc_id)
        if _done(out_dir) and not args.force:
            skipped += 1
            continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            t = time.time()
            if ext == "pdf":
                pages = parse_pdf(path, dpi=args.dpi, max_concurrency=args.max_concurrency)
                parser = "paddleocr-vl"
            elif ext in ("html", "htm"):
                pages = _html_pages(path); parser = "html"
            else:
                pages = _txt_pages(path); parser = "txt"
            for p in pages:
                open(os.path.join(out_dir, f"p{p['page_no']}.md"), "w", encoding="utf-8").write(p["markdown"])
            json.dump({"doc_id": doc_id, "domain": dom, "ext": ext, "parser": parser,
                       "n_pages": len(pages)},
                      open(os.path.join(out_dir, "manifest.json"), "w"), ensure_ascii=False)
            total_pages += len(pages)
            done += 1
            dt = time.time() - t
            rate = total_pages / max(1e-9, time.time() - t0)
            print(f"[{i}/{len(targets)}] {dom}/{doc_id} {parser} {len(pages)}页 {dt:.1f}s "
                  f"| 累计{total_pages}页 {rate:.2f}页/s", flush=True)
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(targets)}] {dom}/{doc_id} ERR: {e}", flush=True)

    print(f"\n完成 {done} / 跳过 {skipped} / 失败 {failed}; {total_pages}页 "
          f"{(time.time()-t0)/60:.1f}min -> {OUT}")


if __name__ == "__main__":
    main()
