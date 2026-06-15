"""扫描全库 PDF，报告质量门控会把哪些文档路由到 MinerU fallback。
用法：python -m eval.fallback_scan
"""
from __future__ import annotations
import os
import sys
import json
import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent.parser.pdf_pymupdf import PdfParser   # noqa: E402
from agent.parser.quality import assess          # noqa: E402


def main():
    idx = json.load(open(os.path.join(ROOT, "doc_index.json"), encoding="utf-8"))
    pdfs = [(d, v) for d, v in idx.items() if v["ext"] == "pdf"]
    flagged = []
    for doc_id, v in pdfs:
        try:
            doc = PdfParser(domain=v["domain"]).parse(doc_id, v["path"])
            npg = fitz.open(v["path"]).page_count
            rep = assess(doc, npg)
            if rep.needs_fallback:
                flagged.append((doc_id, v["domain"], rep))
        except Exception as e:
            flagged.append((doc_id, v["domain"], f"PARSE_ERROR:{str(e)[:40]}"))
    print(f"=== 质量门控扫描 {len(pdfs)} 篇 PDF ===")
    print(f"触发 MinerU fallback: {len(flagged)} 篇\n")
    for doc_id, dom, rep in flagged:
        print(f"  {doc_id:30} [{dom}] {rep}")


if __name__ == "__main__":
    main()
