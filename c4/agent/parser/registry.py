"""解析器选择 + fallback 链。

策略（面向 B 榜未知文档鲁棒性）：
  PDF: PyMuPDF(主) --质量门控不过--> MinerU(次) --仍不过/未装--> OCR 兜底
  html/txt: 直读（无需 fallback）

MinerU 不是按 A 榜人工挑的名单，而是由 quality.assess() 自动触发，
保证 B 榜出现大量 PyMuPDF 抽不动的文档时能自动补救。
"""
from __future__ import annotations
import fitz
from .txt import TxtParser
from .html import HtmlParser
from .pdf_pymupdf import PdfParser
from .quality import assess, QualityReport
from ..schema import Doc


def get_parser(domain: str, ext: str):
    """单解析器（不含 fallback），保留给需要指定解析器的场景。"""
    if ext == "txt":
        return TxtParser()
    if ext in ("html", "htm"):
        return HtmlParser()
    if ext == "pdf":
        return PdfParser(domain=domain)
    raise ValueError(f"no parser for ext={ext}")


def parse_with_fallback(doc_id: str, path: str, domain: str, ext: str,
                        enable_mineru: bool = True) -> tuple[Doc, QualityReport | None, str]:
    """返回 (doc, 质量报告, 实际使用的解析器名)。"""
    if ext != "pdf":
        return get_parser(domain, ext).parse(doc_id, path), None, ext

    # 1) 主解析 PyMuPDF
    doc = PdfParser(domain=domain).parse(doc_id, path)
    try:
        n_pages = fitz.open(path).page_count
    except Exception:
        n_pages = max((b.page or 1) for b in doc.blocks) if doc.blocks else 1
    report = assess(doc, n_pages)
    if not report.needs_fallback:
        return doc, report, "pymupdf"

    # 2) 质量不过 → MinerU
    if enable_mineru:
        try:
            from .pdf_mineru import MinerUParser, MinerUNotAvailable
            try:
                mdoc = MinerUParser(domain=domain).parse(doc_id, path)
                mrep = assess(mdoc, n_pages)
                if mdoc.char_len() > doc.char_len():   # 取更充分的结果
                    return mdoc, mrep, "mineru"
            except MinerUNotAvailable as e:
                report.reasons.append(f"mineru_unavailable:{e.args[0][:30]}")
        except Exception as e:
            report.reasons.append(f"mineru_error:{str(e)[:40]}")

    # 3) 仍不过：保留 PyMuPDF 结果并标记（后续 OCR/VL 兜底由检索层按需触发）
    return doc, report, "pymupdf_low_quality"
