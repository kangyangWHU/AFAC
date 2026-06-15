"""解析质量门控：判断一篇 PyMuPDF 解析结果是否可疑，需要 fallback 到 MinerU/OCR。
面向 B 榜未知文档的鲁棒性——不靠人工挑名单，靠自动信号。
"""
from __future__ import annotations
from dataclasses import dataclass
from ..schema import Doc

# 触发阈值（保守、高精度，避免误伤正常文字层文档）
# 设计教训：不能用"空页比例"——长财报/募集书天然有大量稀疏页(封面/分隔/纯图/签字页)，
# 但整篇 cpp 仍高、抽取正常。真正该 fallback 的信号是"整篇文字密度过低"。
MIN_CHARS_PER_PAGE = 100     # 整篇 cpp 低于此 ≈ 扫描件/图片型(需OCR/MinerU)
MIN_TOTAL_CHARS = 200        # 整篇近空
# 大部分页都近空 且 整篇密度也偏低 时才算可疑（双条件，避免误伤长文档）
SUSPECT_CPP = 250
MAX_EMPTY_PAGE_RATIO = 0.85


@dataclass
class QualityReport:
    needs_fallback: bool
    reasons: list[str]
    chars_per_page: float
    n_pages: int

    def __str__(self):
        tag = "FALLBACK" if self.needs_fallback else "OK"
        return f"[{tag}] cpp={self.chars_per_page:.0f} pages={self.n_pages} {self.reasons}"


def assess(doc: Doc, n_pages: int) -> QualityReport:
    reasons: list[str] = []
    total = doc.char_len()
    n_pages = max(n_pages, 1)
    cpp = total / n_pages

    if total < MIN_TOTAL_CHARS:
        reasons.append("near_empty_doc")
    if cpp < MIN_CHARS_PER_PAGE:
        reasons.append("low_text_density")  # 扫描件/图片型 → OCR/MinerU

    # 空页比例仅在"整篇密度本就偏低"时才作为辅助信号（双条件，避免误伤长文档）
    if n_pages > 3 and cpp < SUSPECT_CPP:
        page_chars: dict[int, int] = {}
        for b in doc.blocks:
            if b.page:
                page_chars[b.page] = page_chars.get(b.page, 0) + len(b.text)
        empty = sum(1 for p in range(1, n_pages + 1) if page_chars.get(p, 0) < 20)
        if empty / n_pages > MAX_EMPTY_PAGE_RATIO:
            reasons.append(f"sparse_low_density({empty}/{n_pages}pg, cpp={cpp:.0f})")

    return QualityReport(bool(reasons), reasons, cpp, n_pages)
