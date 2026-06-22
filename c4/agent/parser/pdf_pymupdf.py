"""PDF 解析器（PyMuPDF）。提取正文 + 表格 + 页码。
MinerU 可作为更强的替换实现，接口一致。"""
from __future__ import annotations
import fitz  # PyMuPDF
from .base import Parser, detect_article_no, heading_level, dedup_repeated_lines
from ..schema import Doc, Block, Table


class PdfParser(Parser):
    def __init__(self, domain: str = ""):
        self.domain = domain

    def parse(self, doc_id: str, path: str) -> Doc:
        pdf = fitz.open(path)
        title = (pdf.metadata or {}).get("title") or doc_id
        all_lines: list[tuple[str, int]] = []   # (line, page)
        table_blocks: list[Block] = []
        ti = 0
        for pno in range(pdf.page_count):
            page = pdf[pno]
            # 表格
            try:
                for tb in page.find_tables().tables:
                    md = _grid_to_md(tb.extract())
                    if md:
                        table_blocks.append(Block(
                            block_id=f"{doc_id}#t{ti:03d}", type="table",
                            text=md, page=pno + 1, table=Table(markdown=md)))
                        ti += 1
            except Exception:
                pass
            text = page.get_text("text")
            for ln in text.splitlines():
                if ln.strip():
                    all_lines.append((ln.strip(), pno + 1))
        pdf.close()

        kept = dedup_repeated_lines([l for l, _ in all_lines])
        keptset_pages = _realign(all_lines, kept)
        blocks = _build_with_pages(doc_id, keptset_pages)
        blocks = table_blocks + blocks
        first_text = next((b.text for b in blocks if b.type != "table"), doc_id)
        if not title or title == doc_id:
            title = first_text[:40]
        return Doc(doc_id=doc_id, domain=self.domain, title=str(title)[:80],
                   source_path=path, blocks=blocks)


def _realign(all_lines, kept):
    keptset = set(kept)
    return [(l, p) for l, p in all_lines if l in keptset]


def _build_with_pages(doc_id: str, lines_pages: list[tuple[str, int]]) -> list[Block]:
    blocks: list[Block] = []
    section_path: list[str] = []
    sec_stack: list[tuple[int, str]] = []          # (层级, 标题), 同级替换式嵌套
    cur: list[str] = []
    cur_page = None
    cur_article = None
    cur_type = "text"
    idx = 0

    def flush():
        nonlocal cur, cur_article, cur_type, idx
        if cur:
            text = "\n".join(cur).strip()
            if text:
                blocks.append(Block(block_id=f"{doc_id}#b{idx:04d}", type=cur_type,
                                    text=text, section_path=list(section_path),
                                    article_no=cur_article, page=cur_page))
                idx += 1
        cur, cur_article, cur_type = [], None, "text"

    for line, page in lines_pages:
        h = heading_level(line)
        if h:
            lvl, htext = h
            flush()
            while sec_stack and sec_stack[-1][0] >= lvl:
                sec_stack.pop()
            sec_stack.append((lvl, htext))
            section_path[:] = [t for _, t in sec_stack]
            blocks.append(Block(block_id=f"{doc_id}#b{idx:04d}", type="heading",
                                text=htext, section_path=list(section_path), page=page))
            idx += 1
            continue
        art = detect_article_no(line)
        if art:
            flush()
            cur, cur_article, cur_type, cur_page = [line], art, "article", page
        else:
            if not cur:
                cur_page = page
            cur.append(line)
    flush()
    return blocks


def _grid_to_md(grid) -> str:
    rows = [["" if c is None else str(c).replace("\n", " ").strip() for c in r]
            for r in grid if r and any(c is not None and str(c).strip() for c in r)]
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)
