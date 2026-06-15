"""法规 html 解析器。csrc 页面正文统一在 div.detail-news。"""
from __future__ import annotations
from bs4 import BeautifulSoup
from .base import Parser
from .structure import build_blocks
from ..schema import Doc, Block, Table


class HtmlParser(Parser):
    domain = "regulatory"

    def parse(self, doc_id: str, path: str) -> Doc:
        html = open(path, encoding="utf-8", errors="ignore").read()
        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        title = t.get_text(strip=True).split("_")[0] if t else doc_id

        body = soup.find(class_="detail-news") or soup.find(class_="content") or soup
        blocks: list[Block] = []

        # 先抽表格，转 markdown
        tables = body.find_all("table")
        for i, tb in enumerate(tables):
            md = _table_to_md(tb)
            if md:
                blocks.append(Block(block_id=f"{doc_id}#t{i:03d}", type="table",
                                    text=md, table=Table(markdown=md)))
            tb.decompose()

        text = body.get_text("\n")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        blocks.extend(build_blocks(doc_id, lines))
        return Doc(doc_id=doc_id, domain=self.domain, title=title[:80],
                   source_path=path, blocks=blocks)


def _table_to_md(tb) -> str:
    rows = []
    for tr in tb.find_all("tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)
