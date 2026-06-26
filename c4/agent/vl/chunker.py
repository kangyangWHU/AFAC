"""新 chunk:结构感知,只产段落块(BM25 唯一合法召回单位)。
- 正式 QA 禁用一切 DL 语义表征(embedding/dense/rerank),故无句子/embedding 子层。
- 段落为基本单元(单行也算),过大按句末切,带 breadcrumb(文档名/章节/条款号)入索引补词面。
- 表格:每行+表头 = 一个段落块。条款当正文按段落处理(不整条留)。
- rrf_fuse:融合多条 **BM25** 排名(原 query + 分解子 query),全程免费、合规。
输入:processed_vl/{domain}/{doc_id}/(p*.md + manifest.json)。

依赖 agent.parser.base 的中文标题/条款正则(纯工具)。
"""
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from typing import Iterator, cast
from html.parser import HTMLParser

from ..textutil import heading_level, detect_article_no

_SENT_SPLIT = re.compile(r"[^。！？；]*[。！？；]|[^。！？；]+$")
_TABLE_ROW = re.compile(r"^\|.*\|$")
_SEP_ROW = re.compile(r"^\|[\s:|\-]+\|$")
_MD_HEAD = re.compile(r"^(#{1,6})\s+(.+)$")
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)


class _HTMLRows(HTMLParser):
    """从 HTML 表(含合并单元格)抽每行单元格文本。"""
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._r: list[str] | None = None
        self._c: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._r = []
        elif tag in ("td", "th"):
            self._c = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._c is not None and self._r is not None:
            self._r.append("".join(self._c).strip())
            self._c = None
        elif tag == "tr" and self._r is not None:
            self.rows.append(self._r)
            self._r = None

    def handle_data(self, data):
        if self._c is not None:
            self._c.append(data)


def _html_table_to_pipe(html: str) -> list[str]:
    """HTML 表 → 管道行(复用 markdown 表的逐行+表头切分)。"""
    p = _HTMLRows()
    p.feed(html)
    return ["| " + " | ".join(r) + " |" for r in p.rows if any(c for c in r)]
# LaTeX 噪声清洗(幂等;覆盖解析时未清的旧缓存)
_LATEX_CMD = re.compile(r"\\(?:underline|textbf|textit|text|mathrm|mathbf|emph|boldsymbol)\s*\{([^{}]*)\}")
_LATEX_SUP = re.compile(r"[\^_]\{([^{}]*)\}")


def _strip_latex(s: str) -> str:
    for _ in range(3):                       # 迭代剥嵌套 \underline{\text{..}}
        ns = _LATEX_CMD.sub(r"\1", s)
        if ns == s:
            break
        s = ns
    s = _LATEX_SUP.sub(r"\1", s)
    return s.replace("$", "")


@dataclass
class Chunk:                       # 段落块:父单位(喂LLM/引用/BM25)
    chunk_id: str
    doc_id: str
    domain: str
    page_no: int
    type: str                      # text | clause | table_row
    section_path: list[str]
    breadcrumb: str
    article_no: str | None
    text: str                      # 纯内容
    seq: int = 0                   # 篇内序号(neighbors 回填用)

    def to_dict(self) -> dict:     # 对齐 BM25Index 所需字段(page 非 page_no)
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "domain": self.domain,
                "page": self.page_no, "type": self.type, "section_path": self.section_path,
                "article_no": self.article_no, "text": self.text, "seq": self.seq,
                "breadcrumb": self.breadcrumb}


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.findall(text) if s.strip()]


def _blocks(md: str) -> Iterator[tuple[str, object]]:
    """把一页 md 切成 ('table',[rows]) / ('head',(lvl,title)) / ('para',text)。"""
    paras = re.split(r"\n\s*\n", md)
    for para in paras:
        if "<table" in para.lower():               # HTML 表(含合并单元格)→ 抽行, 表前后残文也保留
            last = 0
            for mt in _HTML_TABLE.finditer(para):
                pre = para[last:mt.start()].strip()
                if pre:
                    yield ("para", pre)
                rows = _html_table_to_pipe(mt.group(0))
                if rows:
                    yield ("table", rows)
                last = mt.end()
            tail = para[last:].strip()
            if tail:
                yield ("para", tail)
            continue
        lines = [l for l in para.split("\n") if l.strip()]
        if not lines:
            continue
        if all(_TABLE_ROW.match(l.strip()) for l in lines):
            yield ("table", [l.strip() for l in lines])
            continue
        # 单行 markdown 标题
        m = _MD_HEAD.match(lines[0].strip())
        if m and len(lines) == 1:
            yield ("head", (len(m.group(1)), m.group(2).strip()))
            continue
        text = " ".join(l.strip() for l in lines)
        # 中文编号纯标题行(无#)
        h = heading_level(text)
        if h and len(text) <= 30:
            yield ("head", (h[0] + 6, text))      # +6 让其级别低于 md # 标题
            continue
        yield ("para", text)


def _table_rows(rows: list[str]) -> tuple[str, list[str]]:
    """返回 (表头行, [数据行...]);跳过 |---| 分隔行。"""
    real = [r for r in rows if not _SEP_ROW.match(r)]
    if not real:
        return "", []
    return real[0], real[1:]


def _extract_title(pages) -> str:
    """取首个 markdown H1 作文档标题;无则回退空。"""
    for pg in pages[:3]:
        for ln in pg["markdown"].split("\n"):
            m = _MD_HEAD.match(ln.strip())
            if m and len(m.group(1)) == 1:
                return m.group(2).strip()
    return ""


def _good_heading(s: str, cap: int) -> bool:
    """干净标题才进 breadcrumb:不超长、无句中标点(滤掉句子型假标题)。"""
    return len(s) <= cap and not re.search(r"[。，；：、？！]", s)


def chunk_doc(doc_id, domain, pages, title=None, target=300, max_chars=400,
              bc_levels=2, bc_doc_title=False, bc_cap=24) -> list[Chunk]:
    if title is None:
        title = _extract_title(pages)
    chunks: list[Chunk] = []
    sec_stack: list[tuple[int, str]] = []
    ci = 0

    def breadcrumb(sp, article):
        levels = [s for s in sp if _good_heading(s, bc_cap)][-bc_levels:]
        parts = ([title.strip()[:bc_cap]] if (bc_doc_title and title) else []) + levels
        if article:
            parts.append(article)
        return "[" + " › ".join(parts) + "]" if parts else ""

    def sec_path():
        return [t for _, t in sec_stack]

    def add_chunk(text, page, typ, article=None):
        nonlocal ci
        sp = sec_path()
        chunks.append(Chunk(f"{doc_id}::c{ci:04d}", doc_id, domain, page, typ,
                            list(sp), breadcrumb(sp, article), article, text, seq=ci))
        ci += 1

    for pg in pages:
        page_no = pg["page_no"]
        for kind, payload in _blocks(_strip_latex(pg["markdown"])):
            if kind == "head":
                lvl, t = cast("tuple[int, str]", payload)
                while sec_stack and sec_stack[-1][0] >= lvl:
                    sec_stack.pop()
                sec_stack.append((lvl, t))
            elif kind == "table":
                header, datarows = _table_rows(cast("list[str]", payload))
                for row in datarows:
                    add_chunk(f"{header}\n{row}", page_no, "table_row")
            else:  # para
                text = cast(str, payload)
                art = detect_article_no(text)
                typ = "clause" if art else "text"
                if len(text) <= max_chars:
                    add_chunk(text, page_no, typ, art)
                else:                              # 过大→按句末聚到 target
                    buf = ""
                    for s in _split_sentences(text):
                        if buf and len(buf) + len(s) > target:
                            add_chunk(buf, page_no, typ, art)
                            buf = s
                        else:
                            buf += s
                    if buf:
                        add_chunk(buf, page_no, typ, art)
    return chunks


# ---------- IO ----------

def read_doc(doc_dir: str):
    man = json.load(open(os.path.join(doc_dir, "manifest.json"), encoding="utf-8"))
    pages = []
    for n in range(1, man["n_pages"] + 1):
        p = os.path.join(doc_dir, f"p{n}.md")
        if os.path.exists(p):
            pages.append({"page_no": n, "markdown": open(p, encoding="utf-8").read()})
    return man, pages


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_dir", help="如 processed_vl/insurance/1")
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()
    man, pages = read_doc(args.doc_dir)
    chunks = chunk_doc(man["doc_id"], man["domain"], pages)
    import collections
    typ = collections.Counter(c.type for c in chunks)
    print(f"{man['doc_id']}: {len(pages)}页 → {len(chunks)}段落块  类型{dict(typ)}")
    lens = [len(c.text) for c in chunks]
    print(f"段落块长度: 均{sum(lens)//max(1,len(lens))} max{max(lens)} 字")
    print(f"\n=== 前{args.show}个段落块 ===")
    for c in chunks[:args.show]:
        print(f"\n[{c.type}|p{c.page_no}|{c.breadcrumb}]\n{c.text[:160]}")
