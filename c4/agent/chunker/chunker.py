"""结构感知分块：Doc 的 block 序列 -> 检索 chunk。
- 表格独立成块
- 法规短条文按 target_chars 合并，长块按 max_chars 切
- 携带元数据(doc_id/domain/section_path/article_no/page)供检索与回填
- small-to-big：每块记录在文档内的序号，便于父块回填
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from .. import config
from .base import normalize
from ..retriever.filters import clean_pseudo_table, low_value


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    domain: str
    text: str
    seq: int                       # 文档内顺序号（父块回填用）
    type: str = "text"             # text | article | table
    section_path: list[str] = field(default_factory=list)
    article_no: str | None = None
    page: int | None = None
    breadcrumb: str = ""           # [文档名 › 章节 › 第X条] 仅供展示/审核, 不进 BM25 索引

    def to_dict(self) -> dict:
        return asdict(self)


def _breadcrumb(name: str, sp: list[str] | None, art: str | None) -> str:
    """块定位面包屑：文档名 › 章节路径 › 条号。去重、去空。"""
    parts: list[str] = []
    if name:
        parts.append(name)
    seen = set(parts)
    for s in (sp or []):
        s = (s or "").strip()
        if s and s not in seen:
            parts.append(s)
            seen.add(s)
    if art and art not in seen:
        parts.append(art)
    return " › ".join(parts)


def chunk_doc(doc: dict) -> list[Chunk]:
    c = config.load()["chunker"]
    nz = config.load().get("normalize", {})
    tc = config.load().get("table_clean", {})
    drop_lv = c.get("drop_low_value", True)
    target = c["target_chars"]
    maxc = c["max_chars"]
    minc = c["min_chars"]
    table_as_chunk = c.get("table_as_chunk", True)

    doc_id = doc["doc_id"]
    domain = doc["domain"]
    bc_on = c.get("breadcrumb", True)
    doc_name = ""
    if bc_on:
        from ..agentic.outline import clean_doc_name
        doc_name = clean_doc_name(doc)
    chunks: list[Chunk] = []
    seq = 0

    def emit(text, typ, sp, art, page):
        nonlocal seq
        text = normalize(text, nz.get("strip_cjk_spaces", True),
                         nz.get("collapse_spaces", True))
        if not text:
            return
        # 伪表清洗：释义/正文被当表 -> 去管道符并重分类
        if typ == "table" and tc.get("enabled", True):
            cleaned, reclassed = clean_pseudo_table(text, tc.get("reclass_min_digits", 5))
            if reclassed:
                text, typ = cleaned, "text"
        # 丢弃目录/空壳/碎片
        if drop_lv and low_value(text):
            return
        # 统一强制 max_chars 切分（防某些块未经 _split_long 而超长）
        for piece in _split_long(text, maxc):
            if drop_lv and low_value(piece):
                continue
            chunks.append(Chunk(chunk_id=f"{doc_id}::c{seq:04d}", doc_id=doc_id,
                                domain=domain, text=piece, seq=seq, type=typ,
                                section_path=sp or [], article_no=art, page=page,
                                breadcrumb=_breadcrumb(doc_name, sp, art) if bc_on else ""))
            seq += 1

    buf: list[str] = []
    buf_meta = None   # (section_path, article_no, page)

    def flush_buf():
        nonlocal buf, buf_meta
        if buf and buf_meta:
            emit("\n".join(buf), "text", *buf_meta)
        buf, buf_meta = [], None

    for b in doc["blocks"]:
        btype = b["type"]
        text = b.get("text", "")
        sp = b.get("section_path", [])
        art = b.get("article_no")
        page = b.get("page")

        if btype == "table" and table_as_chunk:
            flush_buf()
            # 大表按 max 切
            for piece in _split_long(text, maxc):
                emit(piece, "table", sp, art, page)
            continue

        if btype == "heading":
            continue  # 标题并入后续块的 section_path，不单独成块

        if btype == "article":
            flush_buf()
            for piece in _split_long(text, maxc):
                emit(piece, "article", sp, art, page)
            continue

        # 普通正文：按 target 聚合
        if buf_meta is None:
            buf_meta = (sp, art, page)
        buf.append(text)
        if sum(len(x) for x in buf) >= target:
            flush_buf()
    flush_buf()

    if c.get("merge_short_articles", True):
        chunks = _merge_short(chunks, target, minc)
    return chunks


def _split_long(text: str, maxc: int) -> list[str]:
    if len(text) <= maxc:
        return [text]
    out, cur = [], ""
    for line in text.split("\n"):
        # 单行超长（如压扁的财报表/附录）：硬切
        while len(line) > maxc:
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:maxc])
            line = line[maxc:]
        if len(cur) + len(line) > maxc and cur:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def _merge_short(chunks: list[Chunk], target: int, minc: int) -> list[Chunk]:
    """相邻同类型(article)的过短块合并，减少碎片。"""
    out: list[Chunk] = []
    for ch in chunks:
        if (out and ch.type == "article" and out[-1].type == "article"
                and len(out[-1].text) < target and len(ch.text) < minc * 3
                and out[-1].doc_id == ch.doc_id):
            out[-1].text += "\n" + ch.text
        else:
            out.append(ch)
    return out
