"""把清洗后的文本行序列结构化为 Block 列表（条款/章节感知）。
法规、合同等条文型文档共用。"""
from __future__ import annotations
from ..schema import Block
from .base import detect_article_no, is_section_heading


def build_blocks(doc_id: str, lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    section_path: list[str] = []
    cur: list[str] | None = None
    cur_article: str | None = None
    cur_type = "text"
    idx = 0

    def flush():
        nonlocal cur, cur_article, cur_type, idx
        if cur:
            text = "\n".join(cur).strip()
            if text:
                blocks.append(Block(
                    block_id=f"{doc_id}#b{idx:04d}",
                    type=cur_type,
                    text=text,
                    section_path=list(section_path),
                    article_no=cur_article,
                ))
                idx += 1
        cur, cur_article, cur_type = None, None, "text"

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if is_section_heading(line) and len(line.strip()) <= 40:
            flush()
            # 维护章/节层级：遇"章"重置到一级，"节"追加二级
            if "章" in line[:6] or "编" in line[:6]:
                section_path[:] = [line.strip()]
            else:
                section_path[:] = section_path[:1] + [line.strip()]
            blocks.append(Block(
                block_id=f"{doc_id}#b{idx:04d}", type="heading",
                text=line.strip(), section_path=list(section_path)))
            idx += 1
            continue
        art = detect_article_no(line)
        if art:
            flush()
            cur = [line.strip()]
            cur_article = art
            cur_type = "article"
        else:
            if cur is None:
                cur, cur_type = [line.strip()], "text"
            else:
                cur.append(line.strip())
    flush()
    return blocks
