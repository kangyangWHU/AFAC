"""把清洗后的文本行序列结构化为 Block 列表（条款/章节感知）。
法规、合同等条文型文档共用。"""
from __future__ import annotations
from ..schema import Block
from .base import detect_article_no, heading_level


def build_blocks(doc_id: str, lines: list[str]) -> list[Block]:
    blocks: list[Block] = []
    section_path: list[str] = []
    sec_stack: list[tuple[int, str]] = []          # (层级, 标题), 同级替换式嵌套
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
        h = heading_level(line)
        if h:
            lvl, htext = h
            flush()
            # 层级栈：弹出同级及更深，再压入 → 同级标题相互替换而非堆叠
            while sec_stack and sec_stack[-1][0] >= lvl:
                sec_stack.pop()
            sec_stack.append((lvl, htext))
            section_path[:] = [t for _, t in sec_stack]
            blocks.append(Block(
                block_id=f"{doc_id}#b{idx:04d}", type="heading",
                text=htext, section_path=list(section_path)))
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
