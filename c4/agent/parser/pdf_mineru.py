"""MinerU fallback 解析器。强于 PyMuPDF 的场景：扫描件OCR、跨页表、合并单元格、多栏。
仅在质量门控触发时调用，避免对全库跑重型流程。

依赖（需在运行环境预装，B 榜前就绪）：
    pip install mineru   # 或 magic-pdf
MinerU 输出 markdown + content_list.json，这里转成统一 Doc。
未安装时抛 MinerUNotAvailable，registry 据此优雅降级到 OCR/保留 PyMuPDF 结果。
"""
from __future__ import annotations
import os
from .base import Parser
from .structure import build_blocks
from ..schema import Doc, Block, Table


class MinerUNotAvailable(RuntimeError):
    pass


def _check_available():
    try:
        import mineru  # noqa: F401
        return "mineru"
    except Exception:
        pass
    try:
        import magic_pdf  # noqa: F401
        return "magic_pdf"
    except Exception:
        pass
    raise MinerUNotAvailable(
        "MinerU 未安装。运行前执行 `pip install mineru` 并下载模型；"
        "在此之前 fallback 链会保留 PyMuPDF 结果。")


class MinerUParser(Parser):
    def __init__(self, domain: str = "", out_root: str = "processed/_mineru"):
        self.domain = domain
        self.out_root = out_root

    def parse(self, doc_id: str, path: str) -> Doc:
        backend = _check_available()
        md_path, content_list = self._run(doc_id, path, backend)
        return self._to_doc(doc_id, path, md_path, content_list)

    # --- MinerU 调用：用已验证的 CLI(pipeline 后端, 中文)。---
    # 实测 MinerU 3.3.1 输出: {out}/{doc_id}/auto/{doc_id}.md + _content_list.json
    def _run(self, doc_id: str, path: str, backend: str):
        out_dir = os.path.join(self.out_root, doc_id)
        os.makedirs(out_dir, exist_ok=True)
        env = "MINERU_MODEL_SOURCE=modelscope "
        rc = os.system(f'{env}mineru -p "{path}" -o "{out_dir}" -b pipeline -l ch '
                       f'> "{out_dir}/mineru.log" 2>&1')
        if rc != 0:
            raise RuntimeError(f"mineru CLI failed rc={rc}, see {out_dir}/mineru.log")
        md_path = _find(out_dir, ".md")
        content_list = _find(out_dir, "_content_list.json")
        return md_path, content_list

    def _to_doc(self, doc_id, path, md_path, content_list) -> Doc:
        import json
        blocks: list[Block] = []
        if content_list and os.path.exists(content_list):
            ti = 0
            items = json.load(open(content_list, encoding="utf-8"))
            txt_lines: list[str] = []
            for it in items:
                t = it.get("type")
                if t == "table":
                    md = it.get("table_body") or it.get("text", "")
                    blocks.append(Block(block_id=f"{doc_id}#t{ti:03d}", type="table",
                                        text=md, page=it.get("page_idx"),
                                        table=Table(markdown=md,
                                                    caption=it.get("table_caption", ""))))
                    ti += 1
                elif t in ("text", "title"):
                    txt = it.get("text", "")
                    if txt:
                        txt_lines.append(txt)
            blocks.extend(build_blocks(doc_id, txt_lines))
        elif md_path and os.path.exists(md_path):
            blocks = build_blocks(doc_id, open(md_path, encoding="utf-8").read().splitlines())
        title = next((b.text[:40] for b in blocks if b.type != "table"), doc_id)
        return Doc(doc_id=doc_id, domain=self.domain, title=title,
                   source_path=path, blocks=blocks)


def _find(root: str, suffix: str) -> str | None:
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(suffix):
                return os.path.join(r, f)
    return None
