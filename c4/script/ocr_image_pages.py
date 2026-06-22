"""逐页 OCR enrichment（plan §v4）：把图片页/封面页用 MinerU OCR 出来，按位置插回 processed JSON。
混合策略（用户设计）：PyMuPDF 抽文字页(快、留图表文字)，MinerU 只补图片页(封面标题/扫描内容)。
效率：MinerU CLI 每次重载模型 ~50s，所以把【所有 doc 的图片页】抽成单页 PDF 放一个目录，MinerU 批跑一次。

用法：python -m script.ocr_image_pages [--domain D] [--limit N] [--max-pages-per-doc 12]
"""
from __future__ import annotations
import os
import sys
import json
import glob
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config                       # noqa: E402

BATCH_DIR = "/tmp/ocr_batch"
TXT_THRESH = 80          # 页文字 < 此 且 有图 → 图片页, 该 OCR


def detect_image_pages(fitz_doc, max_pages: int) -> list[int]:
    """需要 OCR 的页：①封面 p0 只要有图就 OCR(标题常是图形, 文字判据抓不到)；
    ②其余低文字但有图/矢量绘制的页(图表/扫描页)。总数封顶。"""
    out = []
    for i, pg in enumerate(fitz_doc):
        txt = len(pg.get_text("text").strip())
        has_graphic = bool(pg.get_images()) or len(pg.get_drawings()) > 20
        if has_graphic and (txt < TXT_THRESH or i == 0):
            out.append(i)
    return out[:max_pages]


def main():
    import fitz
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-pages-per-doc", type=int, default=12)
    args = ap.parse_args()

    di = json.load(open(os.path.join(ROOT, config.path("doc_index")), encoding="utf-8"))
    shutil.rmtree(BATCH_DIR, ignore_errors=True)
    os.makedirs(BATCH_DIR, exist_ok=True)

    # 1) 检测图片页 + 抽成单页 PDF（命名 {doc_id}__p{page}.pdf 便于回填）
    jobs: dict[str, list[int]] = {}
    items = [(k, v) for k, v in di.items() if v["ext"] == "pdf"
             and (not args.domain or v["domain"] == args.domain)]
    if args.limit:
        items = items[:args.limit]
    for doc_id, v in items:
        try:
            d = fitz.open(v["path"])
        except Exception:
            continue
        pages = detect_image_pages(d, args.max_pages_per_doc)
        if not pages:
            continue
        jobs[doc_id] = pages
        for p in pages:
            sub = fitz.open()
            sub.insert_pdf(d, from_page=p, to_page=p)
            sub.save(os.path.join(BATCH_DIR, f"{doc_id}__p{p}.pdf"))
    npages = sum(len(p) for p in jobs.values())
    print(f"图片页: {len(jobs)} 篇 / {npages} 页 → 批量 MinerU")
    if not jobs:
        return

    # 2) MinerU 批跑一次（模型只加载一次）
    out_root = os.path.join(BATCH_DIR, "_out")
    rc = os.system(f'MINERU_MODEL_SOURCE=modelscope mineru -p "{BATCH_DIR}" -o "{out_root}" '
                   f'-b pipeline -l ch > "{BATCH_DIR}/mineru.log" 2>&1')
    if rc != 0:
        print(f"MinerU 批跑失败 rc={rc}, 见 {BATCH_DIR}/mineru.log")
        return

    # 3) 回填：每页 OCR 文本作为 ocr 块, 按页号插回 processed JSON
    proc = config.path("processed_dir")
    merged = 0
    for doc_id, pages in jobs.items():
        dom = di[doc_id]["domain"]
        pj = os.path.join(ROOT, proc, dom, f"{doc_id}.json")
        if not os.path.exists(pj):
            continue
        doc = json.load(open(pj, encoding="utf-8"))
        ocr_blocks = []
        for p in pages:
            cl = _find_content_list(out_root, f"{doc_id}__p{p}")
            if not cl:
                continue
            txt = "\n".join(it.get("text", "") for it in json.load(open(cl, encoding="utf-8"))
                            if it.get("type") in ("text", "title") and it.get("text"))
            if txt.strip():
                ocr_blocks.append({"block_id": f"{doc_id}#ocr_p{p}", "type": "text",
                                   "text": txt, "section_path": [], "article_no": None,
                                   "page": p + 1})
        if not ocr_blocks:
            continue
        # 按 page 插回（OCR 页内容放到该页区域；封面 p0 自然落最前 → clean_doc_name 拿到标题）
        doc["blocks"] = _insert_by_page(doc["blocks"], ocr_blocks)
        json.dump(doc, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        merged += 1
    print(f"回填完成: {merged} 篇 processed JSON 已加 OCR 块。记得 build_index 重建。")


def _find_content_list(out_root: str, stem: str) -> str | None:
    hits = glob.glob(os.path.join(out_root, stem, "**", "*_content_list.json"), recursive=True)
    return hits[0] if hits else None


def _insert_by_page(blocks: list[dict], ocr_blocks: list[dict]) -> list[dict]:
    """把 OCR 块按 page 插到原 blocks 里（稳定：OCR 页 P 放到第一个 page>P 的块之前）。"""
    out = list(blocks)
    for ob in ocr_blocks:
        p = ob["page"]
        idx = next((i for i, b in enumerate(out) if (b.get("page") or 0) > p), len(out))
        out.insert(idx, ob)
    return out


if __name__ == "__main__":
    main()
