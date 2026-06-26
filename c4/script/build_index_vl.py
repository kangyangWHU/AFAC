"""用新 parse+chunk 建 BM25 索引:processed_vl/*/* → 段落块 → BM25Index → index/bm25_vl.pkl。
检索/judge 复用现有 agent.index.bm25 + agentic。
用法:
  python -m script.build_index_vl              # 全部已解析文档
  python -m script.build_index_vl --domain insurance
"""
from __future__ import annotations
import os
import sys
import glob
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json                                           # noqa: E402
from agent.vl.chunker import read_doc, chunk_doc, _extract_title  # noqa: E402
from agent.index.bm25 import BM25Index                # noqa: E402
from agent import config                              # noqa: E402

PROC = os.path.join(ROOT, "processed_vl")
OUT = os.path.join(ROOT, "processed_vl", "bm25_vl.pkl")
OUT_OUTLINE = os.path.join(ROOT, "processed_vl", "outlines_vl.json")


def _doc_outline(doc_id, domain, title, chunks) -> dict:
    """从新 chunk 造分解器要的"文档地图"(等价 outline.py 产物)。"""
    headings, seen = [], set()
    for c in chunks:
        for h in c.section_path:
            if h not in seen:
                seen.add(h); headings.append(h)
    toc = [{"article_no": c.article_no, "head": c.text.split("\n")[0][:40], "page": c.page_no}
           for c in chunks if c.type == "clause" and c.article_no]
    return {"doc_id": doc_id, "domain": domain, "name": title, "title": title,
            "headings": headings, "toc": toc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    v = config.load().get("vl", {})
    kw = dict(target=v.get("target_chars", 300), max_chars=v.get("max_chars", 400),
              bc_levels=v.get("bc_levels", 2), bc_doc_title=v.get("bc_doc_title", False),
              bc_cap=v.get("bc_cap", 24))

    pat = os.path.join(PROC, args.domain or "*", "*", "manifest.json")
    dirs = sorted(os.path.dirname(p) for p in glob.glob(pat))
    all_chunks, outlines, n_doc = [], {}, 0
    t0 = time.time()
    for d in dirs:
        man, pages = read_doc(d)
        title = _extract_title(pages)
        chunks = chunk_doc(man["doc_id"], man["domain"], pages, title=title, **kw)
        all_chunks.extend(c.to_dict() for c in chunks)
        outlines[man["doc_id"]] = _doc_outline(man["doc_id"], man["domain"], title, chunks)
        n_doc += 1

    idx = BM25Index().build(all_chunks)
    idx.save(args.out)
    json.dump(outlines, open(OUT_OUTLINE, "w"), ensure_ascii=False)
    import collections
    typ = collections.Counter(c["type"] for c in all_chunks)
    print(f"{n_doc} 篇 → {len(all_chunks)} 段落块  类型{dict(typ)}")
    print(f"index_breadcrumb={idx.index_breadcrumb}  {time.time()-t0:.1f}s")
    print(f"-> {args.out}\n-> {OUT_OUTLINE} ({len(outlines)} 篇大纲)")


if __name__ == "__main__":
    main()
