"""扫描 raw/ 构建 doc_id -> 文件 的全量索引（A 榜题目 doc 子集 + B 榜全库）。
各域 doc_id 命名不同，这里统一解析。"""
from __future__ import annotations
import os
import json

DOMAINS = ["insurance", "financial_contracts", "financial_reports",
           "regulatory", "research"]


def build_doc_index(raw_root: str) -> dict[str, dict]:
    """返回 {doc_id: {path, domain, ext}}。doc_id = 文件名去扩展名。"""
    index: dict[str, dict] = {}
    for domain in DOMAINS:
        ddir = os.path.join(raw_root, domain)
        if not os.path.isdir(ddir):
            continue
        for root, _, files in os.walk(ddir):
            for fn in files:
                if fn.startswith("."):
                    continue
                base, ext = os.path.splitext(fn)
                ext = ext.lower().lstrip(".")
                if ext not in {"pdf", "html", "htm", "txt"}:
                    continue
                path = os.path.join(root, fn)
                # 同名冲突时优先非附件目录
                if base in index and "attachments" in path:
                    continue
                index[base] = {"path": path, "domain": domain, "ext": ext}
    return index


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_root = os.path.join(here, "data", "public_dataset_upload", "raw")
    idx = build_doc_index(raw_root)
    out = os.path.join(here, "doc_index.json")
    json.dump(idx, open(out, "w"), ensure_ascii=False, indent=1)
    by_dom: dict[str, int] = {}
    for v in idx.values():
        by_dom[v["domain"]] = by_dom.get(v["domain"], 0) + 1
    print(f"doc_index: {len(idx)} docs -> {out}")
    for d, n in sorted(by_dom.items()):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    main()
