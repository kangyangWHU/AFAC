"""预处理驱动：解析文档 -> processed/{domain}/{doc_id}.json，并出验收报告。

用法：
  python -m script.preprocess                # 仅解析 A 榜题目涉及的文档（快速验收）
  python -m script.preprocess --all          # 解析全库（B 榜需要）
  python -m script.preprocess --domain regulatory
"""
from __future__ import annotations
import os
import sys
import json
import glob
import argparse
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agent.doc_index import build_doc_index, DOMAINS          # noqa: E402
from agent.parser.registry import get_parser                  # noqa: E402
from agent.chunker.base import normalize                      # noqa: E402
from agent import config                                      # noqa: E402

RAW = os.path.join(ROOT, "data", "public_dataset_upload", "raw")
QDIR = os.path.join(ROOT, "data", "public_dataset_upload", "questions", "group_a")
PROC = os.path.join(ROOT, "processed")
LOGS = os.path.join(ROOT, "logs")


def question_doc_ids() -> set[str]:
    ids: set[str] = set()
    for f in glob.glob(os.path.join(QDIR, "*.json")):
        for q in json.load(open(f, encoding="utf-8")):
            ids.update(q.get("doc_ids", []))
    return ids


# 已知题目证据原文片段 -> 回指校验（doc 解析有效性的硬证据）
EVIDENCE_PROBES = {
    "strict_csrc_035": "为资产负债率超过百分之七十的担保对象提供的担保",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="解析全库")
    ap.add_argument("--domain", default=None)
    args = ap.parse_args()

    index = build_doc_index(RAW)
    if args.all:
        targets = set(index)
    else:
        targets = question_doc_ids()
    if args.domain:
        targets = {d for d in targets if index.get(d, {}).get("domain") == args.domain}

    os.makedirs(LOGS, exist_ok=True)
    report = []
    ok = fail = 0
    _NZ = config.load().get("normalize", {})
    for doc_id in sorted(targets):
        meta = index.get(doc_id)
        if not meta:
            report.append([doc_id, "?", "MISSING", 0, 0, 0, ""])
            fail += 1
            continue
        domain, ext, path = meta["domain"], meta["ext"], meta["path"]
        try:
            parser = get_parser(domain, ext)
            doc = parser.parse(doc_id, path)
            for b in doc.blocks:                          # 解析即归一: 去 PDF 伪空格(中文/数字间), 让 processed/ 即干净源
                b.text = normalize(b.text, _NZ.get("strip_cjk_spaces", True), _NZ.get("collapse_spaces", True))
            outdir = os.path.join(PROC, domain)
            os.makedirs(outdir, exist_ok=True)
            json.dump(doc.to_dict(), open(os.path.join(outdir, f"{doc_id}.json"), "w"),
                      ensure_ascii=False, indent=1)
            n_blk = len(doc.blocks)
            n_art = sum(1 for b in doc.blocks if b.article_no)
            n_tab = sum(1 for b in doc.blocks if b.type == "table")
            clen = doc.char_len()
            probe = ""
            if doc_id in EVIDENCE_PROBES:
                hit = EVIDENCE_PROBES[doc_id] in "".join(b.text for b in doc.blocks)
                probe = "HIT" if hit else "MISS"
            status = "OK" if clen > 50 else "EMPTY"
            report.append([doc_id, domain, status, clen, n_blk, n_art, n_tab, probe])
            ok += 1
        except Exception as e:
            report.append([doc_id, domain, f"ERR:{e}", 0, 0, 0, 0, ""])
            fail += 1

    rpath = os.path.join(LOGS, "preprocess_report.csv")
    with open(rpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "domain", "status", "char_len", "n_blocks",
                    "n_articles", "n_tables", "probe"])
        w.writerows(report)

    print(f"parsed {ok} ok / {fail} fail -> {PROC}")
    print(f"report -> {rpath}")
    # 汇总
    empt = [r for r in report if isinstance(r[2], str) and r[2] not in ("OK",)]
    if empt:
        print(f"非OK文档 {len(empt)} 个：")
        for r in empt[:20]:
            print("  ", r[0], r[2])
    probes = [r for r in report if len(r) > 7 and r[7]]
    for r in probes:
        print(f"回指校验 {r[0]}: {r[7]}")


if __name__ == "__main__":
    main()
