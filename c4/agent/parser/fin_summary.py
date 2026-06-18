"""财报"主要会计数据"结构化抽取：营收/净利/现金流等关键值 -> 干净 key-value 文本块。
优先用 MinerU 的干净 HTML 表（processed/_mineru_reports/<doc>/auto/<doc>.md），
回退到 PyMuPDF 的 markdown 表。规则解析，零模型，泛化到所有财报数值题。
"""
from __future__ import annotations
import os
import re
from bs4 import BeautifulSoup

_KEYS = ["营业总收入", "营业收入", "归属于上市公司股东的净利润",
         "归属于上市公司股东的扣除非经常性损益的净利润", "经营活动产生的现金流量净额",
         "基本每股收益", "加权平均净资产收益率", "归属于上市公司股东的净资产", "资产总额"]
_MINERU_DIR = "processed/_mineru_reports"


def _rows_to_summary(rows: list[list[str]]) -> str | None:
    if not rows:
        return None
    hdr = " ".join(rows[0])
    years = re.findall(r"(20\d\d)\s*年", hdr)
    if "营业收入" not in " ".join(" ".join(r) for r in rows) or not years:
        return None
    out = []
    for r in rows[1:]:
        if not r:
            continue
        label = re.sub(r"（.*?）|\(.*?\)", "", r[0]).strip()
        key = next((k for k in _KEYS if k in label), None)
        if not key:
            continue
        vals = [c for c in r[1:] if c and "%" not in c
                and re.fullmatch(r"-?[\d,]+\.?\d*", c.replace(",", ""))]
        pct = next((c for c in r[1:] if "%" in c), "")
        parts = [f"{years[i]}年={vals[i]}" for i in range(min(len(years), len(vals)))]
        if pct:
            parts.append(f"同比={pct}")
        if parts:
            out.append(f"{key}: " + ", ".join(parts))
    return "【主要会计数据(结构化)】\n" + "\n".join(out) if len(out) >= 3 else None


def from_mineru(doc_id: str, root: str = ".") -> str | None:
    md_path = os.path.join(root, _MINERU_DIR, doc_id, "auto", f"{doc_id}.md")
    if not os.path.exists(md_path):
        return None
    md = open(md_path, encoding="utf-8").read()
    for tb in re.findall(r"<table>.*?</table>", md, re.S):
        if "营业收入" not in tb:
            continue
        soup = BeautifulSoup(tb, "html.parser")
        rows = [[td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                for tr in soup.find_all("tr")]
        s = _rows_to_summary(rows)
        if s:
            return s
    return None


def from_pymupdf(doc: dict) -> str | None:
    """回退：PyMuPDF markdown 表（脏，可能失败）。"""
    for b in doc["blocks"]:
        if b["type"] != "table" or "营业收入" not in b["text"]:
            continue
        if not re.search(r"增减|同比", b["text"]):
            continue
        rows = [[c.strip() for c in ln.split("|") if c.strip()]
                for ln in b["text"].split("\n")]
        rows = [r for r in rows if r and "---" not in r[0]]
        s = _rows_to_summary(rows)
        if s:
            return s
    return None


def extract_summary(doc: dict, root: str = ".") -> str | None:
    return from_mineru(doc["doc_id"], root) or from_pymupdf(doc)
