"""per-doc 大纲 + 干净文档名（plan.md §v4.3 阶段1）。
- clean_doc_name: 从噪声 title/首块里抽出可读的产品/文档名（条款名/办法名/公告名）。
  raw title 对保险/财报常是首行地址/页码/"阅读指引" → 不可直接用。
- build_outline: 每篇 doc -> {doc_id, title, name, toc, headings}，喂给分解器当"地图"。
纯规则零模型；名字抽不准只影响展示(面包屑)与分解提示，不影响 BM25 排序。"""
from __future__ import annotations
import os
import re
import json
import glob
from .. import config

# 像"标题"的文档/产品/法名特征（而非正文句）
_NAMEY = re.compile(
    r"(寿险|养老年金|年金保险|保险|基金|债券|办法|规定|规则|细则|准则|章程|指引|"
    r"公告|决定书|年度报告|年报|报告|募集说明书|招股说明书|《.+》)")
# 句子特征：命中说明这是正文而非标题（标题极少含句读/人称/连接词）。半角全角都要。
_SENT = re.compile(r"[，。；！？、：,;!?]|我们|您|如果|根据|依据|以下简称|本合同|被保险人|提供")
# 明显不是文档名的结构词
_BLOCKLIST = re.compile(r"^(阅读指引|目录|释义|重要提示|重要事项|特别提示|声明|前言|引言)$")
# 页码/页眉残留："（第〇页）" "-1-" "第3页" 等
_PAGE_TAIL = re.compile(r"[（(]\s*第?[〇零一二三四五六七八九十百0-9]+\s*页\s*[)）]\s*$")
# 墙文本里"{产品名}阅读指引/产品/条款/提供..."：抽产品名前缀
_LEAD = re.compile(r"^(.{4,36}?(?:保险|寿险|养老年金|年金保险|基金))"
                   r"(?:阅读指引|产品|条款|提供|合同|利益条款|计划)")
_PUA = re.compile(r"[-]")    # 私有区符号(图标/项目符)


def _norm_line(s: str) -> str:
    s = _PUA.sub("", s)
    s = re.sub(r"\s+", "", s.strip())    # 标题/名字里的空格都是 PDF 伪空格, 全清
    s = _PAGE_TAIL.sub("", s).strip()
    return s


def clean_doc_name(doc: dict) -> str:
    """抽"文档/产品名"。只认像标题的行(有产品/法名特征、无句读)；没把握回 ""
    (空字符串 → 面包屑只用定位, 比塞个错名更利于审核)。
    优先 title 字段(最可能是真标题), 再看前几块; 标题只在最前面, 深入会抓到条标题。"""
    a = config.load().get("agentic", {})
    scan = a.get("outline_scan_blocks", 3)
    minlen, maxlen = a.get("outline_name_minlen", 6), a.get("outline_name_maxlen", 36)
    title_lines = (doc.get("title", "") or "").split("\n")
    block_lines = [ln for b in doc.get("blocks", [])[:scan]
                   for ln in (b.get("text", "") or "").split("\n")]
    for raw in title_lines + block_lines:
        if "|" in raw:                                      # 表格行
            continue
        ln = _norm_line(raw)
        if not (minlen <= len(ln) <= maxlen) or _BLOCKLIST.match(ln):
            continue                                        # ≥minlen 字: 滤掉"养老年金/年度报告"等泛词碎片
        if _NAMEY.search(ln) and not _SENT.search(ln):      # 像标题、不像句子
            return ln
    # 墙文本首块抽 "{产品名}阅读指引/条款/..." 前缀
    b0 = (doc.get("blocks") or [{}])[0].get("text", "") or ""
    if "|" not in b0[:80]:                                  # 非表格首块才抽
        m = _LEAD.match(_norm_line(b0[:80]))
        if m:
            return m.group(1)
    return ""


def build_outline(doc: dict) -> dict:
    """一篇 doc 的地图：标题树 + 条文目录。供分解器把子问题绑到章节/文档。"""
    name = clean_doc_name(doc)
    toc: list[dict] = []          # 条文/标题级目录
    headings: list[str] = []
    for b in doc.get("blocks", []):
        if b.get("type") == "heading":
            h = _norm_line(b.get("text", ""))
            if h:
                headings.append(h)
        art = b.get("article_no")
        if art:
            head = _norm_line((b.get("text", "") or "").split("\n")[0])[:40]
            toc.append({"article_no": art, "head": head, "page": b.get("page")})
    return {"doc_id": doc["doc_id"], "domain": doc.get("domain"),
            "name": name, "title": (doc.get("title", "") or "")[:80],
            "headings": list(dict.fromkeys(headings)), "toc": toc}


def build_all(processed_dir: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in glob.glob(os.path.join(processed_dir, "*", "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        out[d["doc_id"]] = build_outline(d)
    return out
