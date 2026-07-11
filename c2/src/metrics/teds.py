# -*- coding: utf-8 -*-
"""TEDS（Tree Edit Distance based Similarity）表格结构相似度。

官方 TEDS = 1 − 树编辑距离 / 节点数，∈[0,1]（官方再 ×100）。

本赛题表格特点（实测）：纯网格 <tr><td>，**无 colspan/rowspan**，单表最多 3.4 万单元格。
标准 APTED（≈O(n³)）在万级节点会卡死，故：
  - flat grid（无跨行跨列）→ 走快速 `teds_grid`：行序列对齐 + 单元格级 DP（rapidfuzz C 级 cdist 加速）；
    对规则网格而言与真 TEDS 几乎等价，且可在秒级处理万级单元格。
  - 小型/含跨行跨列表格（节点 < 阈值）→ 走精确 APTED。
解析不出表格时返回相应缺省（见各函数说明）。
"""
import numpy as np
import lxml.html as lhtml
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein as _Lev

_APTED_NODE_LIMIT = 1200          # 节点数超过此值不走 APTED（太慢）


# ===========================================================================
# HTML 解析
# ===========================================================================
def _first_table(html_str):
    """解析字符串里的第一个 <table> 元素，失败返回 None。"""
    html_str = (html_str or "").strip()
    if "<table" not in html_str.lower():
        return None
    try:
        frag = lhtml.fromstring("<div>" + html_str + "</div>")
    except Exception:
        return None
    tables = frag.xpath("//table")
    return tables[0] if tables else None


def _parse_grid(el):
    """把 table 元素解析成二维网格：List[row]，row = List[cell_text]。"""
    rows = []
    for tr in el.xpath(".//tr"):
        cells = []
        for td in tr.xpath("./td|./th"):
            txt = " ".join("".join(td.itertext()).split())
            cells.append(txt)
        rows.append(cells)
    return rows


def _has_span(el):
    """是否存在 colspan/rowspan > 1。"""
    for c in el.xpath(".//td|.//th"):
        if int(c.get("colspan", 1)) > 1 or int(c.get("rowspan", 1)) > 1:
            return True
    return False


def _grid_nodes(rows):
    """网格树节点数：1(table) + 行数 + 单元格总数。"""
    return 1 + len(rows) + sum(len(r) for r in rows)


# ===========================================================================
# 快速网格 TEDS
# ===========================================================================
def _cellseq_dist(a, b):
    """两行单元格序列的编辑距离：
    rename 代价 = 单元格文本归一化 Levenshtein ∈[0,1]；插入/删除单元格代价 = 1。
    用 rapidfuzz.cdist 在 C 层算出 |a|×|b| 代价矩阵，再跑小型 DP。
    """
    la, lb = len(a), len(b)
    if la == 0:
        return float(lb)
    if lb == 0:
        return float(la)
    # 代价矩阵：归一化编辑距离（相同文本=0，完全不同=1）
    M = process.cdist(a, b, scorer=_Lev.normalized_distance, workers=1)
    dp = np.empty((la + 1, lb + 1))
    dp[:, 0] = np.arange(la + 1)
    dp[0, :] = np.arange(lb + 1)
    for i in range(1, la + 1):
        Mi = M[i - 1]
        di = dp[i]
        dim1 = dp[i - 1]
        for j in range(1, lb + 1):
            di[j] = min(dim1[j] + 1.0, di[j - 1] + 1.0, dim1[j - 1] + Mi[j - 1])
    return float(dp[la, lb])


def teds_grid(pred_rows, gt_rows, band=None):
    """规则网格的 TEDS ∈[0,1]（深度2树编辑距离的带状DP，对齐 APTED 口径）。

    行级编辑距离：替换两行(tr匹配)=单元格级编辑距离 `_cellseq_dist`；
    删/插整行 = 1 + 该行单元格数（tr 节点 + 叶子）。**按内容做最优行对齐**——
    旧版用"行签名精确匹配"，因 OCR/列数差异行签名永不相等 → 退化成按位置配对、
    行数不等时整体错位 → TEDS 被严重低估（实测 0.37 应为 ~0.91）。
    带状(|i-j|≤band)把复杂度降到 O(R·band)，行大致有序时与全 DP 等价。
    """
    n = max(_grid_nodes(pred_rows), _grid_nodes(gt_rows))
    if n == 0:
        return 1.0
    Rp, Rg = len(pred_rows), len(gt_rows)
    if Rp == 0 or Rg == 0:
        return 0.0
    if band is None:
        band = max(8, abs(Rp - Rg) + 6)

    INF = float("inf")
    pw = [1 + len(r) for r in pred_rows]          # 删 pred 行代价
    gw = [1 + len(r) for r in gt_rows]            # 插 gt 行代价
    prev = [0.0] * (Rg + 1)
    for j in range(1, Rg + 1):
        prev[j] = prev[j - 1] + gw[j - 1]
    for i in range(1, Rp + 1):
        cur = [INF] * (Rg + 1)
        lo = max(0, i - band)
        hi = min(Rg, i + band)
        if lo == 0:
            cur[0] = prev[0] + pw[i - 1]          # 前 i 行全删
        for j in range(max(1, lo), hi + 1):
            best = prev[j] + pw[i - 1]            # 删 pred 行 i
            if cur[j - 1] + gw[j - 1] < best:     # 插 gt 行 j
                best = cur[j - 1] + gw[j - 1]
            sub = prev[j - 1] + _cellseq_dist(pred_rows[i - 1], gt_rows[j - 1])
            if sub < best:
                best = sub
            cur[j] = best
        prev = cur
    return max(0.0, 1.0 - prev[Rg] / n)


# ===========================================================================
# 精确 APTED（仅小型 / 含跨行跨列表格）
# ===========================================================================
def _teds_apted(pred_el, gt_el):
    from apted import APTED, Config

    class TableTree:
        def __init__(self, tag, colspan, rowspan, content, children):
            self.tag, self.colspan, self.rowspan = tag, colspan, rowspan
            self.content, self.children = content, children

    def build(el):
        tag = el.tag
        if tag in ("td", "th"):
            txt = " ".join("".join(el.itertext()).split())
            return TableTree("td", int(el.get("colspan", 1)),
                             int(el.get("rowspan", 1)), txt, [])
        kids = [build(ch) for ch in el if ch.tag in
                ("table", "thead", "tbody", "tr", "td", "th")]
        norm = "tr" if tag == "tr" else "table"
        return TableTree(norm, None, None, None, kids)

    def count(n):
        return 1 + sum(count(c) for c in n.children)

    class Cfg(Config):
        def rename(self, a, b):
            if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
                return 1.0
            if a.tag == "td" and (a.content or b.content):
                m = max(len(a.content or ""), len(b.content or ""))
                return _Lev.distance(a.content or "", b.content or "") / m if m else 0.0
            return 0.0
        def children(self, node):
            return node.children

    tp, tg = build(pred_el), build(gt_el)
    n = max(count(tp), count(tg))
    if n == 0:
        return 1.0
    d = APTED(tp, tg, Cfg()).compute_edit_distance()
    return max(0.0, 1.0 - d / n)


# ===========================================================================
# 统一入口
# ===========================================================================
def teds_score(pred_html, gt_html):
    """单表 TEDS ∈[0,1]。GT 无表格→None（不计入）；该出表却没出→0.0。"""
    gt_el = _first_table(gt_html)
    if gt_el is None:
        return None
    pred_el = _first_table(pred_html)
    if pred_el is None:
        return 0.0

    gt_rows = _parse_grid(gt_el)
    pred_rows = _parse_grid(pred_el)
    n = max(_grid_nodes(gt_rows), _grid_nodes(pred_rows))

    spanned = _has_span(gt_el) or _has_span(pred_el)
    if spanned and n <= _APTED_NODE_LIMIT:
        return _teds_apted(pred_el, gt_el)        # 小型含跨格：精确
    return teds_grid(pred_rows, gt_rows)          # 网格（主力路径）：快速
