# -*- coding: utf-8 -*-
"""LONG 路表格结构自愈：满宽横幅行的 colspan 归一。

依据(训练集 GT 实测)：表内「只有一格有内容」的横幅行(分组标题、年龄段、疾病类别…)，
GT 写成 `<td colspan="列数">文本</td>` 的有 102 例、写成裸 `<td>` 的 5 例 —— 主流写法
是补满 colspan。而 API 返回里同一张表的三个同类横幅行可能出现三种写法
(colspan=2 + 空 td / colspan=4 / 裸 td)，行宽对不齐就是错位。

只做一件事：把行宽 ≠ 表列数、且只有一格有内容的行，改写成单个 colspan=列数 的格。
不新增/删除行，不动有多格内容的行(那是 rowspan 续行，GT 里合法地窄)。
"""
import re
import collections

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.I | re.S)
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.I | re.S)
_CELL_RE = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1>", re.I | re.S)
_SPAN_RE = re.compile(r"\b(colspan|rowspan)\s*=\s*\"?(\d+)\"?", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_LEAD_NUM_RE = re.compile(r"^\s*(\d{1,3})\b")


def _spans(attrs):
    d = {"colspan": 1, "rowspan": 1}
    for k, v in _SPAN_RE.findall(attrs or ""):
        d[k.lower()] = max(1, int(v))
    return d


def _cells(row):
    return [(m.group(1), _spans(m.group(2)), m.group(3)) for m in _CELL_RE.finditer(row)]


def _width(cells):
    return sum(c[1]["colspan"] for c in cells)


def _n_cols(rows_cells):
    """表列数。

    取「无 rowspan、且不止一格」的行的行宽众数 —— rowspan 续行天然比整行窄
    (被上一行的格占位)，单格行是横幅，两者都不能用来定列数。
    都不满足时退回全表最大行宽。
    """
    cand = [_width(cs) for cs in rows_cells
            if len(cs) > 1 and all(c[1]["rowspan"] == 1 for c in cs)]
    if cand:
        return collections.Counter(cand).most_common(1)[0][0]
    widths = [_width(cs) for cs in rows_cells]
    return max(widths) if widths else 1


def _is_banner(cells):
    """只有一格有文本、其余格全空 → 满宽横幅行。"""
    filled = [c for c in cells if c[2].strip()]
    return len(filled) == 1 and filled[0][2].strip()


def fix_tables(md):
    """返回修好横幅行 colspan 的 Markdown。无表格则原样返回。"""
    def fix_one(m):
        table = m.group(0)
        rows = _ROW_RE.findall(table)
        if len(rows) < 2:
            return table
        rows_cells = [_cells(r) for r in rows]
        nc = _n_cols(rows_cells)
        if nc < 2:
            return table

        out = table
        for row, cells in zip(rows, rows_cells):
            if not cells or _width(cells) == nc:
                continue
            text = _is_banner(cells)
            if not text:
                continue                       # 多格有内容 = rowspan 续行,不动
            tag = cells[0][0]
            new = f'<tr><{tag} colspan="{nc}">{text}</{tag}></tr>'
            out = out.replace(row, new, 1)
        return out

    return _TABLE_RE.sub(fix_one, md or "")


# ---------------------------------------------------------------------------
# 子表切分：满宽横幅行 + 其后首列编号重启 → 这里是两张表的边界
# ---------------------------------------------------------------------------
def _lead_num(row):
    """行首列的前导整数(疾病表的序号列)。取不到返回 None。"""
    cells = _cells(row)
    if not cells:
        return None
    m = _LEAD_NUM_RE.match(_TAG_RE.sub("", cells[0][2]).strip())
    return int(m.group(1)) if m else None


def split_tables(md):
    """把被读成一张的多张子表切开。

    源文档里「轻症疾病 / 中症疾病 / 重大疾病」这类各自成框的清单表，API 常合并成
    一张、用满宽横幅行表示分界；而表【内部】的分组横幅(「第1组：恶性肿瘤类疾病」)
    长得一模一样，不能切。区分两者的是**首列编号**:
      - 分表边界 → 编号重启(…35 → 01)
      - 表内分组 → 编号连续(…第1组…1..40…第2组…41..)
    所以只在「满宽横幅行 + 其后首行编号 ≤ 此前最大编号」处切。

    依赖 fix_tables 先跑:横幅行要先归一成「单格 colspan=列数」才认得出。
    """
    def split_one(m):
        table = m.group(0)
        rows = _ROW_RE.findall(table)
        rows_cells = [_cells(r) for r in rows]
        nc = _n_cols(rows_cells)
        if nc < 2 or len(rows) < 4:
            return table

        groups, last = [[]], None
        for i, (row, cells) in enumerate(zip(rows, rows_cells)):
            banner = (len(cells) == 1 and _width(cells) == nc
                      and cells[0][2].strip())
            if banner and groups[0]:
                nxt = _lead_num(rows[i + 1]) if i + 1 < len(rows) else None
                if nxt is not None and last is not None and nxt <= last:
                    groups.append([])          # 编号重启 → 新表从这条横幅开始
                    last = None
            groups[-1].append(row)
            n = _lead_num(row)
            if n is not None:
                last = n

        if len(groups) < 2:
            return table
        return "\n".join("<table>" + "".join(g) + "</table>"
                         for g in groups if g)

    return _TABLE_RE.sub(split_one, md or "")
