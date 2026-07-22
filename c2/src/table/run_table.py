# -*- coding: utf-8 -*-
"""TABLE Stage II/III：逐块识别(ocr/ocr_text) + 合并拼接(merge) + 入口 parse_table。

主路:crop 裁块 → ocr_seg 骨架 OCR;列错位表回退 ocr_table 自由读(tile+重组),
坏 tile 由 _refine_bad_tiles 裁剪重读。main() 是抽样冒烟测试 CLI。所有 API 调用走缓存。
"""
import os
import re
import glob
import argparse
from collections import Counter

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import common.api_client as api
from common.config import TRAIN_TABLE_DIR, BIN_FAINT
from common.preprocess import prep
from table.slicer_table import slice_table
from table.crop import crop
from table.grid_ocr import ocr_seg
from table.tiles import upscale, pad_white, call_tiles, ASPECT_SAFE
from table.stitch_single import stitch_single_table, parse_tile, parse_tile_segments, rows_to_html
from metrics.evaluate import table_teds, text_edit_loss


def _merge_side_by_side(left_rows, right_rows):
    """左右半块按行号拼接；行数不等时短侧补空行。"""
    n = max(len(left_rows), len(right_rows))
    out = []
    for i in range(n):
        row = []
        if i < len(left_rows):
            row.extend(left_rows[i])
        if i < len(right_rows):
            row.extend(right_rows[i])
        out.append(row)
    return out


def _split_call_merge(img, timeout, *, axis, depth=0, cache_dir=None):
    """把坏 tile 裁剪成两半重读后合并。

    axis="h": 上/下拆，适合截断、少行(row_under)。
    axis="v": 左/右拆，适合超宽展平(1 行 x 几百列)。
    """
    w, h = img.size
    if axis == "v":
        first = img.crop((0, 0, w // 2, h))
        second = img.crop((w // 2, 0, w, h))
    else:
        first = img.crop((0, 0, w, h // 2))
        second = img.crop((0, h // 2, w, h))

    out1 = api.call_safe(first, timeout=timeout, cache_dir=cache_dir)
    out2 = api.call_safe(second, timeout=timeout, cache_dir=cache_dir)
    if depth < 2 and api.is_truncated(out1) and min(first.size) > 200:
        out1 = _split_call_merge(first, timeout, axis=axis, depth=depth + 1, cache_dir=cache_dir)
    if depth < 2 and api.is_truncated(out2) and min(second.size) > 200:
        out2 = _split_call_merge(second, timeout, axis=axis, depth=depth + 1, cache_dir=cache_dir)

    rows1, rows2 = parse_tile(out1), parse_tile(out2)
    if axis == "v":
        merged = _merge_side_by_side(rows1, rows2)
    else:
        merged = rows1 + rows2
    return rows_to_html(merged) if merged else (out1 or out2)


def _tile_shape(html):
    rows = parse_tile(html)
    if not rows:
        return 0, 0
    return len(rows), max(len(r) for r in rows)


def _bad_tile_reason(img, html, expected_rows=None):
    """识别后处理难以挽救的 tile。

    返回 None 表示可用；返回字符串表示建议重读。这里刻意只抓高置信坏块，
    避免把稀疏表/短表误判后引入更多 API 调用。
    """
    nrows, maxw = _tile_shape(html)
    w = img.size[0]
    # 典型展平幻觉：1~2 行、几百列；或者超过像素物理上限。
    if nrows <= 2 and maxw >= 80:
        return "flat"
    if maxw > max(80, w // 6):
        return "too_wide"
    if api.is_truncated(html):
        return "truncated"
    if nrows == 0:
        return None
    if expected_rows and expected_rows >= 8 and nrows < expected_rows * 0.45:
        # tile 很高却只读出极少行，通常是漏读或折叠；短/稀疏块不走这条。
        return "row_under"
    return None


def _repair_bad_tile(img, html, reason, timeout, expected_rows=None, cache_dir=None):
    """按坏块类型裁剪重读；若仍坏则尝试另一方向，最后保留原输出。"""
    if reason in ("flat", "too_wide"):
        fixed = _split_call_merge(img, timeout, axis="v", cache_dir=cache_dir)
        if _bad_tile_reason(img, fixed, expected_rows) is None:
            return fixed
        fixed2 = _split_call_merge(img, timeout, axis="h", cache_dir=cache_dir)
        return fixed2 if _bad_tile_reason(img, fixed2, expected_rows) is None else html
    if reason in ("truncated", "row_under"):
        fixed = _split_call_merge(img, timeout, axis="h", cache_dir=cache_dir)
        if _bad_tile_reason(img, fixed, expected_rows) is None:
            return fixed
        fixed2 = _split_call_merge(img, timeout, axis="v", cache_dir=cache_dir)
        return fixed2 if _bad_tile_reason(img, fixed2, expected_rows) is None else html
    return html


def _refine_bad_tiles(tiles, outs, meta, timeout=240, upsample=1, cache_dir=None):
    """对所有高置信坏 tile 局部裁剪重读。upsample>1 时坏块按上采样后的图重读、
    回写进 cache_dir(与原始缓存分离)。"""
    row_cells = meta.get("row_cells", []) if meta else []
    todo = []
    for r in range(len(outs)):
        for c in range(len(outs[r])):
            if tiles[r][c] is None:
                continue
            expected = row_cells[r] if r < len(row_cells) else None
            reason = _bad_tile_reason(tiles[r][c], outs[r][c], expected)
            if reason:
                todo.append((r, c, reason))
    if not todo:
        return outs
    priority = {"flat": 0, "too_wide": 1, "truncated": 2, "row_under": 3}
    todo = sorted(todo, key=lambda x: priority.get(x[2], 9))
    fixed = api.map_io(                              # 统一发送层并发(修复函数内部只用
        lambda item: _repair_bad_tile(               # 同步 call_safe,单层池无死锁)
            upscale(tiles[item[0]][item[1]], upsample), outs[item[0]][item[1]], item[2],
            timeout, row_cells[item[0]] if item[0] < len(row_cells) else None,
            cache_dir=cache_dir),
        todo)
    for (r, c, _), o in zip(todo, fixed):
        outs[r][c] = o
        # 修复成功的**完整**结果回写缓存：截断 tile 此前未落盘（api 不缓存截断），
        # 这里把拆分重读合并的完整结果写回原 tile key，下次直接命中、不再拆分。
        exp = row_cells[r] if r < len(row_cells) else None
        up_img = upscale(tiles[r][c], upsample)
        if o and _bad_tile_reason(up_img, o, exp) is None:
            api.write_cache(up_img, o, cache_dir=cache_dir)
    return outs


def _call_grid(tiles, timeout=240, upsample=1, cache_dir=None):
    """并发调用 2D tiles，保持 [r][c] 结构。None（空白块）不调 API。
    调用层复用 tiles.call_tiles；retry_rounds=4:非空白 tile 返回空串(限流被
    call_safe 静默置空)会被当空白块丢内容→分数随运行随机波动,重试保证可复现。"""
    flat = [(r, c) for r in range(len(tiles))
            for c in range(len(tiles[r])) if tiles[r][c] is not None]
    outs = call_tiles([tiles[r][c] for r, c in flat], timeout=timeout,
                      upsample=upsample, cache_dir=cache_dir, retry_rounds=4)
    grid: list = [[None] * len(tiles[r]) for r in range(len(tiles))]
    for (r, c), o in zip(flat, outs):
        grid[r][c] = o
    return grid


def _strip_html(s):
    """表外文字块按纯文本拼回：剥 HTML 标签，**保留逻辑行**(表行 </tr> / <br> / 空行 →
    段落断),仅段内折叠空白。GT 把多行标题(主标题/副标题/单位)拆成独立逻辑块按阅读流计分,
    若把多行折成一行→pred 1 块 vs GT N 块、RO 腰斩(e3e19b92/051fa323 等 TEDS≈100 却 RO50)。"""
    s = re.sub(r"<\s*br\s*/?>", "\n\n", s or "", flags=re.I)
    s = re.sub(r"</\s*tr\s*>", "\n\n", s, flags=re.I)        # 表行边界 → 段落断
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"(?m)^\s*[#＃*]+\s*", "", s)                 # 去 markdown 标记(API 偶给标题加 #)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", s)]
    return "\n\n".join(p for p in paras if p)


def _merge_text_blocks(md):
    """把表格之间/之前/之后的【连续文字段落】合并成一个逻辑块,只在表格边界保留空行。

    官方 read_order 是块级 edit-distance。GT 按物理区域把相近文字聚成少数块(中位 2),
    我们按 API 的 \\n\\n 拆得过碎(中位 3、长尾到 20+)→ 过拆 = 大量"插入"被扣分。
    实测合并后 honest RO 56→75、Overall 81→87(零 TEDS/text 影响,只改块边界)。
    粗暴版:连续文字无脑合一块;少数 GT 本就多块的(主/副/单位)会被过合并,但聚合净 +18.7。"""
    parts = re.split(r"(<table[^>]*>.*?</table>)", md or "", flags=re.S | re.I)
    out = []
    for part in parts:
        if re.match(r"\s*<table", part, flags=re.I):
            out.append(part.strip())
        else:
            txt = " ".join(part.split())                    # 连续文字 → 折叠成一段
            if txt:
                out.append(txt)
    return "\n\n".join(out)


def _ocr_strip(img, timeout):
    """细长条(>180:1)分段 OCR:API 拒收 >200:1(400)。子表间被 peel 剥出的表头小条
    (a1aaef73 4300×17≈250:1)是**真内容**,读不到 → tfrag 判不成 → 子表首行丢失
    (tr56/60)。不垫白:在中部±20%找最白列切开,递归读两半,文本拼接——切在真实白缝,
    内容无损。"""
    import numpy as np
    w, h = img.size
    if w <= ASPECT_SAFE * max(1, h):
        return api.call_safe(img, timeout=timeout)
    g = np.asarray(img.convert("L"))
    lo, hi = int(w * 0.3), int(w * 0.7)
    colf = (g < BIN_FAINT).mean(axis=0)[lo:hi]
    cut = lo + int(colf.argmin())
    left = _ocr_strip(img.crop((0, 0, cut, h)), timeout)
    right = _ocr_strip(img.crop((cut, 0, w, h)), timeout)
    return " ".join(x for x in (left, right) if x)


def ocr_text(im, bbox, timeout):
    """Stage II — 裁一个文字块 → 单独 API 识别 → 剥成纯文本。表外 furniture 与子表上方
    的标题段共用这一套(裁剪→识别→拼接),不让小文字被 stitch 包成空 <table>。"""
    x0, y0, x1, y1 = bbox
    crop = pad_white(im.crop((x0, y0, x1, y1)))      # 白pad与tile同原则(34e53b1c
    crop = upscale(crop, 2)                                # 表1末行半截字被读成'000'垃圾)
    #                                                    2x 上采样:标题/页脚小字提精度
    return _strip_html(_ocr_strip(crop, timeout))


def ocr_table(im, timeout=240):
    """列错位表的自由读回退:切片→并发OCR→补差tile→单表 2D 重组。
    进来的一定是几何层已裁好的**单张表**(gt_tables=1 实证),stitch 不做子表检测。"""
    tiles, meta = slice_table(im)
    up = meta.get("upsample", 1)
    cdir = api.CACHE_UP_DIR if up > 1 else None      # 上采样 tile 缓存与原始分离
    outs = _call_grid(tiles, timeout, upsample=up, cache_dir=cdir)
    if not api.CACHE_ONLY:
        outs = _refine_bad_tiles(tiles, outs, meta, timeout, upsample=up, cache_dir=cdir)
    pred = stitch_single_table(outs, meta)
    ncalls = sum(1 for row in tiles for t in row if t is not None)
    return pred, ncalls, meta


_MIN_TABLE_CELLS = 10   # 段识别出 td≥此数=真子表;否则=标题(转文本)。实测真子表td≥72、标题≤5


def _grid_cells(g):
    return sum(len(r) for r in g)


def _grid_cols(g):
    ws = [len(r) for r in g if r]
    return Counter(ws).most_common(1)[0][0] if ws else 0


def ocr(im, blocks, timeout=240):
    """Stage II — 逐块 OCR + 就地定表/标题。返回 (items, ncalls)。
      item = ["table", grid, html|None]  |  ["text", str, None]

    'text' 块 → ocr_text 纯文本。'title' 块(peel 的 mark) → OCR 判真身:读出表格行
    (td≥阈值) = peel 误剥的数据/表头行 → 拼回紧随其后的 seg 表格(32799493 分节线上方
    两行数据被剥,靠此救回);否则按纯文本。'seg' 块 → **骨架 OCR(grid_ocr.ocr_seg)**:
    行列以几何估计为准切 tile、结果强制对齐骨架、稀疏区按骨架补空 cell;列错位表
    (骨架不可信)回退 ocr_table 自由读。"""
    items, ncalls = [], 0
    for kind, bb in blocks:
        if kind == "text":
            txt = ocr_text(im, bb, timeout)
            if txt:
                items.append(["text", txt, None])
            continue
        if kind == "title":               # mark:OCR 定真身——表格行 → 'tfrag'(Stage III 拼回)
            x0, y0, x1, y1 = bb
            raw = _ocr_strip(upscale(pad_white(im.crop((x0, y0, x1, y1))), 2), timeout)
            ncalls += 1
            rows = parse_tile(raw) if raw and raw.lower().count("<td") >= 6 else None
            if rows:
                items.append(["tfrag", rows, None])   # peel 误剥的表格行(mark),merge 定拼回
            else:
                txt = _strip_html(raw)
                if txt:
                    items.append(["text", txt, None])
            continue
        grid, nc, gmeta = ocr_seg(im.crop(bb), timeout)   # 骨架 OCR(一律以估计为准)
        for a in gmeta.get("audit", []):                  # 实读与估计不一致 → 审计日志
            print(f"  [audit] seg@y{bb[1]} {a}", flush=True)
        for a in gmeta.get("adopt", []):                  # 估计被API裁决改写 → 裁决日志
            print(f"  [adopt] seg@y{bb[1]} {a}", flush=True)
        ncalls += nc
        if grid is not None:
            grid = _fix_grid(grid)
            for sp in reversed(gmeta.get("splits") or []):
                # API证词拆表(caption+轴行双条件):轴行前的连续"文本行"(≤3非空且含中文)
                # 是节头,提出为表间文本;其余劈成两表
                if not (0 < sp < len(grid)):
                    continue
                head_txt = []
                for lo, txts in sorted((gmeta.get("split_txt") or {}).items()):
                    if lo <= sp:                   # 该带溢出改道的节头文本,归此拆点
                        head_txt += txts
                cut = sp
                while cut > 1:
                    ne = [c for c in grid[cut - 1] if c.strip()]
                    if 0 < len(ne) <= 3 and any('一' <= ch <= '鿿' for c in ne for ch in c):
                        head_txt.insert(0, " ".join(ne)); cut -= 1
                    else:
                        break
                lower = grid[sp:]
                upper = grid[:cut]
                if upper and lower:
                    items.append(["table", upper, None])
                    if head_txt:
                        items.append(["text", "\n\n".join(head_txt), None])
                    grid = lower
            cells = sum(1 for row in grid for v in row if v.strip())
            if cells >= _MIN_TABLE_CELLS:
                items.append(["table", grid, None])
            else:                                         # 读出内容太少=标题/文本块
                txt = ocr_text(im, bb, timeout)
                if txt:
                    items.append(["text", txt, None])
            continue
        # 列错位(rows_misaligned):骨架不可信 → 自由读回退(旧路)
        p, nc2, _ = ocr_table(im.crop(bb), timeout)
        ncalls += nc2
        if p.lower().count("<td") >= _MIN_TABLE_CELLS:
            grids = [g for _, g in parse_tile_segments(p) if g]
            if len(grids) == 1:
                items.append(["table", grids[0], p])      # 整段一表:留原始 HTML
            else:
                for g in grids:                           # 一段多表(碎块):待合并,重渲染
                    items.append(["table", g, None])
        else:                                             # td 少=标题:整块重读成纯文本
            txt = ocr_text(im, bb, timeout) or _strip_html(p)
            if txt:
                items.append(["text", txt, None])
    return items, ncalls


_MDOT = re.compile(r"^\d{1,3}(?:\.\d{3})+\.\d{2}$")


def _fix_mdot(s):
    """多点数字规范化: '2.357.38'→'2,357.38'。欧式点千分位在本域不存在(用户判定),
    pred侧是OCR把千分位逗号误读成点(f501eee1/f5009a2d),GT侧同款样式者视为GT标注
    误差,统一赌逗号。严格整形匹配:粘连残渣('107.8.11',第二段非3位)不动。"""
    if s and _MDOT.match(s):
        head, dec = s.rsplit(".", 1)
        return head.replace(".", ",") + "." + dec
    return s


def _fix_grid(grid):
    return [[_fix_mdot(c) for c in row] for row in grid]


def _grid_html(grid):
    """网格 → HTML,两步增强(治多表头族,GT 19/226 张表首行用合并单元格):
    ① 全空列剔除:整列全空 = 骨架赝品(列带边界空列),留着会撑坏 colspan 宽度
    ② 表头 span 合成(row1 证据驱动):row1 有字的连续列段 = colspan 表头覆盖区
       (子表头 1..30);段内 row0 文字 → colspan=段宽;段外 row0 文字 → rowspan=2。
       门槛:row0 既有文字又有空格、行数≥2、模式典型(段内仅一个 row0 文字,
       rowspan 候选的 row1 同列为空),否则放弃合成、平面输出——普通表不受影响。"""
    if not grid:
        return rows_to_html(grid)
    C = max(len(r) for r in grid)
    grid = [list(r) + [""] * (C - len(r)) for r in grid]
    # 不剔全空列:有框表的空列由框线定义、真实存在(90c8cdb9 尾部10列全空,GT保留为
    # 空td,剔除致td-690);无框赝品列(1674392a"）"列)代价仅一个空格,两害相权留着
    row0 = grid[0]
    n = len(row0)
    if len(grid) < 2 or not any(not c.strip() for c in row0) \
            or not any(c.strip() for c in row0):
        return rows_to_html(grid)
    row1 = grid[1]
    runs = []
    j = 0
    while j < n:
        if row1[j].strip():
            k = j + 1
            while k < n and row1[k].strip():
                k += 1
            runs.append((j, k))
            j = k
        else:
            j += 1
    # 典型多表头模式:恰一个宽段(≥4列)覆盖子表头,其余 row0 文字都在段外
    wide = [r for r in runs if r[1] - r[0] >= 4]
    if len(wide) != 1:
        return rows_to_html(grid)
    a, b = wide[0]
    texts_in = [t for t in range(a, b) if row0[t].strip()]
    if len(texts_in) > 1:
        return rows_to_html(grid)
    outside = [j for j in range(n) if not (a <= j < b)]
    if any(row1[j].strip() and not row0[j].strip() for j in outside):
        return rows_to_html(grid)               # 段外 row1 有字但 row0 没有 → 非典型
    parts = ['<table border="1" cellpadding="8" cellspacing="0">', "      <tr>"]
    for j in range(n):
        if j == a:
            cap = row0[texts_in[0]] if texts_in else ""
            parts.append(f'<td colspan="{b - a}">{cap}</td>')
        elif a < j < b:
            continue
        elif row0[j].strip():
            span = ' rowspan="2"' if not row1[j].strip() else ""
            parts.append(f"<td{span}>{row0[j]}</td>")
        else:
            parts.append("<td></td>")
    parts.append("</tr>")
    parts.append("<tr>" + "".join(f"<td>{row1[j]}</td>" for j in range(n)
                                  if a <= j < b or row1[j].strip()) + "</tr>")
    for r in grid[2:]:
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def _is_header_strip(prev, body):
    """上表是下表的表头小条:cells<body/8 且 列差≤6(两条AND缺一不可,判据史见merge)。"""
    return (_grid_cells(prev) < _grid_cells(body) / 8
            and abs(_grid_cols(prev) - _grid_cols(body)) <= 6)


def merge(items):
    """Stage III — 表头小条合并 + 按阅读顺序拼接成 md。返回 (md, nsubtables)。

    表头小条合并(单向,只上面小条并入下面表身):相邻两表,若上表是小条(cells<下表/8
    **且** 列数差≤6)→ 行拼接归一表。两条 AND 缺一不可:只看列数会误并真子表(97c4c182
    列72≈67);只看 cells 小会误并真小子表(19a15357 顶 td80 但列10≠68)。修竖线断裂把
    表头行从表身剥下来的碎裂(945e8fe9/88c6dbb4/1f4293f3 89→3.5),顺手并一段多吐的碎块。"""
    # tfrag(title mark 判为表格行) → 拼回紧随其后的 table 表顶;无后表则自成一表
    fused = []
    i = 0
    while i < len(items):
        it = items[i]
        if it[0] == "tfrag":
            if i + 1 < len(items) and items[i + 1][0] == "table":
                g = items[i + 1][1]
                C = len(g[0]) if g else 0
                rows = [r[:C] + [""] * max(0, C - len(r)) for r in it[1]]
                items[i + 1] = ["table", rows + g, None]
            else:
                # 拼不进任何相邻表 → **降级为文本**:从子表拆出的块只能是表外标题,
                # 绝不允许以独立表格落地(34e53b1c 残影+标题条自立成第3张表,表数3v2
                # 配对全崩 0.49)。"拆出块必为标题"是不变量
                txt = " ".join(c for r in it[1] for c in r if c.strip())
                if txt.strip():
                    fused.append(["text", txt, None])
            i += 1
            continue
        fused.append(it)
        i += 1
    items = fused
    merged = []
    for it in items:
        # [表A, 短文本, 表B] 且 A 是 B 的表头小条 → 文本当成一行(colspan行)三合一:
        # peel 从表身剥出的组表头行 OCR 成文本后会挡在 A/B 之间,使"相邻两表"合并失效
        # (88c6dbb4 subs=2,TEDS 0.027);GT 恰好把该行算表内 colspan 行(4+1+107=112=GT)
        if (it[0] == "table" and len(merged) >= 2
                and merged[-1][0] == "text" and "\n" not in merged[-1][1]
                and len(merged[-1][1]) <= 80
                and merged[-2][0] == "table"):
            prev = merged[-2][1]
            if _is_header_strip(prev, it[1]):
                C = _grid_cols(it[1])
                trow = [merged[-1][1]] + [""] * max(0, C - 1)
                fusedg = prev + [trow] + it[1]
                merged.pop(); merged.pop()
                merged.append(["table", fusedg, None])
                continue
        if it[0] == "table" and merged and merged[-1][0] == "table":
            prev = merged[-1][1]
            # 方向:表头小条恒在表身【上方】→ 只有 prev(上)是小条、it(下)是表身才并。
            # 反向(下面的小表并入上面)不合理:最下面的真子表(532 seg2)/读崩的子表(90a 表1)
            # 都在后面,prev 是大表 → 不会被误吃。
            if _is_header_strip(prev, it[1]):
                merged[-1] = ["table", prev + it[1], None]    # 小条(上)+表身(下)拼行
                continue
        merged.append(it)
    parts, ntab = [], 0
    for kind, val, html in merged:
        if kind == "table":
            parts.append(html if html is not None else _grid_html(val))
            ntab += 1
        else:
            parts.append(val)
    return _merge_text_blocks("\n\n".join(parts)), ntab


def parse_table(im, timeout=240):
    """表格图 → md,三段式:crop(纯几何裁块) → ocr(逐块识别) → merge(合并+拼接)。"""
    blocks = crop(im)                        # Stage I
    items, ncalls = ocr(im, blocks, timeout)  # Stage II
    md, ntab = merge(items)                    # Stage III
    return md, ncalls, {"subs": ntab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--pick", choices=["median", "small", "spread"],
                    default="median")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(TRAIN_TABLE_DIR, "mds", "*.md")),
                   key=os.path.getsize)
    if args.pick == "median":
        mid = len(files) // 2
        sel = files[mid: mid + args.n]
    elif args.pick == "small":
        sel = files[:args.n]
    else:  # spread
        step = max(1, len(files) // args.n)
        sel = files[::step][:args.n]

    teds_list = []
    for md in sel:
        uuid = os.path.basename(md)[:-3]
        img = os.path.join(TRAIN_TABLE_DIR, "images", uuid + ".jpg")
        if not os.path.exists(img):
            continue
        gt = open(md, encoding="utf-8").read()
        im = prep(Image.open(img))
        pred, ncalls, meta = parse_table(im, args.timeout)
        teds = table_teds(pred, gt)
        te = text_edit_loss(pred, gt, include_tables=True)
        teds_list.append(teds if teds is not None else 0.0)
        print(f"[{uuid[:8]}] gt_len={len(gt):>7} subs={meta.get('subs')} calls={ncalls} "
              f"| TEDS={teds:.4f} textScore={(1-te)*100:.1f} pred_len={len(pred)}")

    if teds_list:
        print(f"\n===== 均值 TEDS = {sum(teds_list)/len(teds_list):.4f} "
              f"(×100 = {sum(teds_list)/len(teds_list)*100:.1f}) =====")


if __name__ == "__main__":
    main()
