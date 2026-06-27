# -*- coding: utf-8 -*-
"""TABLE 端到端 runner + TEDS 评测。

流程：预处理 → 网格切片 → 并发调用 → 2D 重组 → 与 GT 比 TEDS / 文本编辑距离。
所有 API 调用走缓存。
"""
import os
import re
import glob
import argparse
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import api_client as api
from config import TRAIN_TABLE_DIR, API_USER_IDS
from preprocess import prep
from slicer_table import slice_table
from split_table import subtables
from stitch_table import stitch_table, parse_tile, parse_tile_segments, rows_to_html
from evaluate import table_teds, text_edit_loss


def _up(t, factor):
    """密集小字 tile 上采样：放大 factor 倍让 API 读得清(修行幻觉/列漂移)。
    factor≤1 或 None → 原样返回。"""
    if t is not None and factor and factor > 1:
        return t.resize((round(t.width * factor), round(t.height * factor)), Image.LANCZOS)
    return t


def _is_truncated(o):
    """tile 输出被 ~12k 上限截断：有 <table 却无 </table>。"""
    return bool(o) and "<table" in o.lower() and "</table>" not in o.lower()


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


def _split_call_merge(img, timeout, depth=0, axis="h", cache_dir=None):
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
    if depth < 2 and _is_truncated(out1) and min(first.size) > 200:
        out1 = _split_call_merge(first, timeout, depth + 1, axis=axis, cache_dir=cache_dir)
    if depth < 2 and _is_truncated(out2) and min(second.size) > 200:
        out2 = _split_call_merge(second, timeout, depth + 1, axis=axis, cache_dir=cache_dir)

    rows1, rows2 = parse_tile(out1), parse_tile(out2)
    if axis == "v":
        merged = _merge_side_by_side(rows1, rows2)
    else:
        merged = rows1 + rows2
    return rows_to_html(merged) if merged else (out1 or out2)


def _tile_shape(html):
    rows = parse_tile(html)
    if not rows:
        return 0, 0, 0
    widths = [len(r) for r in rows]
    return len(rows), max(widths), sorted(widths)[len(widths) // 2]


def _bad_tile_reason(img, html, expected_rows=None):
    """识别后处理难以挽救的 tile。

    返回 None 表示可用；返回字符串表示建议重读。这里刻意只抓高置信坏块，
    避免把稀疏表/短表误判后引入更多 API 调用。
    """
    nrows, maxw, medw = _tile_shape(html)
    w, h = img.size
    # 典型展平幻觉：1~2 行、几百列；或者超过像素物理上限。
    if nrows <= 2 and maxw >= 80:
        return "flat"
    if maxw > max(80, w // 6):
        return "too_wide"
    if _is_truncated(html):
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
    from config import MAX_CONCURRENCY
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
    workers = min(MAX_CONCURRENCY, max(1, len(todo)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fixed = list(ex.map(
            lambda item: _repair_bad_tile(
                _up(tiles[item[0]][item[1]], upsample), outs[item[0]][item[1]], item[2],
                timeout, row_cells[item[0]] if item[0] < len(row_cells) else None,
                cache_dir=cache_dir),
            todo))
    for (r, c, _reason), o in zip(todo, fixed):
        outs[r][c] = o
        # 修复成功的**完整**结果回写缓存：截断 tile 此前未落盘（api 不缓存截断），
        # 这里把拆分重读合并的完整结果写回原 tile key，下次直接命中、不再拆分。
        exp = row_cells[r] if r < len(row_cells) else None
        up_img = _up(tiles[r][c], upsample)
        if o and _bad_tile_reason(up_img, o, exp) is None:
            api.write_cache(up_img, o, cache_dir=cache_dir)
    return outs


def _call_grid(tiles, timeout=240, upsample=1, cache_dir=None):
    """并发调用 2D tiles，保持 [r][c] 结构。None（空白块）不调 API。
    upsample>1 时每个 tile 先放大再调（修密集小字读崩），缓存进 cache_dir(分离)。

    **失败重试**：非空白 tile 若返回空串（=API 限流/失败被 call_safe 静默置空），
    会被当成空白块丢内容 → **分数随运行随机波动**(同一图两次跑分不同)。
    故对"非空白却空"的 tile 做多轮重试(降并发避开限流)，直到拿到内容或轮次用尽，
    保证结果可复现、不随机丢行。
    """
    from config import MAX_CONCURRENCY
    flat = [(r, c) for r in range(len(tiles))
            for c in range(len(tiles[r])) if tiles[r][c] is not None]

    def _call_set(items, workers):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(
                lambda x: api.call_safe(_up(tiles[x[1][0]][x[1][1]], upsample), timeout=timeout,
                                        user_id=API_USER_IDS[x[0] % len(API_USER_IDS)],
                                        cache_dir=cache_dir),
                list(enumerate(items))))

    grid = [[None] * len(tiles[r]) for r in range(len(tiles))]
    for (r, c), o in zip(flat, _call_set(flat, min(MAX_CONCURRENCY, max(1, len(flat))))):
        grid[r][c] = o

    # 重试失败置空的 tile（CACHE_ONLY 离线评测下跳过——空=未缓存,重调也是空）
    if not getattr(api, "CACHE_ONLY", False):
        for _round in range(4):
            empties = [(r, c) for (r, c) in flat if not (grid[r][c] or "").strip()]
            if not empties:
                break
            for (r, c), o in zip(empties, _call_set(empties, max(1, min(6, len(empties))))):
                if (o or "").strip():
                    grid[r][c] = o
    return grid


def _strip_html(s):
    """表外文字块按纯文本拼回：剥掉模型可能裹上的 HTML 标签，折叠空白。"""
    return " ".join(re.sub(r"<[^>]+>", " ", s or "").split())


def _recognize_text(im, bbox, timeout, pad=6):
    """裁一个小文字块 → 单独 API 识别 → 剥成纯文本。表外文字块与子表上方的标题段
    共用这一套（裁剪→识别→拼接），不让小文字被 stitch 包成空 <table>。"""
    x0, y0, x1, y1 = bbox
    crop = im.crop((max(0, x0 - pad), max(0, y0 - pad),
                    min(im.width, x1 + pad), min(im.height, y1 + pad)))
    return _strip_html(api.call_safe(crop, timeout=timeout))


def _call_text_blocks(im, blocks, timeout):
    """表外孤立文字块(页眉/页脚/水印/页码)单独识别，按位置返回拼到表格前/后的文本。
    放在主表之后调(query 最后)；CACHE_ONLY 离线评测下未缓存→空，不影响既有评测。"""
    if not blocks:
        return "", ""
    from config import MAX_CONCURRENCY
    pad = 6
    imgs = [im.crop((max(0, x0 - pad), max(0, y0 - pad),
                     min(im.width, x1 + pad), min(im.height, y1 + pad)))
            for _where, (x0, y0, x1, y1) in blocks]
    with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, max(1, len(imgs)))) as ex:
        outs = list(ex.map(lambda t: api.call_safe(t, timeout=timeout), imgs))
    before, after = [], []
    for (where, _box), o in zip(blocks, outs):
        txt = _strip_html(o)
        if txt:
            (before if where == "before" else after).append(txt)
    b = ("\n".join(before) + "\n") if before else ""
    a = ("\n" + "\n".join(after)) if after else ""
    return b, a


def run_one(im, timeout=240, peel=True, col_tile_max=None, max_rows=None):
    tiles, meta = slice_table(im, peel=peel, col_tile_max=col_tile_max, max_rows=max_rows)
    up = meta.get("upsample", 1)
    cdir = api.CACHE_UP_DIR if up > 1 else None      # 上采样 tile 缓存与原始分离
    outs = _call_grid(tiles, timeout, upsample=up, cache_dir=cdir)
    if not getattr(api, "CACHE_ONLY", False):
        outs = _refine_bad_tiles(tiles, outs, meta, timeout, upsample=up, cache_dir=cdir)
    # 全宽模式:走 stitch 的单表模式(关子表检测,保留列重建/稀疏补位)。子表已在几何层切好,
    # stitch 不该再拆——否则全宽单列tile被它的 caption 边界/表头行检测误拆成多表(8a4 11.3)。
    pred = stitch_table(outs, meta, single=bool(meta.get("fullwidth")))
    # 表外文字块：单独识别后按位置拼回表格前/后(query 放最后)
    before, after = _call_text_blocks(im, meta.get("text_blocks", []), timeout)
    pred = before + pred + after
    ncalls = sum(1 for row in tiles for t in row if t is not None) + len(meta.get("text_blocks", []))
    return pred, ncalls, meta


_MIN_TABLE_CELLS = 10   # 段识别出 td≥此数=真子表;否则=标题(转文本)。实测真子表td≥72、标题≤5


def _grid_cells(g):
    return sum(len(r) for r in g)


def _grid_cols(g):
    ws = [len(r) for r in g if r]
    return Counter(ws).most_common(1)[0][0] if ws else 0


def run_one_split(im, timeout=240):
    """多子表：subtables 把整图切成有序块 [(kind,bbox)]。'text' 块(表外文字)单独识别；
    'seg' 块走 run_one，**按识别出的 td 数判**:多格(≥10)=真子表(保留 table)、单格/几格
    =标题(转纯文本)。按阅读顺序拼接。单表(只 1 个 seg)退回 run_one 整图。

    table_teds 按出现顺序逐表配对,切成 N 个独立 <table> 逐个对齐 GT[i]——避免旧 stitch
    把 N 子表读成 1 大 table。按 td 数(识别结果)判表/标题,不靠几何段高,矮的真子表(ec745
    147px、td=306)不会被误当标题合并掉。

    **表头小条合并**:竖线断裂(③)有时把单表的「列号表头行」从表身剥下来切成两段——表头
    小条(td≈220)+表身大表(td≈6000),列数几乎相同。这会让 pred 表数>GT、索引错位 TEDS 崩
    (945e8fe9/88c6dbb4/1f4293f3 89→3.5)。判据:相邻两表,小表 cells < 大表/8 **且** 列数差
    ≤6 → 判表头小条,行拼接(上+下)归一表。两条 AND 缺一不可:只看列数会误并真子表(97c4c182
    列72≈67);只看 td 小会误并真小子表(19a15357 顶部 td80 但列10≠68)。真子表(列不同)一律
    不动。同一合并顺手修「一段被 run_one 多吐碎块」(c713916a td=39 碎块顶歪索引)。
    **未合并的表沿用 run_one 原始 HTML 输出,与改动前逐字节一致,不影响其它表。**

    **稀疏薄条退 fallback**:稀疏尾行(只剩行号、数据空)被白缝切下来后,墨量低(<1%),
    本身不是子表;若除它之外只剩 1 个密集真表(墨≥1%),整图其实是单表 → 退回整图 run_one
    (整图能把稀疏尾正确补成满列宽,如 80995347 读全 107/107 行)。**只数密集 seg 决定是否
    多子表**,稀疏薄条不计入——避免 94352240(1密集+2稀疏尾)被当 3 子表切进 split 路径、稀疏
    尾读成 1 列假表顶歪索引(98→68)。阈值 1% 边际极大(实测薄条≤0.72%、真表≥4.13%),真小
    子表(19a15357 顶部 9.71%)稳算密集、不误退。"""
    blocks = subtables(im)
    g0 = np.asarray(im.convert("L")) < 128
    def _seg_ink(bb):
        x0, y0, x1, y1 = bb
        return g0[y0:y1, x0:x1].mean() if y1 > y0 and x1 > x0 else 0.0
    dense = [bb for k, bb in blocks if k == "seg" and _seg_ink(bb) >= 0.01]
    if len(dense) <= 1:                              # 密集真表≤1 → 单表,退整图 fallback
        return run_one(im, timeout)
    items, ncalls = [], 0          # item: ["table", grid, html|None] | ["text", str, None]
    for kind, bb in blocks:
        if kind == "text":
            txt = _recognize_text(im, bb, timeout)
            if txt:
                items.append(["text", txt, None])
        else:                                  # seg: run_one 后按 td 数判表/标题
            # peel=False:子表裁块已是独立子表,不再剥表外文字(否则矮子表薄数据行被误剥)
            p, nc, _ = run_one(im.crop(bb), timeout, peel=False)
            ncalls += nc
            if p.lower().count("<td") >= _MIN_TABLE_CELLS:
                grids = [g for _, g in parse_tile_segments(p) if g]
                if len(grids) == 1:
                    items.append(["table", grids[0], p])      # 整段一表:留原始 HTML
                else:
                    for g in grids:                           # 一段多表(碎块):待合并,重渲染
                        items.append(["table", g, None])
            else:
                txt = _strip_html(p)
                if txt:
                    items.append(["text", txt, None])
    merged = []                    # 表头小条合并:只「上面的小条 并入 下面的表身」(单向)
    for it in items:
        if it[0] == "table" and merged and merged[-1][0] == "table":
            prev = merged[-1][1]
            # 方向:表头小条恒在表身【上方】→ 只有 prev(上)是小条、it(下)是表身才并。
            # 反向(下面的小表并入上面)不合理:最下面的真子表(532 seg2)/读崩的子表(90a 表1)
            # 都在后面,prev 是大表 → 不会被误吃。
            if _grid_cells(prev) < _grid_cells(it[1]) / 8 \
                    and abs(_grid_cols(prev) - _grid_cols(it[1])) <= 6:
                merged[-1] = ["table", prev + it[1], None]    # 小条(上)+表身(下)拼行
                continue
        merged.append(it)
    parts, ntab = [], 0
    for kind, val, html in merged:
        if kind == "table":
            parts.append(html if html is not None else rows_to_html(val))
            ntab += 1
        else:
            parts.append(val)
    return "\n".join(parts), ncalls, {"subs": ntab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--pick", choices=["median", "small", "spread"],
                    default="median")
    ap.add_argument("--split", action="store_true",
                    help="多子表先几何切分再各自 run_one")
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
        pred, ncalls, meta = run_one(im, args.timeout)
        teds = table_teds(pred, gt)
        te = text_edit_loss(pred, gt, include_tables=True)
        teds_list.append(teds if teds is not None else 0.0)
        nr, nc = len(meta["row_cuts"]) - 1, len(meta["col_cuts"]) - 1
        print(f"[{uuid[:8]}] gt_len={len(gt):>7} 网格={nr}x{nc}={ncalls}块 "
              f"grid={meta['grid']} | TEDS={teds:.4f} textScore={(1-te)*100:.1f} "
              f"pred_len={len(pred)}")

    if teds_list:
        print(f"\n===== 均值 TEDS = {sum(teds_list)/len(teds_list):.4f} "
              f"(×100 = {sum(teds_list)/len(teds_list)*100:.1f}) =====")


if __name__ == "__main__":
    main()
