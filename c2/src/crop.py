# -*- coding: utf-8 -*-
"""Stage I — 裁剪(纯几何,不调 OCR)。整图 → 有序裁块 [(kind, bbox)]。

kind:
  'text'  : 表外 furniture(页眉/页脚/水印/页码),Stage III 用 ocr_text 读纯文本。
  'seg'   : 候选表格区域,Stage II 用 ocr_table 读;是表还是标题由 Stage III 按 td 数定。
  'title' : 从 seg 顶部剥下的标题/副标题/colspan组表头(mark),Stage III 按 keep_title
            决定丢为文本 or 拼回 colspan 表头行。

三步:① split_table_texts 剥表外文字(geom) ② subtables 按子表缝切段 ③ _peel_title 剥表顶标题。
产物是可复核裁块,dump_plan 落盘 overlay + crop + manifest 供人工审阅。所有 OCR 相关判定
(表/标题、表头小条合并)都在 Stage III,不在这里。子表缝/框线/并排等几何原语见 geom.py。
"""
import os
import json
import numpy as np
from PIL import ImageDraw

from geom import (split_table_texts, _runlen_lines, _content_segs,
                  _boundaries, column_cuts)
from config import BIN_INK, BIN_FAINT, BIN_LINE
from imcache import cached


# ---------------------------------------------------------------------------
# 子表几何切分(原 split_table.py):把含多子表的整图切成单子表段
# ---------------------------------------------------------------------------
def _panel_seam_xs(g):
    """左右并排子表的竖直中缝 x 坐标:靠"横向长墨线(行)被竖直白带打断"判定。
    与 geom._panel_seams 同源,但返回缝的 x 位置(用于切分)而非缝数。零误拆。"""
    W = g.shape[1]
    step = max(1, W // 1500)
    g2 = g[::step, ::step]
    Hs, Ws = g2.shape
    dark = (g2 < BIN_LINE)
    thr = 0.25 * Ws
    minrun = 0.1 * Ws
    covx = np.zeros(Ws, bool)
    found = False
    for y in range(Hs):
        row = dark[y]
        if row.sum() < thr:
            continue
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        runs = e - s
        if runs.size == 0 or runs.max() < thr:
            continue
        found = True
        for a, b, L in zip(s, e, runs):
            if L >= minrun:
                covx[a:b] = True
    if not found:
        return []
    xs = np.where(covx)[0]
    xlo, xhi = xs[0], xs[-1]
    seams = []
    i = xlo
    while i <= xhi:
        if not covx[i]:
            j = i
            while j <= xhi and not covx[j]:
                j += 1
            if j - i > 0.03 * Ws:
                seams.append((i + j) // 2 * step)
            i = j
        else:
            i += 1
    return seams


_SEAM_K = 2.0
_SEAM_VMAX = 2


def _vline_break_ys(dark180):
    """竖线断裂找横缝:框线竖线(纵向长墨柱)在子表之间**全部断开**的 y。对称于 panel 的
    横线断裂找左右(这里纵向墨柱被横向白带打断找上下)。仅有框表(竖线≥3)。救"白带被稀疏
    行打断、缝高不足"的有框密集表(88684b6b/471413c1)。

    两条收紧(避免顶部抬头标题被切、切到文字):
    ① 只取**中间**的断裂带(3%<s<97%H)——排除顶部抬头标题/表尾,它们不是子表间。
    ② 切点取断裂带**顶部 s**(上子表框结束处)——标题归下子表、不切到标题文字。
    注:"只剩第1列"(88c6dbb4 GT1)和"真子表标题区"(1c9ac6d2 GT3)的断裂带 cov 都=2、几何
    无法区分,故仍按<20%判断裂——宁可 88c6dbb4 多切一刀(td 判后 table 仍=1、无害),也不能
    漏切 1c9ac6d2(漏切=table 数错、TEDS 崩)。"""
    H = dark180.shape[0]
    vlines = _runlen_lines(dark180, min_run=120)
    if len(vlines) < 3:
        return []
    cov = np.zeros(H)
    for x in vlines:
        cov += dark180[:, x]
    brk = cov < len(vlines) * 0.2                   # 贯穿竖线<20%=断裂
    ys = []
    y = 0
    while y < H:
        if brk[y]:
            s = y
            while y < H and brk[y]:
                y += 1
            if y - s >= 3 and H * 0.03 < s < H * 0.97:   # ② 中间的带; ③ 切点取顶部 s
                ys.append(s)
        else:
            y += 1
    return ys


def _row_bounds(dark, dark180=None, k=_SEAM_K, vmax=_SEAM_VMAX):
    """单栏内的上下子表切点:**子表缝 = 高白带 且 无框线竖线贯穿**。

    判据一(缝高):连续白带(整行墨<0.3%×W)高度 ≥ k×行缝基准(本栏白带高度中位)。子表
    边界纵向远高于普通行间缝——实测误拆单表的"缝"=行缝(缝高/行缝≈1.0)、真子表缝是
    行缝 2.5~10 倍,k=2 落在空带 [1.1,2.5] 正中。**不用 width_gate Otsu**:单表行缝
    等高时 Otsu 退化成全留、把每条行缝当切点。

    判据二(无内容贯穿):真子表缝 = 整带**所有列全空**。分两类贯穿各挡:
    · 内容列贯穿(中墨 0.15~0.8):阶梯/三角表下部数据逐行变短,只剩行号列/稀疏数据列
      连续穿过——整行墨和低会被 white 误判纯白,但该列一直有墨,是同表延续非子表缝
      (aef3bf0c 行号列 74~106 贯穿,列墨 0.62)。
    · 框线列贯穿(高墨 >0.8)数 ≤ vmax:稀疏表底部空行边框/列线仍穿过(cabe16d5 底部 9
      条竖线=仍在表内),真子表缝处框线断开(db515/1de69 = 0 条)。
    这条按"完全空白才是缝"替掉原魔数"段数<4"。"""
    H, W = dark.shape
    white = dark.sum(1) < max(2, 0.003 * W)          # 连续白线（纯白行）
    bands = []                                        # 连续白带 (起,止,高)
    y = 0
    while y < H:
        if white[y]:
            s = y
            while y < H and white[y]:
                y += 1
            bands.append((s, y, y - s))
        else:
            y += 1
    if not bands:
        return [0, H]
    row_gap = float(np.median([h for _, _, h in bands]))   # 典型行缝高
    cuts = []
    for s, e, h in bands:
        if h < k * row_gap or not (20 < (s + e) // 2 < H - 20):
            continue
        col = dark[s:e].mean(0)
        if bool(((col >= 0.15) & (col < 0.8)).any()):      # 内容列贯穿(行号列/稀疏数据列)
            continue                                       # = 同表延续(阶梯/三角表),非子表缝
        if int((col > 0.8).sum()) > vmax:                  # 框线竖线贯穿 → 表内，非缝
            continue
        cuts.append((s + e) // 2)
    if dark180 is not None:                                # 判据三:竖线断裂(有框表)
        cuts += _vline_break_ys(dark180)
    cuts = sorted(set(cuts))
    merged = []                                            # 相邻(<120px)缝合并=同一缝
    for c in cuts:
        if not merged or c - merged[-1] >= 120:
            merged.append(c)
    return [0] + merged + [H]


@cached("subtables", __file__)
def subtables(im, pad=4):
    """把整图切成**有序块**列表 `[(kind, bbox), ...]`(阅读顺序,先上后下)。

    kind='text' : split_table_texts 剥出的表外页眉/页脚/水印/页码。
    kind='seg'  : 主表区切出的子表段——**不在几何层预判它是标题还是表格**(交 Stage III 按 td 判)。
    单表(只 1 个 seg)由 crop 合回一块。"""
    tb, texts = split_table_texts(im)
    if tb is None:
        return [('seg', (0, 0, im.width, im.height))]
    x0, y0, x1, y1 = tb
    items = [(by0, 'text', (bx0, by0, bx1, by1))      # ① 表外小文字块
             for (bx0, by0, bx1, by1) in texts]
    g = np.asarray(im.crop(tb).convert("L"))
    dark = (g < BIN_INK)
    dark180 = (g < BIN_FAINT)                          # 松二值化:给竖线断裂检测(救浅灰框线)
    H, W = g.shape
    colb = [0] + _panel_seam_xs(g) + [W]
    for c in range(len(colb) - 1):                    # ② 主表内切子表段(不预判类型)
        cx0, cx1 = colb[c], colb[c + 1]
        rb = _row_bounds(dark[:, cx0:cx1], dark180[:, cx0:cx1])
        for ra, rbb in zip(rb[:-1], rb[1:]):
            i128 = dark[ra:rbb, cx0:cx1].mean()
            i180 = dark180[ra:rbb, cx0:cx1].mean()
            if i180 < 0.002:                         # 松二值化都近空 → 真空白带,丢
                continue
            bb = (x0 + cx0, y0 + max(0, ra - pad), x0 + cx1, y0 + min(H, rbb + pad))
            # kind='text'(表外标题/说明,交 Stage III 当文本) 的两类:① 矮段(<20px,如首标题
            # 12px);② **淡段**(g128 近空但 g180 有墨 = 灰度 130~180 的淡标题/说明,如
            # f64061da 首标题灰度177 g128墨0.0008被误当空丢)。其余浓段=子表 'seg'。
            kind = 'text' if (i128 < 0.002 or rbb - ra < 20) else 'seg'
            items.append((y0 + ra, kind, bb))
    items.sort(key=lambda t: t[0])                     # 阅读顺序
    return [(k, bb) for _, k, bb in items] or [('seg', tb)]


# ---------------------------------------------------------------------------
# 裁块规划:单表折叠 + 表顶标题剥离
# ---------------------------------------------------------------------------
_DENSE_INK = 0.01   # seg 墨量≥此=密集真表;<此=稀疏薄条,不计入子表计数


def _ink(g, bb):
    x0, y0, x1, y1 = bb
    return g[y0:y1, x0:x1].mean() if y1 > y0 and x1 > x0 else 0.0


def _union(bbs):
    xs0, ys0, xs1, ys1 = zip(*bbs)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


_COV_MIN = 0.50     # 真数据行:命中 ≥此比例 的列格
_CROSS_MAX = 0.30   # 真数据行:落在列切线上的墨 ≤此比例(标题连续跨列会超)


def _peel_title(im, bb):
    """剥 seg 顶部标题/副标题,返回 (title_bboxes, trimmed_seg_bbox)。剥出块由调用方标
    'title'(mark),Stage III 定去留。分两路(与列检测对称):

    · **有横框线**(全宽横墨线≥2条):表格上边界=最上一条横框线,标题必在其上方(框外)。
      cut=上边框——明确几何,不依赖列估计;数据在框内(cut下方)天然**不会被切**。
    · **无横框线**:用 column_cuts 的列切线判「真数据行」(cov 填多列 且 cross 不跨列缝),
      从顶剥到第一真数据行,cut 落其上——遇数据行即停,也**不切数据**。

    去掉了旧的 _PEEL_CAP(高度占比上限):它对"标题占大半的矮碎片"误挡;不切数据已由
    "框线上边界 / 遇数据行即停"两路各自保证。"""
    x0, y0, x1, y1 = bb
    gray = np.asarray(im.crop(bb).convert("L"))
    g, g180 = gray < BIN_INK, gray < BIN_FAINT
    W = g.shape[1]
    # 路① 有横框线:最上横框线=表格上边,其上方有内容=标题。
    # 用**松二值化 g180**(g<180)检横线——框线常是淡灰线(灰度128~180),严二值化 g<128 会
    # 漏掉最外框上边(如 4df76a25 淡线 y251 严0.02/松0.82),致误把框内表头当框上方标题剥。
    hl, prev = [], -99
    for y in np.where(g180.mean(axis=1) > 0.5)[0]:
        if y - prev > 3:
            hl.append(int(y))
        prev = int(y)
    hl = [y for y in hl if y == 0 or y >= 8]      # 滤 pad 残余(0<y<8, seg重叠带入上一seg底框线)
    if len(hl) >= 6:                              # **大量横框线=密集网格表** → 不 peel:
        return [], bb                            #   框线已定表结构(表头/数据/框内标题都在框里),
        #   交 ocr_table 读, 避免误剥表头/框内标题(有框不重要;"有框"指大量框线而非几根)
    # 路② 少量框线(<6,几根)/无框:列对齐判「真数据行」,从顶剥标题到第一真数据行
    bands = [(s, e + 1) for s, e in _content_segs(g.mean(axis=1), gap=6, thr=0.004)
             if e - s >= 4]
    if len(bands) < 3:
        return [], bb
    col_bnd = _boundaries(column_cuts(g, g180)[0], W)
    cells = list(zip(col_bnd[:-1], col_bnd[1:]))   # 列格
    cuts = col_bnd[1:-1]                           # 内部列切线(白缝中心)
    if len(cells) < 4 or not cuts:
        return [], bb

    def is_data_row(s, e):
        rowink = g[s:e].any(axis=0)
        cov = sum(rowink[a:b].any() for a, b in cells) / len(cells)
        cross = rowink[cuts].mean()                # 墨落在列切线上 = 连续跨列(标题)
        return cov >= _COV_MIN and cross <= _CROSS_MAX

    k = 0
    while k < len(bands) and not is_data_row(*bands[k]):   # 剥到第一真数据行
        k += 1
    if k == 0 or k >= len(bands):                  # 无前导标题 / 全非数据行 → 不剥
        return [], bb
    if k > 3:                                       # **保数据底线**:标题最多~3行,剥超过=列估计
        return [], bb                              #   出错误把大块数据判成非数据(如 f23fb56f
        #   1483px 数据被剥) → 宁可不剥标题,绝不切数据
    return [(x0, y0, x1, y0 + bands[k][0])], (x0, y0 + bands[k][0], x1, y1)


def crop(im):
    """Stage I 入口:返回阅读顺序的 [(kind, bbox)]。纯几何,不调 API。

    单表折叠(墨判据):subtables 若把一张稀疏单表切成 1 密集 seg + N 稀疏薄条,整图
    其实是单表 → 把所有 seg 并成一块(交给 Stage II 整块读满列宽,如 80995347 读全 107 行)。
    只数密集 seg(墨≥1%)决定是否多子表,稀疏薄条不计入——避免 94352240 被当多子表切碎。"""
    blocks = subtables(im)
    texts = [bb for k, bb in blocks if k == 'text']
    segs = [bb for k, bb in blocks if k == 'seg']
    if len(segs) >= 2:
        g = np.asarray(im.convert("L")) < BIN_INK
        dense = [bb for bb in segs if _ink(g, bb) >= _DENSE_INK]
        if len(dense) <= 1:                              # ≤1 密集真表 → 单表,合回一块
            segs = [_union(segs)]
    seg_items = []                                       # 剥每个 seg 顶部标题(→ 'title' 块)
    for bb in segs:
        titles, seg = _peel_title(im, bb)
        seg_items += [('title', tb) for tb in titles]
        seg_items.append(('seg', seg))
    items = [('text', bb) for bb in texts] + seg_items
    items = [(k, bb) for k, bb in items if bb[2] > bb[0] and bb[3] > bb[1]]  # 丢退化空块
    items.sort(key=lambda kb: (kb[1][1], kb[1][0]))      # 阅读顺序 (y, x)
    return items or [('seg', (0, 0, im.width, im.height))]


def dump_plan(im, blocks, out_dir):
    """把 Stage I 裁块落盘供人工审阅:overlay.jpg(彩框标注)+ crop_NN.jpg + manifest.json。"""
    os.makedirs(out_dir, exist_ok=True)
    ov = im.convert("RGB").copy()
    dr = ImageDraw.Draw(ov)
    color = {'text': (0, 128, 255), 'seg': (255, 0, 0), 'title': (0, 200, 0)}
    manifest = []
    for i, (kind, bb) in enumerate(blocks):
        c = color.get(kind, (0, 255, 0))
        dr.rectangle(bb, outline=c, width=4)
        dr.text((bb[0] + 4, bb[1] + 4), f"{i}:{kind}", fill=c)
        cr = im.crop(bb)
        if cr.width > 0 and cr.height > 0:
            cr.save(os.path.join(out_dir, f"crop_{i:02d}_{kind}.jpg"))
        manifest.append({"order": i, "kind": kind, "bbox": [int(v) for v in bb]})
    ov.save(os.path.join(out_dir, "overlay.jpg"))
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"blocks": manifest}, f, ensure_ascii=False, indent=2)
    return out_dir
