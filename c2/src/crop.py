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

from geom import (split_table_texts, _runlen_lines, _content_segs, LINE_FULL,
                  LINE_COVER, FRAME_MIN_RUN, DATA_SPAN_MIN, DATA_RUN_MIN,
                  rows_misaligned)
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


_SEAM_K = 2.5   # 缝高门槛(×行缝中位)。下限由**表内分组空隙**定(8ff15f9a 2.25×,切了
#                 一表变多表,K=2 实测碎成14段);上限由密排小表真缝定(225fb899 2.5~2.6×,
#                 K=3 实测粘连)。真管线 2.4~2.5 平台无差(peel/分类吸收刀刃),取 2.5 与
#                 历史实测"真缝=行缝2.5~10倍"吻合。标题-表空隙(3.0×+)切开无害:标题段
#                 由分类收编。
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
    vlines = _runlen_lines(dark180, min_run=FRAME_MIN_RUN)
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
    # **严格白**(与 geom 行估计同一把尺子:0.05%×宽+floor3):行号/标签行有墨→不算白,
    # 天然被排除在带外——阶梯稀疏区的空隙只剩 1.4~2.3×(行号不再被吞),缝中标签自动把
    # 带劈开、大空独立成缝、标签随下表。旧松白(0.3%W)需要整套"贯穿否决"来补救,已删。
    # 算行墨前**先去贯穿竖线列**(对称 geom.row_bnds):有框表的框线给每行贡献~10px墨,
    # 不去则没有一行能过严格白、bands 全空(1c9ac6d2 白带0条,B探测器被早退跳过)。
    keep = None
    if dark180 is not None:
        km = dark180.mean(axis=0) <= 0.5
        if km.any():
            keep = km
    d2 = dark[:, keep] if keep is not None else dark
    W2 = d2.shape[1]
    white = d2.sum(1) < max(3, 0.0005 * W2)
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
    cuts = []
    if bands:
        row_gap = float(np.median([h for _, _, h in bands]))   # 典型行缝高
        for s, e, h in bands:
            if h < k * row_gap or not (20 < (s + e) // 2 < H - 20):
                continue
            col = dark[s:e].mean(0)
            if int((col > 0.8).sum()) > vmax:              # 框线竖线贯穿 → 表内，非缝
                continue
            cuts.append((s + e) // 2)
    if dark180 is not None:                                # 判据三:竖线断裂(有框表)——
        cuts += _vline_break_ys(dark180)                   # 不因 bands 空而跳过
    cuts = sorted(set(cuts))
    merged = []                                            # 相邻(<40px)缝合并=同一缝去重
    for c in cuts:                                         # (40=同缝两探测器的抖动幅度;
        if not merged or c - merged[-1] >= 40:             #  120会吞相距114px的两条真缝,
            merged.append(c)                               #  a1aaef73 小表高仅~115px)
    if len(merged) > 15:                                   # 碎尸护栏:正常多子表≤11刀;
        return [0, H]                                      #  列错位表(b326)行缝被行号劈成
    #                                                        上百条过阈假缝 → 整栏不切,交OCR
    return [0] + merged + [H]


@cached("subtables", __file__, os.path.join(os.path.dirname(os.path.abspath(__file__)), "geom.py"))
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
            band180 = dark180[ra:rbb, cx0:cx1]
            # 空段判据=行投影(用户设计,与tile空白同一把尺):去线后每行墨<3px才是真空。
            # 旧均值判据尺度相关:全宽段里两行标题(~3000墨px/33万px≈0.009)被稀释成
            # "空白"静默吞掉,子表间标题整个从输出消失(A榜0cd74f08/3b243a83)
            vk = band180.mean(axis=0) <= LINE_FULL
            b2 = band180[:, vk] if vk.any() else band180
            rf_ = b2.mean(axis=1) if b2.size else np.zeros(1)
            rowink = b2[rf_ <= LINE_FULL].sum(axis=1) if (rf_ <= LINE_FULL).any() else np.zeros(1)
            if rowink.size == 0 or rowink.max() < 3:  # 真空白带,丢
                continue
            bb = (x0 + cx0, y0 + max(0, ra - pad), x0 + cx1, y0 + min(H, rbb + pad))
            items.append((y0 + ra, 'seg', bb))   # 只切不判:类型统一由 crop 的结构判据定
            #   (原矮段20px/淡段g128两条像素阈规则删除,其判别对象——1~3行的标题/说明——
            #    由"骨架行数<4"结构判据统一覆盖)
    items.sort(key=lambda t: t[0])                     # 阅读顺序
    return [(k, bb) for _, k, bb in items] or [('seg', tb)]


# ---------------------------------------------------------------------------
# 裁块规划:单表折叠 + 表顶标题剥离
# ---------------------------------------------------------------------------
_DENSE_INK = 0.01   # seg 墨量≥此=密集真表;<此=稀疏薄条,不计入子表计数
_JUNK_CELLS = 30      # seg 骨架格数(行×列) < 此 = 文字(标题/说明),否则留 seg 交 API。
#   格数=行列结构的直接证据。**不对称原则:表格归文字=致命(数据丢),文字归表格=无害
#   (OCR 读出 td<10 → Stage II 降级纠正,代价 1 次调用)**——故阈值取"毫无疑义是文字"
#   的下界 30(实测 junk 簇 ≤54、真表 ≥78;30~60 的模糊块留给 API 裁决)。
#   无高度门槛(h298~413 高个说明块照抓)。曾试 2D熵(墨迹分散度):紧裁的大字标题填满
#   自己的框(e3e19b92 熵0.95)会漏,块内熵量的是"填框"不是"结构",弃。
#   薄宽真表(1de69d49 2×100=200格)天然保住;稀疏条由"单表折叠先于分类"保护。
_TIGHT_INK = 0.001   # _tighten:行/列均墨 >此 = 有内容(抗单点噪,空区≈0)
_PAD_GUARD = 8       # pad 残余滤除带px:0<y<此 的框线是上一 seg 重叠带入的底框
_TOP_TEXT_PEAK = 0.003  # 贴顶判据:最上框线上方行墨峰值 <此 = 无文字行(无标题)


def _ink(g, bb):
    x0, y0, x1, y1 = bb
    return g[y0:y1, x0:x1].mean() if y1 > y0 and x1 > x0 else 0.0


def _union(bbs):
    xs0, ys0, xs1, ys1 = zip(*bbs)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _tighten(im, bb, drop_lines=False):
    """把裁块收紧到实际内容边界(去四周空白 margin)。用松二值化 g180。

    关键:定**左右界**时先去掉跨全宽的横线行(mean>0.5),定**上下界**时去掉贯穿全高的
    竖线列——否则一条延伸到空白区的框线(只上/下边线、无数据)会把边界撑大(如 1c9ac6d2
    上框线延伸到 x2685,数据实际只到 1672)。去线后仍保留的竖框线(带数据列)照常算边界,
    所以有框空单元格不丢。mean>0.001 判该行/列有内容(抗单点噪,空区≈0 不过)。
    drop_lines(title/text 专用):横线行在定上下界时也剔——文字块里的横线是邻表借来的
    边框,算内容会把标题框拉长贴到表格线上;表格块(seg)不启用,框线即其合法边界。"""
    x0, y0, x1, y1 = bb
    d = np.asarray(im.crop(bb).convert("L")) < BIN_FAINT
    dc = d.copy(); dc[d.mean(1) > LINE_COVER, :] = False   # 去横线行 → 定左右界
    dr = d.copy(); dr[:, d.mean(0) > LINE_COVER] = False   # 去竖线列 → 定上下界
    if drop_lines:
        dr[d.mean(1) > LINE_COVER, :] = False              # 文字块:横线行也不算内容
    cols = np.where(dc.mean(0) > _TIGHT_INK)[0]
    rows = np.where(dr.mean(1) > _TIGHT_INK)[0]
    if len(rows) == 0 or len(cols) == 0:
        return bb
    return (x0 + int(cols[0]), y0 + int(rows[0]),
            x0 + int(cols[-1]) + 1, y0 + int(rows[-1]) + 1)


_SPAN_MIN = DATA_SPAN_MIN   # 真数据行判据与 geom 共用(0.4:数据常不填满,如 4f27636c span0.54)
_RUN_MIN = DATA_RUN_MIN     # 列向独立墨段 ≥3;标题=1~2 连续块
_PEEL_MAX = 3       # 保数据底线:最多剥 ~3 行标题,剥超过=判据出错,宁可不剥


def _peel_title(im, bb):
    """剥 seg 顶部标题/副标题,返回 (title_bboxes, trimmed_seg_bbox)。剥出块由调用方标
    'title'(mark),Stage III 按 td 数定去留(整块无 td → 当标题/文本)。分两路:

    · **有横框线**(≥2,含密集网格表):最上一条=表格上边界,其上方(框外)=标题;cut=上边框,
      框内表头/数据在 cut 下方天然不被切。框线多≠无标题——标题就在最上框线之上。
      **最上框线上方无文字行** → 表从顶开始=无标题,不切;判据用上方**行墨峰值**(不用相对
      表高:大表 3%H 达数百px 会把顶部真标题误判贴顶,如 0e8a501f 标题229px<3%H;也不用
      全宽均值:淡/短标题会被稀释,如 1b8718d0 标题行墨0.0066 但全宽均值仅0.0017)。
    · **无框/单框**(难点):不靠列格(鸡生蛋),直接看行墨迹**横向分布**——数据行墨迹横跨
      全宽(span 大)且分成多列墨段(runs 多);标题墨迹局部集中(左/中/右)或单一连续块。
      从顶剥到第一数据行;剥 > _PEEL_MAX 行则判据可疑 → 不剥(保数据,不误切)。"""
    x0, y0, x1, y1 = bb
    gray = np.asarray(im.crop(bb).convert("L"))
    g, g180 = gray < BIN_INK, gray < BIN_FAINT
    W = g.shape[1]
    # 横框线检测:松二值化 g180(框线常淡灰,灰度128~180,严二值化会漏,如 4df76a25 淡线 y251)
    hl, prev = [], -99
    for y in np.where(g180.mean(axis=1) > LINE_COVER)[0]:
        if y - prev > 3:
            hl.append(int(y))
        prev = int(y)
    hl = [y for y in hl if y == 0 or y >= _PAD_GUARD]   # 滤 pad 残余(seg重叠带入上一seg底框线)
    # 路① 有横框线(≥2,含密集网格):最上一条=表顶,其上=标题(框线多≠无标题)
    if len(hl) >= 2:
        top = hl[0]
        if top < _PAD_GUARD or g180[:top].mean(1).max() < _TOP_TEXT_PEAK:  # 上方无文字行 → 无标题
            return [], bb
        return [(x0, y0, x1, y0 + top)], (x0, y0 + top, x1, y1)
    # 路② 无框/单框:按行墨迹横向分布判数据行(不依赖列格)
    # 行带用 row_bnds(与行计数**同一把尺子**,绝对墨判白):相对阈 _content_segs(0.4%×宽
    # ≈15px/行)看不见小标签("X岁"每行十几px墨)——标签对 peel 隐形、却被行计数数进去,
    # 堆叠小表家族全部 +1 行(ce5799a7×6/1de69d49×5/225fb899×4…)。row_bnds 的行带里
    # 标签作为 band0 现身 → is_data 判非数据 → 剥走。
    from geom import row_bnds as _rb
    rbnd, _fr = _rb(g, g180)
    bands = [(rbnd[i], rbnd[i + 1]) for i in range(len(rbnd) - 1)]
    if len(bands) < 3:
        return [], bb

    def is_data_row(s, e):
        cols = g[s:e].any(axis=0)                  # 该行各列是否有墨
        xs = np.where(cols)[0]
        if len(xs) == 0:
            return False
        span = (xs[-1] - xs[0] + 1) / W            # 横向跨度:数据行首列→末列≈1,标题局部小
        runs = int((np.diff(np.r_[np.int8(0), cols.view(np.int8), np.int8(0)]) == 1).sum())
        return span >= _SPAN_MIN and runs >= _RUN_MIN   # 铺满全宽 且 多列墨段 = 数据行

    k = 0
    while k < len(bands) and not is_data_row(*bands[k]):   # 从顶剥到第一数据行
        k += 1
    if k == 0 or k >= len(bands) or k > _PEEL_MAX:         # 无标题/全非数据/剥超上限 → 不剥
        return [], bb
    cut = bands[k][0]
    if cut < 8:            # <8px 的"标题"是上段字脚残余(pad重叠带入,如 945e8fe9 4px屑,
        return [], bb      #  4473×4=1118:1 被API 400拒)——真标题行至少一个字高
    return [(x0, y0, x1, y0 + cut)], (x0, y0 + cut, x1, y1)


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
    def _is_texty(bb):
        """发路条(在 peel 之前,对每个原始段):
        text ⟺ 格数<30(结构底线) ∨ ((行数<4 ∨ span<0.5) ∧ 格数<200)
        判为 seg 的才进 peel;peel 火过的躯干由构造保证是表,不再复审(出身原则,
        且纯标题段/垃圾条在此已被拦下,peel 不再接触文字——34e残影条反例被顺序消灭)"""
        g2 = np.asarray(im.crop(bb).convert("L"))
        d = g2 < BIN_INK
        if d.shape[0] < 8 or d.shape[1] < 8 or not d.any():
            return True
        from geom import row_bnds as _rb, col_bnds as _cb
        d180 = g2 < BIN_FAINT
        r = len(_rb(d, d180)[0]) - 1
        c = len(_cb(d, d180)[0]) - 1
        cells = r * c
        if cells < _JUNK_CELLS:
            return True                       # 结构底线:无二维结构恒文字
        xs = np.where(d.any(0))[0]
        span_small = (xs[-1] - xs[0] + 1) < 0.5 * d.shape[1]
        return (r < 4 or span_small) and cells < 200

    items = [('text', bb) for bb in texts]
    for bb in segs:
        if _is_texty(bb):                     # 先发路条
            items.append(('text', bb))
            continue
        titles, seg = _peel_title(im, bb)     # 只对表格剥标题;躯干免复审直接保seg
        items += [('title', tb) for tb in titles]
        items.append(('seg', seg))
    items = [(k, _tighten(im, bb, drop_lines=(k != 'seg'))) for k, bb in items]
    #        ^ title/text 收界时剔横线(邻表边框非内容);seg 框线即边界不剔
    items = [(k, bb) for k, bb in items if bb[2] > bb[0] and bb[3] > bb[1]]  # 丢退化空块
    # title 出口统一过滤:tighten 后高 <12px = 字脚残屑(pad 重叠带入上段末行底,如
    # 945e8fe9 4px屑,4473×4=1118:1 被 API 400 拒;真标题最矮 17px)——不论出生路径一律丢
    items = [(k, bb) for k, bb in items if not (k == 'title' and bb[3] - bb[1] < 12)]
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
