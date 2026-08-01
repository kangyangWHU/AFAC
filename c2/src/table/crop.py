# -*- coding: utf-8 -*-
"""Stage I — 裁剪(纯几何,不调 OCR)。整图 → 有序裁块 [(kind, bbox)]。

kind:
  'text'  : 表外 furniture(页眉/页脚/水印/页码),Stage III 用 ocr_text 读纯文本。
  'seg'   : 候选表格区域,Stage II 用 grid_ocr.ocr_seg 骨架 OCR(列错位回退 ocr_table);
            是表还是标题由 Stage III 按 td 数定。
  'title' : 从 seg 顶部剥下的标题/副标题/colspan组表头(mark),Stage III 按 keep_title
            决定丢为文本 or 拼回 colspan 表头行。

三步:① split_table_texts 剥表外文字(geom) ② subtables 按子表缝切段 ③ _peel_title 剥表顶标题。
所有 OCR 相关判定(表/标题、表头小条合并)都在 Stage III,不在这里。
子表缝/框线/并排等几何原语见 geom.py。
"""
import os
import numpy as np

from table.geom import (split_table_texts, _runlen_lines, band_blank, panel_seam_xs,
                        row_bnds, col_bnds, LINE_COVER, FRAME_MIN_RUN,
                        DATA_SPAN_MIN, DATA_RUN_MIN)
from common.config import BIN_INK, BIN_FAINT
from common.imcache import cached


# ---------------------------------------------------------------------------
# 子表几何切分:把含多子表的整图切成单子表段
# ---------------------------------------------------------------------------
_SEAM_K = 2.5   # 缝高门槛(×行缝中位)。下限由**表内分组空隙**定(切了
#                 一表变多表,K=2 实测碎成过多段);上限由密排小表真缝定(
#                 K=3 实测粘连)。真管线 2.4~2.5 平台无差(peel/分类吸收刀刃),取 2.5 与
#                 历史实测"真缝=行缝2.5~10倍"吻合。标题-表空隙(3.0×+)切开无害:标题段
#                 由分类收编。
_SEAM_VMAX = 2


def _vline_break_ys(dark180):
    """竖线断裂找横缝:框线竖线(纵向长墨柱)在子表之间**全部断开**的 y。对称于 panel 的
    横线断裂找左右(这里纵向墨柱被横向白带打断找上下)。仅有框表(竖线≥3)。救"白带被稀疏
    行打断、缝高不足"的有框密集表。

    两条收紧(避免顶部抬头标题被切、切到文字):
    ① 只取**中间**的断裂带(3%<s<97%H)——排除顶部抬头标题/表尾,它们不是子表间。
    ② 切点取断裂带**顶部 s**(上子表框结束处)——标题归下子表、不切到标题文字。
    注:"只剩第1列"和"真子表标题区"的断裂带 cov 都=2、几何
    无法区分,故仍按<20%判断裂——宁可多切一刀(td 判后 table 仍=1、无害),也不能
    漏切(漏切=table 数错、TEDS 崩)。"""
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
      (如行号列纵向贯穿,列墨偏高)。
    · 框线列贯穿(高墨 >0.8)数 ≤ vmax:稀疏表底部空行边框/列线仍穿过(仍在表内),
      真子表缝处框线断开(= 0 条)。
    这条按"完全空白才是缝"替掉原魔数"段数<4"。"""
    H, W = dark.shape
    # **严格白**(与 geom 行估计同一把尺子:0.05%×宽+floor3):行号/标签行有墨→不算白,
    # 天然被排除在带外——阶梯稀疏区的空隙只剩 1.4~2.3×(行号不再被吞),缝中标签自动把
    # 带劈开、大空独立成缝、标签随下表。旧松白(0.3%W)需要整套"贯穿否决"来补救,已删。
    # 算行墨前**先去贯穿竖线列**(对称 geom.row_bnds):有框表的框线给每行贡献~10px墨,
    # 不去则没有一行能过严格白、bands 全空(白带0条,B探测器被早退跳过)。
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
        if not merged or c - merged[-1] >= 40:             #  120会吞相距很近的两条真缝,
            merged.append(c)                               #  小表高可只有百余px)
    if len(merged) > 15:                                   # 碎尸护栏:正常多子表≤11刀;
        return [0, H]                                      #  列错位表行缝被行号劈成
    #                                                        上百条过阈假缝 → 整栏不切,交OCR
    return [0] + merged + [H]


@cached("subtables", __file__,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "geom.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common", "preprocess.py"))
def subtables(im):
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
    colb = [0] + panel_seam_xs(g) + [W]
    for c in range(len(colb) - 1):                    # ② 主表内切子表段(不预判类型)
        cx0, cx1 = colb[c], colb[c + 1]
        rb = _row_bounds(dark[:, cx0:cx1], dark180[:, cx0:cx1])
        for ra, rbb in zip(rb[:-1], rb[1:]):
            band180 = dark180[ra:rbb, cx0:cx1]
            # 空段判据(geom.band_blank,与空tile同一把尺):松档180计墨(两行标题也算内容)
            if band_blank(band180, band180):
                continue
            bb = (x0 + cx0, y0 + ra, x0 + cx1, y0 + rbb)
            items.append((y0 + ra, 'seg', bb))   # 只切不判:类型由 crop 的结构判据(骨架格数)定
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
#   自己的框(熵很高)会漏,块内熵量的是"填框"不是"结构",弃。
#   薄宽真表(格数≥200)天然保住;稀疏条由"单表折叠先于分类"保护。
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
    竖线列——否则一条延伸到空白区的框线(只上/下边线、无数据)会把边界撑大(如上框线
    延伸到空白区、数据实际范围更小)。去线后仍保留的竖框线(带数据列)照常算边界,
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


_SPAN_MIN = DATA_SPAN_MIN   # 真数据行判据与 geom 共用(0.4:数据常不填满)
_RUN_MIN = DATA_RUN_MIN     # 列向独立墨段 ≥3;标题=1~2 连续块


def _datalike(g, band, cbnd, tol=3):
    """带是否【数据行样】(列对齐判据,替代 _PEEL_MAX 深度魔数的护栏):
    ≥2 个独立墨成分,且每个成分都落在**单个身体列区间**内(±tol 抗反锯齿灰边)
    = 稀疏数据行(三角顶:行号+值各居其列)→ 不可剥。
    标题的反例二形:长成分横跨多条列线('新华人寿…价值表'),或孤零单成分('单位:元')
    ——都不满足,判非数据、可剥。"""
    s, e = band
    cols = g[s:e].any(axis=0)
    runs = []
    x, W = 0, len(cols)
    while x < W:
        if cols[x]:
            x0 = x
            while x < W and cols[x]:
                x += 1
            runs.append((x0, x))
        else:
            x += 1
    # 字缝桥接:间隙 ≤ 0.6×行高(字号代理,自适应无绝对常数)的相邻墨段并成一个
    # 短语级成分——否则标题按【字】散成几十个小段,个个都能塞进宽列,误判数据行样
    bridge = max(2, int((e - s) * 0.6))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= bridge:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    runs = [(a, b) for a, b in merged]
    if len(runs) < 2:
        return False
    for (a, b) in runs:
        if not any(lo - tol <= a and b <= hi + tol
                   for lo, hi in zip(cbnd[:-1], cbnd[1:])):
            return False                    # 该成分跨列界 → 非数据
    return True


def _peel_title(im, bb):
    """剥 seg 顶部标题/副标题,返回 (title_bboxes, trimmed_seg_bbox)。剥出块由调用方标
    'title'(mark),Stage III 按 td 数定去留(整块无 td → 当标题/文本)。分两路:

    · **有横框线**(≥2,含密集网格表):最上一条=表格上边界,其上方(框外)=标题;cut=上边框,
      框内表头/数据在 cut 下方天然不被切。框线多≠无标题——标题就在最上框线之上。
      **最上框线上方无文字行** → 表从顶开始=无标题,不切;判据用上方**行墨峰值**(不用相对
      表高:大表 3%H 达数百px 会把顶部真标题误判贴顶,真标题绝对高度虽大却<3%H;也不用
      全宽均值:淡/短标题会被稀释,标题行墨虽有、全宽均值仍被拉低)。
    · **无框/单框**(难点):不靠列格(鸡生蛋),直接看行墨迹**横向分布**——数据行墨迹横跨
      全宽(span 大)且分成多列墨段(runs 多);标题墨迹局部集中(左/中/右)或单一连续块。
      从顶剥到第一数据行;剥 > _PEEL_MAX 行则判据可疑 → 不剥(保数据,不误切)。"""
    x0, y0, x1, y1 = bb
    gray = np.asarray(im.crop(bb).convert("L"))
    g, g180 = gray < BIN_INK, gray < BIN_FAINT
    W = g.shape[1]
    # 横框线检测:松二值化 g180(框线常淡灰,灰度128~180,严二值化会漏)
    hl, prev = [], -99
    for y in np.where(g180.mean(axis=1) > LINE_COVER)[0]:
        if y - prev > 3:
            hl.append(int(y))
        prev = int(y)
    # 路① 有横框线(≥2,含密集网格):最上一条=表顶,其上=标题(框线多≠无标题)
    if len(hl) >= 2:
        top = hl[0]
        if top < _PAD_GUARD or g180[:top].mean(1).max() < _TOP_TEXT_PEAK:  # 上方无文字行 → 无标题
            return [], bb
        return [(x0, y0, x1, y0 + top)], (x0, y0 + top, x1, y1)
    # 路② 无框/单框:按行墨迹横向分布判数据行(不依赖列格)
    # 行带用 row_bnds(与行计数**同一把尺子**,绝对墨判白):相对阈 _content_segs(0.4%×宽
    # ≈15px/行)看不见小标签("X岁"每行十几px墨)——标签对 peel 隐形、却被行计数数进去,
    # 堆叠小表家族全部 +1 行。row_bnds 的行带里
    # 标签作为 band0 现身 → is_data 判非数据 → 剥走。
    rbnd, _ = row_bnds(g, g180)
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
    while k < len(bands) and not is_data_row(*bands[k]):   # 从顶扫到第一数据行(不设深度上限)
        k += 1
    if k == 0 or k >= len(bands):                          # 无标题/全非数据 → 不剥
        return [], bb
    if k > 3:
        # 深标题栈(>3行)复核:旧版在此一刀切拒绝(深标题全留表内被逐格读碎);
        # 现改为证据门控——前缀带**全部非数据行样**(_datalike:≥2成分且各居单列=三角顶
        # 稀疏行签名)才剥。浅栈(≤3)保持原样零复核:一年验证过的行为,且干跑实测列对齐
        # 判据在宽列表上会误否决真标题(caption片段恰好各居宽列,多数为此类误伤)
        cbnd, _ = col_bnds(g[bands[k][0]:], g180[bands[k][0]:])
        if any(_datalike(g, bands[b], cbnd) for b in range(k)):
            return [], bb
    cut = bands[k][0]
    if cut < 8:            # <8px 的"标题"是上段字脚残余(pad重叠带入,几px的碎屑,
        return [], bb      #  超长宽比被 API 400 拒)——真标题行至少一个字高
    return [(x0, y0, x1, y0 + cut)], (x0, y0 + cut, x1, y1)


def crop(im):
    """Stage I 入口:返回阅读顺序的 [(kind, bbox)]。纯几何,不调 API。

    单表折叠(墨判据):subtables 若把一张稀疏单表切成 1 密集 seg + N 稀疏薄条,整图
    其实是单表 → 把所有 seg 并成一块(交给 Stage II 整块读满列宽)。
    只数密集 seg(墨≥1%)决定是否多子表,稀疏薄条不计入——避免稀疏单表被当多子表切碎。"""
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
        且纯标题段/垃圾条在此已被拦下,peel 不再接触文字——残影条反例被顺序消灭)"""
        g2 = np.asarray(im.crop(bb).convert("L"))
        d = g2 < BIN_INK
        if d.shape[0] < 8 or d.shape[1] < 8 or not d.any():
            return True
        d180 = g2 < BIN_FAINT
        r = len(row_bnds(d, d180)[0]) - 1
        c = len(col_bnds(d, d180)[0]) - 1
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
    items.sort(key=lambda kb: (kb[1][1], kb[1][0]))      # 阅读顺序 (y, x)
    return items or [('seg', (0, 0, im.width, im.height))]
