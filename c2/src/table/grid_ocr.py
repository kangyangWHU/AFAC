# -*- coding: utf-8 -*-
"""Stage II — 骨架切割 OCR(v6 严格版):一把尺、一个判定、一个修复动作。

- **一把尺**:墨迹几何(cell_ink∪cell_gray)给出期望——E=有内容行,inkw_k=每行内容宽。
- **一个判定**(零容差,行级,禁众数禁±1):非空行数==|E| 且 每行有效宽==inkw_k(恒等)。
- **一个修复**:不合格 tile → 格级本地重读(PP-OCRv6s,cell_ocr)整块替换;行=E、列=骨架,
  结构由代码给定,VLM 计数病(劈裂/口吃/塌缩/漏行/爆行/展平/截断/缺头)结构性不可能。
  colspan 行(墨迹横穿内部列边界,逐格裁剪会切碎)保留 API 原读,缺失只 audit。
- **骨架以估计为准**:列校准降为审计模式(投票只上报不采纳——均匀口吃能骗过票数)。
- 读数层只收货不救货(EMPTY/FLAT/截断/塌缩一律交判定层);修复结果不写 API 缓存。
- 列错位表(rows_misaligned)仍回退整段自由读(交上层 ocr_table)。
- 配套:列级数字格式归一(_norm_numeric,只治本地重读格的分隔符损伤,数字位不动)。
"""
import re

import numpy as np
from collections import Counter

from common.config import BIN_INK, BIN_FAINT
from table.geom import row_bnds, col_bnds, rows_misaligned, band_blank
from table.tiles import (MAX_TILE_ROWS, MAX_TILE_COLS, ASPECT_SAFE,
                         upsample_for, pad_white, call_tiles)
from table.stitch_single import parse_tile, COLSPAN
from table.cell_ocr import read_cells, read_strip


def _chunk(bnds, max_cells):
    """把边界序列切成 tile 带:每带 ≤max_cells 个单元格,**均匀分配**(避免 15/15/6 碎尾,
    band 大小一致 API 读得更稳)。返回 [(lo_idx, hi_idx), ...](骨架索引)。"""
    n = len(bnds) - 1
    nb = max(1, -(-n // max_cells))          # ceil
    size = -(-n // nb)
    out, i = [], 0
    while i < n:
        j = min(n, i + size)
        out.append((i, j))
        i = j
    return out


def _split_merged_cols(im, dark, rb, cb, meta):
    """漏检列线的骨架级修复(几何候选 + 本地模型内容复核,取代列校准投票采纳):

    几何候选:骨架列内墨呈双簇、簇间空隙**跨行对齐**(≥80%有缝格包含中位缝心±2px)
    = 疑似两个真实列被并读(a4e24107 行号列+值列,缝[48,60]逐行对齐,'93421.59'
    连体格的根因)。护栏:缝宽≥8px、两侧墨簇各≥10px、有缝格占有墨格≥70%
    (口吃表 6ec325a0 无对齐缝,天然不触发)。

    内容复核(几何的盲区):"88. 81" 式**小数点后空格**排版(186ee68f 全表逐列
    对齐缝)与真漏检几何同构,只能靠内容分辨——抽样 6 格本地 rec,
    『整数 + 完整小数』(9 3421.59)≥2/3 → 真漏检,采纳插线;其余(88. 81 /
    文本/单数)→ 拒绝。命中的边界插入缝心。"""
    # 真合并格形态:纯整数 + 第二簇——数字值(9 3421.59)或非数字标签(2男,rec常不给
    # 空格)。"88. 81"小数空格型整数后紧跟'.',两分支都不中
    two_num = re.compile(r"\d{1,4}(\s+[\d,]+(\.\d+)?|\s*[^\s\d.,]\S*)$")
    inserts = []
    for j in range(len(cb) - 1):
        x0, x1 = cb[j] + 2, cb[j + 1] - 2
        if x1 - x0 < 30:
            continue
        rows_ = [i for i in range(len(rb) - 1)
                 if dark[rb[i] + 2:rb[i + 1] - 2, x0:x1].any()]
        if len(rows_) < 4:
            continue
        gaps = []
        for i in rows_:
            proj = dark[rb[i] + 2:rb[i + 1] - 2, x0:x1].any(0)
            xs = np.where(proj)[0]
            if len(xs) < 2:
                continue
            inner = np.where(~proj[xs[0]:xs[-1] + 1])[0]
            if len(inner) == 0:
                continue
            runs = np.split(inner, np.where(np.diff(inner) > 1)[0] + 1)
            best = max(runs, key=len)
            lo, hi = xs[0] + best[0], xs[0] + best[-1]
            if (hi - lo + 1 >= 8 and lo - xs[0] >= 10 and xs[-1] - hi >= 10):
                gaps.append((lo, hi))
        if len(gaps) < 0.7 * len(rows_):
            continue
        cx = int(np.median([(lo + hi) // 2 for lo, hi in gaps]))
        cover = sum(1 for lo, hi in gaps if lo <= cx - 2 and cx + 2 <= hi)
        if cover < 0.8 * len(gaps):
            continue
        samp = rows_[:: max(1, len(rows_) // 6)][:6]     # 内容复核抽样
        vals = read_cells(im, rb, cb, [(i, j) for i in samp])
        hits = sum(1 for t in vals.values() if two_num.fullmatch(t.strip()))
        if hits * 3 >= len(vals) * 2:                    # ≥2/3 双数格
            inserts.append(x0 + cx)
            meta.setdefault("adopt", []).append(
                f"骨架col{j} 缝隙对齐插线@x{x0 + cx}"
                f"({cover}/{len(rows_)}行,复核{hits}/{len(vals)})")
        else:
            meta.setdefault("audit", []).append(
                f"骨架col{j} 对齐缝复核未过({hits}/{len(vals)}双数格),不插线")
    if inserts:
        return np.asarray(sorted(set(list(cb) + inserts)))
    return cb


def slice_grid(im):
    """按几何骨架切 tile。返回 (tiles, meta);tiles[r][c]=PIL.Image|None(空白)。

    meta: rows/cols(骨架数), rb/cb(边界px), row_bands/col_bands(骨架索引带),
          upsample, misaligned(True=骨架不可信,调用方回退自由读)。"""
    g = np.asarray(im.convert("L"))
    dark, dark180 = g < BIN_INK, g < BIN_FAINT
    rb, _ = row_bnds(dark, dark180)
    cb, cf = col_bnds(dark, dark180)
    meta = {"misaligned": False, "col_framed": bool(cf)}
    if len(rb) >= 5 and len(cb) >= 2:
        cb = _split_merged_cols(im, dark, rb, cb, meta)   # 漏检列线修复(见函数注释)
    R, C = len(rb) - 1, len(cb) - 1
    meta.update({"rows": R, "cols": C, "rb": rb, "cb": cb})
    if R < 1 or C < 1 or (R >= 2 and float(np.median(np.diff(rb))) >= 12
                          and rows_misaligned(dark, dark180)):
        # 错位检测仅在行距≥12px时有效:微距表(行缝2~3px)窄列条里数不出行,
        # "列带行数<<全表"是检测器自身失明,非真错位(A榜e082df7b 354x110
        # 对齐巨表被误标→整段自由读灾难)。真错位表b326/b5cad行距27px不受影响
        meta["misaligned"] = True
        return [], meta
    # 密集判据(与 slicer 共用 upsample_for):每 cell 平均边长小 → 上采样
    edge = (g.shape[0] * g.shape[1] / max(1, R * C)) ** 0.5
    meta["upsample"] = upsample_for(edge)
    # 硬上限 行25×列15(上榜版参数,让渡已废——双田实测密集表35列≈69%/15列≥99.5%,
    # 让渡为省调用把列放大到37~187是密集内容错误的第一元凶;矮宽表也不例外)
    row_bands = _chunk(rb, MAX_TILE_ROWS)
    col_bands = _chunk(cb, MAX_TILE_COLS)
    # 一律 tile+拼接,不整读(小段快路已退役:整读的"单tile无冗余+极端宽比"一次失败全丢)。
    # 像素长宽比约束:API 拒收 >200:1(400)。矮表单tile可到 241:1(1de69d49 尾表 2行
    # 7000×29px)——列带宽 > 180×最矮行带高 时对半加密列带(切在列边界上,骨架拼接原生
    # 支持多tile;不垫白,内容无损)
    min_bh = min(rb[j] - rb[i] for i, j in row_bands)
    while col_bands:
        wmax = max(cb[j] - cb[i] for i, j in col_bands)
        if wmax <= ASPECT_SAFE * max(1, min_bh) or all(j - i <= 1 for i, j in col_bands):
            break
        col_bands = _chunk(cb, max(1, -(-max(j - i for i, j in col_bands) // 2)))
    meta["row_bands"], meta["col_bands"] = row_bands, col_bands
    tiles = []
    for (ri, rj) in row_bands:
        row = []
        for (ci, cj) in col_bands:
            y0, y1 = rb[ri], rb[rj]
            x0, x1 = cb[ci], cb[cj]
            # 空白 tile 判定(geom.band_blank,与 crop 空段同一把尺):严档128计墨,
            # 空 tile 不调 API,由装配按骨架补空 cell
            if band_blank(dark[y0:y1, x0:x1], dark180[y0:y1, x0:x1]):
                row.append(None)
                continue
            # 精确按边界裁剪+白pad:边界都在缝中心/框线上,字不贴边,白边只提供视觉
            # 余量,邻带墨物理进不来(实pad会带残影,见 pad_white)。
            row.append(pad_white(im.crop((x0, y0, x1, y1))))
        tiles.append(row)
    return tiles, meta


def _parse_cap(raw):
    """解析 tile 输出 → (caption, rows)。tile 是从表内切出的,**不存在表外文字**——
    API 把跨列表头(colspan行,如"保单年度")当 caption 放在 <table> 前,必须回收。"""
    rows = parse_tile(raw) if raw else []
    cap = ""
    if raw and "<table" in raw.lower():
        head = raw[:raw.lower().index("<table")]
        cap = head.strip()
    return cap, rows


def _w(row):
    """行有效宽 = 到最后一个非空格的位置(尾部去空——API 常给稀疏行拖尾部空 <td>,
    那是零信息,不算内容宽)。"""
    last = max((k for k, s in enumerate(row) if s.strip()), default=-1)
    return last + 1


def span_cross(seg_gray, rb, cb, ri, rj, ci, cj):
    """跨列笔画检测(纯几何):返回 {骨架行 i: [被横穿的边界 j, ...]}。
    横穿 = 同一像素行的连通笔画过边界中心且两侧各有墨(≥2 像素行);贴线不算
    (密排数字蹭线左右不同时过中心,bc9e6b5d 假阳性教训);竖框线剔除
    (中心条整行高占比>0.8 的像素列=线)。
    用途:压线行的条带合并**只跨被横穿的边界**——没被横穿的格间照旧逐格,
    数据行即使被幻影窄列牵连标为压线,其数据格也不会被误并(7cd180ab 教训)。"""
    dark = seg_gray < BIN_INK
    out = {}
    for i in range(ri, rj):
        y0, y1 = rb[i] + 2, rb[i + 1] - 2
        if y1 <= y0:
            continue
        for j in range(ci + 1, cj):
            x = cb[j]
            win = dark[y0:y1, max(0, x - 6):x + 7]
            if win.size == 0:
                continue
            hline = win.mean(1) > 0.9          # 横线像素行(近全黑)整行剔除,
            left = dark[y0:y1, max(0, x - 6):max(0, x - 2)]   # 否则表头下沿横线
            cent = dark[y0:y1, max(0, x - 1):x + 2]           # 让每条边界都像被
            right = dark[y0:y1, x + 3:x + 7]                  # 横穿(7cd180ab r2)
            if min(left.size, cent.size, right.size) == 0:
                continue
            frac = cent.mean(axis=0)
            cent = cent[:, frac <= 0.8]
            if cent.size == 0:
                continue
            if (left.any(1) & cent.any(1) & right.any(1) & ~hline).sum() >= 2:
                out.setdefault(i, []).append(j)
    return out


def span_rows(seg_gray, rb, cb, ri, rj, ci, cj):
    """压线行号列表(span_cross 的行投影,审计工具用)。"""
    return sorted(span_cross(seg_gray, rb, cb, ri, rj, ci, cj))


# ═════════════════════════ 读数层(调 API + 解析 + 格式废品抢救) ═════════════════════════

def _ink_evidence(im, rb, cb, R, C):
    """格级墨证据:cell_ink[i][j] = 骨架格(i,j)内部(收缩2px避开框线)文字墨≥3px。
    行对齐(哪些行有内容)和列摆放(哪些格有内容)共用——空白判定要求极低墨。
    cell_gray = 淡灰内容(灰度129~180):d1752e16整张数字印成浅灰,128全隐形→cell_ink
    判空→E欠数→裁多丢真值。180看得见,配API门控救回(仅当API确认时启用,_align_tile)。"""
    _gray = np.asarray(im.convert("L"))
    dark = _gray < BIN_INK
    dark180 = _gray < BIN_FAINT
    cell_ink = np.zeros((R, C), dtype=bool)
    cell_gray = np.zeros((R, C), dtype=bool)
    for i in range(R):
        y0, y1 = rb[i] + 2, rb[i + 1] - 2
        if y1 <= y0:
            y0, y1 = rb[i], rb[i + 1]
        for j in range(C):
            x0, x1 = cb[j] + 2, cb[j + 1] - 2
            if x1 <= x0:
                x0, x1 = cb[j], cb[j + 1]
            sub = dark[y0:y1, x0:x1]
            cnt = sub.sum(1)
            frac = sub.mean(1)
            line = frac > 0.5                      # 穿过格内的横线主体行
            lm = line.copy()                       # ±2px 膨胀:线的反锯齿灰边(frac
            for s in (1, 2):                       #  0.2~0.5,几十px)也一并排除,否则
                lm[:-s] |= line[s:]                #  保单年度底线的灰边使 row0 被误判
                lm[s:] |= line[:-s]                #  有字,数据行错进表头(1674392a)
            cell_ink[i, j] = bool(((cnt >= 3) & ~lm).any())
            if not cell_ink[i, j]:                 # 128判空的格才查灰(省算):180去线cnt≥3
                sg = dark180[y0:y1, x0:x1]
                cg = sg.sum(1); fg = sg.mean(1)
                lg = fg > 0.7                      # 灰阈下线更满,门更高防框线灰边
                lmg = lg.copy()
                for s in (1, 2):
                    lmg[:-s] |= lg[s:]; lmg[s:] |= lg[:-s]
                cell_gray[i, j] = bool(((cg >= 3) & ~lmg).any())
    return cell_ink, cell_gray


def _read_tiles(tiles, meta, timeout):
    """读数层只收货不救货:批量调用 + 解析,一次一批,零抢救。
    EMPTY/FLAT/截断/塌缩/缺头全部原样交给判定层——不合格自然落入本地重读。
    返回 parsed[(r,c)] = (caption, rows)。"""
    flat = [(r, c) for r in range(len(tiles)) for c in range(len(tiles[r]))
            if tiles[r][c] is not None]
    outs = dict(zip(flat, call_tiles([tiles[r][c] for r, c in flat],
                                     timeout=timeout, upsample=meta["upsample"])))
    return {(r, c): _parse_cap(outs.get((r, c)))
            for r in range(len(meta["row_bands"]))
            for c in range(len(meta["col_bands"]))}


# ═════════════ 修复层(原则:孤立异常=tile 错→修 tile;多 tile 一致异常=骨架错→改期望) ═════════════

def _calibrate_cols(parsed, meta):
    """**列校准**(骨架级修复,一致性仲裁。行列职责不对称:行估计=真值,列在稀疏区/标签区
    可能少):非空 tile 的行格数众数若一致 = 骨架列数+k(k>0),且 ≥2 个行带的 tile 同票
    (或该列带只有 1 个非空 tile) → 采纳 nc+k(5fdf46b0 三标签列被并 1 列,429 行每行
    一致多读 2 格=最强信号;稀疏空 tile 不投票)。返回各列带的期望列数 band_nc。"""
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    band_nc = []
    for c, (ci, cj) in enumerate(col_bands):
        nc = cj - ci
        votes = []
        if not meta.get("col_framed"):             # **只校准无框表**:有框列=框线,本就精确;
            for r in range(len(row_bands)):        # 且有框 tile 切点在框线上,±3px pad 带进
                _, rws = parsed[(r, c)]            # 框线+邻列残影,VLM 每行一致幻觉出一个
                if not rws:                        # 边缘格(9c7857f3 五个带全被投成+1,
                    continue                       # 数据中间散布假空格)——一致性骗过佐证
                cnts = [len(x) for x in rws if any(s.strip() for s in x)]
                if cnts:
                    votes.append(Counter(cnts).most_common(1)[0][0])
        if votes:
            top, n = Counter(votes).most_common(1)[0]
            if nc < top <= nc + 3 and n >= 2:
                # v6 审计模式:不采纳,只上报。均匀口吃(+1整带一致)能骗过票数判据
                # (6ec325a0 三票"13→14"实为每行复读一格,采纳后=空列+幻读列);
                # 真漏检列线(5fdf 型 +2,新增列内容互不重复)再现时凭本 audit 名单
                # 加内容判据后再启用采纳。
                meta.setdefault("audit", []).append(
                    f"列带{c} 疑漏检列线 骨架{cj - ci}→{top}({n}票) 未采纳(审计模式)")
        band_nc.append(nc)
    return band_nc


def _norm_numeric(grid, marks):
    """列级数字格式归一(只治本地重读格的分隔符损伤,数字位一律不动):
    每列以**非重读格**的多数数字格式为模板(千分位有无 + 小数位数),重读格(marks)
    中字符仅含[数字.,空格]者按模板重写——先剥掉全部非数字字符,再按模板重排分隔符。
    重写前后数字序列恒等(构造保证),模板票数<3 的列不动(证据不足)。"""
    marks = set(marks)
    if not marks or not grid:
        return
    pat = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.(\d+))?")
    ncols = max(len(r) for r in grid)
    for j in range(ncols):
        votes = []
        for i, row in enumerate(grid):
            if j >= len(row) or (i, j) in marks:
                continue
            m = pat.fullmatch(row[j].strip())
            if m:
                votes.append(("," in row[j], len(m.group(1) or "")))
        if not votes:
            continue
        (has_c, ndec), n = Counter(votes).most_common(1)[0]
        if n < 3:
            continue
        for i, row in enumerate(grid):
            if (i, j) not in marks or j >= len(row):
                continue
            v = row[j].strip()
            if (not v or not re.fullmatch(r"[0-9.,\s]+", v)
                    or not any(ch.isdigit() for ch in v)):
                continue
            digits = re.sub(r"\D", "", v)
            if ndec:
                if len(digits) <= ndec:
                    continue
                intpart, dec = digits[:-ndec], digits[-ndec:]
            else:
                intpart, dec = digits, ""
            intpart = f"{int(intpart):,}" if has_c else str(int(intpart))
            row[j] = intpart + ("." + dec if dec else "")


# ═════════════════════════ 装配层(纯摆放,零 API 零修复) ═════════════════════════

def _assemble(im, parsed, meta, band_nc, cell_ink, cell_gray):
    """装配 = 零容差判定 + 本地重读 + 骨架摆放(v6,唯一修复动作在此):
    · 判定(行级,禁众数禁容差):非空行数==|E| 且 每行有效宽∈[inkw_k, nc];
      E/inkw 用 ink∪gray 双档墨(浅灰整表 d1752e16 不再依赖 API 作证)。
    · 不合格 → 格级本地重读整 tile 替换(行=E、列=骨架);colspan 行(span_rows)
      保留 API 原读——逐格裁剪会切碎跨列文字,缺失只 audit 不猜。
    · 摆放只信骨架:按 E 位置放行、溢出弃空格、按位补空;真损失 audit 上报不静默。
    · meta['local_cells'] 记录本地重读格(grid 坐标),供 _norm_numeric 归一。
    返回 (grid, cap_rows_global)。"""
    rb, cb = meta["rb"], meta["cb"]
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    cell_has = cell_ink | cell_gray            # 双档墨证据合一(E/inkw/重读格共用一把尺)
    seg_gray = np.asarray(im.convert("L"))
    grid = []
    cap_rows_global = []                       # (拆表条件1)非首带出现caption的带 → 其全局行范围
    local_marks = []                           # 本地重读格的 grid 坐标(归一用)
    col_off = [sum(band_nc[:c]) for c in range(len(col_bands))]
    for r, (ri, rj) in enumerate(row_bands):
        band_idx = list(range(ri, rj))
        aligned = {}
        for c, (ci, cj) in enumerate(col_bands):
            _, rows = parsed[(r, c)]
            E = [i for i in band_idx if cell_has[i, ci:cj].any()]
            nc = band_nc[c]
            nzrows = [x for x in rows if any(s.strip() for s in x)]

            def _occ_ok(row, hasrow):
                # 占位恒等:每格 非空⟺有墨,逐格判。宽度恒等是它的一维投影——
                # 只查最后非空位置会漏"前/中段占位缺失"(b681f78d 13x7 方洞:
                # API 行前7格空、右侧有值,宽==墨迹宽照样通过)。口吃复制值
                # (6ec325a0)、幻觉填充也全被占位差抓住。
                for j in range(len(hasrow)):
                    if bool(j < len(row) and row[j].strip()) != bool(hasrow[j]):
                        return False
                return not any(s.strip() for s in row[len(hasrow):])
            ok = (len(nzrows) == len(E)
                  and all(_occ_ok(x, cell_has[i, ci:cj])
                          for x, i in zip(nzrows, E)))
            local_k = set()
            if not ok and E:
                # 整 tile 本地重读,无例外(曾有"colspan行保留API原读"豁免,按行序
                # 取行依赖行流与E对齐——不合格tile恰恰对不齐,7cd180ab错位复制教训)。
                # 条带合并只跨**被横穿的边界**:相邻格仅当其间边界确有笔画横穿才并入
                # 同一条带整条识别(跨列文字不切碎、行首空档保真);未被横穿的格间
                # 照旧逐格。**范围限定表首带**(用户拍板):跨列表头/说明行只住在表顶,
                # 数据区的横穿一律视为假信号(幻影列/残线),不做任何合并。
                cross = (span_cross(seg_gray, rb, cb, ri, rj, ci, cj)
                         if r == 0 else {})
                if cross:
                    meta.setdefault("audit", []).append(
                        f"tile[{r}][{c}] 压线行{sorted(cross)} 跨界条带重读")
                cells, strips = [], {}          # strips[i] = [(j0, j1)] 多格条带
                for i in E:
                    bset = set(cross.get(i, []))
                    j = ci
                    while j < cj:
                        if not cell_has[i, j]:
                            j += 1
                            continue
                        j1 = j
                        while (j1 + 1 < cj and (j1 + 1) in bset
                               and cell_has[i, j1 + 1]):
                            j1 += 1
                        if j1 > j:
                            strips.setdefault(i, []).append((j, j1))
                        else:
                            cells.append((i, j))
                        j = j1 + 1
                vals = read_cells(im, rb, cb, cells)
                reasons = (f"实{len(nzrows)}行/期{len(E)}" if len(nzrows) != len(E)
                           else "占位不符")
                nzrows = []
                for k, i in enumerate(E):
                    row = [vals.get((i, j), "") for j in range(ci, cj)]
                    for (j0, j1) in strips.get(i, []):
                        row[j0 - ci] = read_strip(im, rb, cb, i, j0, j1)
                    if i in strips:
                        # 条带行 colspan 化(用户口径:行首空档并入首簇,格间/尾部
                        # 空档并入左簇——GT 表头 colspan 直贯,d9a99684 colspan=46):
                        # 内容格为界分段,段内其余格位放哨兵,渲染时折叠为 colspan td
                        starts = [j for j, s in enumerate(row) if s.strip()]
                        if starts:
                            texts = [row[s] for s in starts]
                            row = [COLSPAN] * len(row)
                            for b, t in zip([0] + starts[1:], texts):
                                row[b] = t
                    nzrows.append(row)
                    if i not in strips:
                        local_k.add(k)
                meta.setdefault("adopt", []).append(
                    f"tile[{r}][{c}] 严判不合格({reasons})→格级本地重读"
                    f"({len(cells)}格,PP-OCRv6s)")
            aligned[c] = (E, nzrows, local_k)
        # 摆放:行以骨架为准,按 E 位置放行、按位补空
        pos = "首带" if r == 0 else ("末带" if r == len(row_bands) - 1 else "中带!")
        for i in band_idx:
            rowcells = []
            for c, (ci, cj) in enumerate(col_bands):
                E, rows, local_k = aligned[c]
                k = E.index(i) if i in E else -1
                cells = rows[k] if 0 <= k < len(rows) else []
                nc = band_nc[c]
                if len(cells) > nc:                # 溢出弃空格:先丢空格格(零信息)再截尾
                    need = len(cells) - nc         # (仅零容差豁免的 colspan 行可能超宽)
                    kept = []
                    for s in cells:
                        if need and not s.strip():
                            need -= 1; continue
                        kept.append(s)
                    cells = kept
                if k in local_k:
                    for j in range(min(nc, len(cells))):
                        if cells[j].strip():
                            local_marks.append((len(grid), col_off[c] + j))
                rowcells += list(cells[:nc]) + [""] * max(0, nc - len(cells))
            grid.append(rowcells)
            # 补空丢行 audit:骨架判此行有内容,装配出的这行却全空 = 真丢,如实上报
            if cell_has[i].any() and not any(s.strip() for s in rowcells):
                meta.setdefault("audit", []).append(
                    f"骨架行{i}({pos}) 补空丢行 墨证据判有内容却装配成全空")
        for c, (ci, cj) in enumerate(col_bands):   # cap 记录(拆表条件1,原行为保留)
            if parsed[(r, c)][0] and r > 0:
                cap_rows_global.append((len(grid) - len(band_idx), len(grid)))
                break
    meta["local_cells"] = local_marks
    return grid, cap_rows_global


def _find_splits(grid, cap_rows_global):
    """表边界(用户双条件): ①非首带tile出现caption ②该带行流含轴行(连续整数序列>10)
    → 在轴行处拆表(deb8de95 '二年交'新块自带[0..14]轴行;1674392a 轴行在首带且
      caption带无轴行,双条件互锁不误拆)"""
    def _axis_at(i):
        vals = [c.strip() for c in grid[i] if c.strip()]
        run = best = 0
        prev = None
        for v in vals:
            if v.isdigit():
                n = int(v)
                run = run + 1 if (prev is not None and n == prev + 1) else 1
                prev = n
                best = max(best, run)
            else:
                prev = None
                run = 0
        return best > 10
    splits = []
    for lo, hi in cap_rows_global:
        for i in range(max(1, lo), min(hi, len(grid))):
            if _axis_at(i) and i not in splits:
                splits.append(i)
    return sorted(set(splits))


# ═════════════════════════ 编排 ═════════════════════════

def ocr_seg(im, timeout=240):
    """单个 seg 的骨架 OCR — v6 三段式:读数 → 零容差判定+本地重读(装配内) → 摆放。
    返回 (grid, ncalls, meta);grid=None 表示骨架不可信需回退(交上层 ocr_table 自由读)。

    · 骨架 = 行列估计(slice_grid);misaligned → 早退。
    · 读数层(_read_tiles):调用+解析,不救货。
    · 列校准(_calibrate_cols):唯一保留的骨架修复(漏检列线,读数投票≥2票)。
    · 装配层(_assemble):零容差判定,不合格→格级本地重读(colspan 行保留 API 原读);
      骨架摆放,真损失 audit。表边界 _find_splits 轴行拆表。
    · _norm_numeric:本地重读格分隔符按列多数格式归一(数字位不动)。"""
    tiles, meta = slice_grid(im)
    if meta["misaligned"]:
        return None, 0, meta
    rb, cb = meta["rb"], meta["cb"]
    cell_ink, cell_gray = _ink_evidence(im, rb, cb, meta["rows"], meta["cols"])
    parsed = _read_tiles(tiles, meta, timeout)
    band_nc = _calibrate_cols(parsed, meta)
    grid, cap_rows_global = _assemble(im, parsed, meta, band_nc, cell_ink, cell_gray)
    # 首行单内容(GT 口径,table集实证):短文本=悬浮轴标签(保单年度/投保年龄...)
    # → 折入下一行角格"角\标签"并删行(96/115 斜线角格主流);长文本=标题
    # → 整行 colspan(34e53b1c 标题+副标题 colspan 连排)。用户拍板阈值:≤8字=短。
    if grid and len(grid[0]) > 1:
        ne = [j for j, s in enumerate(grid[0]) if s.strip() and s != COLSPAN]
        if len(ne) == 1:
            t = grid[0][ne[0]]
            nxt_hdr = (len(grid) > 1 and grid[1] and grid[1][0].strip()
                       and sum(1 for s in grid[1] if s.strip()) >= 3)
            if nxt_hdr and len(t) <= 8:
                grid[1][0] = grid[1][0] + "\\" + t
                grid.pop(0)
                cap_rows_global = [(max(0, lo - 1), hi - 1)
                                   for lo, hi in cap_rows_global]
                meta["local_cells"] = [(i - 1, j) for (i, j)
                                       in meta.get("local_cells", []) if i > 0]
                meta.setdefault("adopt", []).append(
                    f"悬浮轴标签'{t}'折入角格(斜线口径)")
            else:
                grid[0] = [t] + [COLSPAN] * (len(grid[0]) - 1)
    meta["splits"] = _find_splits(grid, cap_rows_global)
    _norm_numeric(grid, meta.get("local_cells", []))
    ncalls = sum(1 for row in tiles for t in row if t is not None)
    return grid, ncalls, meta
