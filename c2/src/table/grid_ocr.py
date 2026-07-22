# -*- coding: utf-8 -*-
"""Stage II — 骨架切割 OCR:行列估计已准(几何层),tile 严格按估计边界切、结果严格按骨架组装。

原则(与几何层的分工):
- **一律以估计为准**:tile 切点=估计的行列边界(切在缝/线上,天然不劈单元格);OCR 返回
  与骨架不符 = 幻觉/漏读,中间 tile 强制对齐骨架(多裁少补);仅首/末 tile 容 ±1 行
  (开边表末行/表头两行等已知几何偏差),取 OCR 实读。
- **稀疏区补零**:多表头/三角表的稀疏区(除表头外基本空白)不靠 OCR 猜——空 tile(墨≈0)
  不调 API,直接按骨架补空 cell;半空 tile 的短行也按骨架列数补 "" 到位。
- 列错位表(rows_misaligned)是唯一例外:骨架不可信,回退整段自由读(交上层 ocr_table)。
"""
import re

import numpy as np
from collections import Counter
from PIL import Image

import common.api_client as api
from common.config import BIN_INK, BIN_FAINT
from table.geom import row_bnds, col_bnds, rows_misaligned, band_blank
from table.tiles import (MAX_TILE_ROWS, MAX_TILE_COLS, ASPECT_SAFE,
                         upsample_for, upscale, pad_white, call_tiles)
from table.stitch_single import parse_tile, rows_to_html


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


def slice_grid(im):
    """按几何骨架切 tile。返回 (tiles, meta);tiles[r][c]=PIL.Image|None(空白)。

    meta: rows/cols(骨架数), rb/cb(边界px), row_bands/col_bands(骨架索引带),
          upsample, misaligned(True=骨架不可信,调用方回退自由读)。"""
    g = np.asarray(im.convert("L"))
    dark, dark180 = g < BIN_INK, g < BIN_FAINT
    rb, _ = row_bnds(dark, dark180)
    cb, cf = col_bnds(dark, dark180)
    R, C = len(rb) - 1, len(cb) - 1
    meta = {"rows": R, "cols": C, "rb": rb, "cb": cb, "misaligned": False,
            "col_framed": bool(cf)}
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


def _grid_reread(im, rb, cb, r0, r1, c0, c1, ci, cj, timeout):
    """flat tile 抢救(方案B'):裁有墨包围盒[r0:r1]×[c0:c1] + 按骨架线画网格 → 重调,
    逼VLM从竖排flat变规整表(实测 baf9 '1000⏎0⏎0'→[1000,0],[0,空])。返回 padding 到
    full tile 列宽[ci:cj]的 rows。只对'API无<table>'的极少数废tile用,不碰正常tile。"""
    from PIL import ImageDraw
    core = im.crop((cb[c0], rb[r0], cb[c1], rb[r1])).convert("RGB")
    sc = 3                                          # 放大让网格线清晰、小字可读
    core = core.resize((core.width * sc, core.height * sc), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(core)
    for i in range(r0, r1 + 1):
        y = (rb[i] - rb[r0]) * sc
        d.line([(0, y), (core.width, y)], fill=(0, 0, 0), width=2)
    for j in range(c0, c1 + 1):
        x = (cb[j] - cb[c0]) * sc
        d.line([(x, 0), (x, core.height)], fill=(0, 0, 0), width=2)
    tt = pad_white(core)
    _, grows = _parse_cap(api.call_safe(tt, timeout=timeout))   # 已放大,不再 upscale
    left, right = c0 - ci, cj - c1                  # 包围盒外的列(全空)左右补齐到full tile
    return [[""] * left + g + [""] * right for g in grows]


def _reread_halved(im, rb, cb, cands, up, timeout, meta=None):
    """**列界对半重读原语**(自愈对半与截断修复共用;统一发送层一波齐发,零串行等待):
    每个 tile 沿骨架列界切左右两半、各读一次,左右行流按行号横拼(短侧补空)。
    读取即真值,不做二次质检——行列若与骨架不符,由装配的补空/裁多收束并 audit 记录。
    半块自身仍截断 = 未完全恢复(简化版无二次加密),audit 点名不静默(暴露原则)。
    cands = [(key, ri, rj, ci, cj)];返回 {key: joined_rows}(两半皆空 → [])。"""
    imgs, slots = [], []
    for k, (key, ri, rj, ci, cj) in enumerate(cands):
        mid = (ci + cj) // 2
        for j, (lo, hi) in enumerate(((ci, mid), (mid, cj))):
            if hi <= lo:
                continue
            imgs.append(upscale(pad_white(im.crop((cb[lo], rb[ri], cb[hi], rb[rj]))), up))
            slots.append((k, j))
    outs = api.call_many(imgs, timeout=timeout)
    hv = {k: [[], []] for k in range(len(cands))}
    for (k, j), o in zip(slots, outs):
        hv[k][j] = _parse_cap(o)[1]
        if meta is not None and o and api.is_truncated(o):
            meta.setdefault("audit", []).append(
                f"tile{cands[k][0]} 对半{'左' if j == 0 else '右'}半块仍截断⚠ 部分行未恢复")
    res = {}
    for k, (key, ri, rj, ci, cj) in enumerate(cands):
        mid = (ci + cj) // 2
        hl, hr_ = hv[k]
        n_ = max(len(hl), len(hr_)) if (hl or hr_) else 0
        res[key] = [(hl[i] if i < len(hl) else [""] * (mid - ci))
                    + (hr_[i] if i < len(hr_) else [""] * (cj - mid))
                    for i in range(n_)]
    return res


_DEC_SP = re.compile(r"(?<=\d)\.\s+(?=\d)")   # '914. 2'(小数点后误加空格)→'914.2'


def _numeric(t):
    return bool(t) and t.replace(".", "").replace(",", "").isdigit()


def _cap_rows(cap):
    """caption → 行流。API 会把 tile 顶部的**前表尾行/节头行**降格为 <table> 前的
    文本(deb8de95: '103 7455.02...'三行三角尾+'## 二年交/男性'节头全在 caption 里,
    表内25物理行只进了19行)。逐行解析回行:数字密集行→按空格拆格(前表尾数据),
    文本行→单格行(节头);markdown 井号剥掉。拆格前先合并'914. 2'式误空格小数。"""
    out = []
    for ln in cap.splitlines():
        ln = ln.strip().lstrip("#").strip()
        if not ln:
            continue
        ln = _DEC_SP.sub(".", ln)
        toks = ln.split()
        num = sum(1 for t in toks if _numeric(t))
        out.append(toks if (len(toks) >= 2 and num >= len(toks) - 1) else [ln])
    return out


def _split_packed(rows, nc, inkws=None):
    """td 塌缩拆格(装配前规整):API 偶发把一行的多个 token 塞进**一个 <td>**(空格相连),
    正常 tile 路径与画网格重读都会出(B榜 a4e24107/7a8f3a74/34821e6c)。两条**结构判据**
    (无 magic 阈值,命中其一即拆,均容-1=API 漏读一格,与骨架对齐"容±1"同源):
    ① 恰好补齐 nc:拆后该行格数(含空格)恰等于期望列数(34821e6c 表头 13 token→14=nc-1);
    ② 恰好补齐墨迹宽:拆后该行**非空格数**恰等于该行骨架墨迹非空格数 inkw(墨证据当期望,
      与 _diagnose 同哲学;救 nc 够不着的场合——画网格重读行'[8值挤1格,垫空×3]' nc=13
      拆后 11 缺口太大,但墨迹恰 8 个非空格 → 拆。节头文本行墨迹仅 1 块、token≥2 → 不拆)。
    合并误空格小数('914. 2'→'914.2')后再数 token,不误拆;文本/数字 token 一视同仁。
    inkws[k] = 第 k 行的墨迹非空格数(与 rows 按序对齐,装配同款假设);None=不启用②。
    nc≤3 的窄带不拆(无塌缩空间)。

    诊断门控(用户判据):塌缩的表征是【列明显变少】——行的**非空格数已达期望 nc**
    (算上空格行宽与期望一致)= 没问题,整行原样不动;非空格数 < nc 才逐格怀疑拆分。"""
    if nc <= 3:
        return rows
    out = []
    for idx, row in enumerate(rows):
        if sum(1 for s in row if s.strip()) >= nc:
            out.append(row)
            continue
        iw = inkws[idx] if inkws is not None and idx < len(inkws) else None
        new = []
        for k, s in enumerate(row):
            t = _DEC_SP.sub(".", s.strip()) if s and s.strip() else ""
            toks = t.split()
            rest = row[k + 1:]
            fit_nc = (len(toks) >= 2
                      and nc - 1 <= len(new) + len(toks) + len(rest) <= nc)
            fit_ink = (iw is not None and len(toks) >= 2
                       and iw - 1 <= sum(1 for x in new if x.strip()) + len(toks)
                       + sum(1 for x in rest if x.strip()) <= iw)
            if fit_nc or fit_ink:
                new.extend(toks)
            else:
                new.append(s)
        out.append(new)
    return out


# ═════════════════════════ 诊断层(纯函数,零 API) ═════════════════════════

def _wstar(rws, dft):
    """有效宽=到【最后一个非空格】的位置(尾部去空),不是整行格数——API常把稀疏行
    补齐拖一堆尾部空<td>(e082 '9946.83,10244.56,0.00'+16空=19格),len会误判
    列多读→空跑自愈。真实内容宽3<<nc,不该触发;稠密劈裂全非空则宽=格数照触发。"""
    def w(x):
        last = max((k for k, s in enumerate(x) if s.strip()), default=-1)
        return last + 1
    cs = [w(x) for x in rws if any(s.strip() for s in x)]
    return Counter(cs).most_common(1)[0][0] if cs else dft


def _diagnose(rows, E=None, nc=None, inkw=None):
    """一致性诊断(架构核心,纯函数零 API):**按墨迹算出的期望非空行列** vs API 实读。
    一致 → 空集 = 没问题,直接装配;不一致 → 标签集,按标签分派修复层。

    · 行轴(传 E=有墨骨架行):实读行数或非空行数 < |E| → ROW_UNDER(漏读/caption降格/
      灰内容);> |E| → ROW_OVER(幻觉 or 骨架欠行,由仲裁定夺)。
    · 列轴(传 nc=期望列数):W* > nc+1 → COL_OVER(劈裂/口吃/重叠,tile 错→对半自愈)。
    · 列少读(传 inkw=墨迹有效宽):W* < inkw-1 → COL_UNDER(实读比墨迹还窄=真漏列),
      暂无修复手段,仅 audit 上报;W* 介于 [inkw-1, nc] 是稀疏表常态,**不是问题**
      (边缘空列 VLM 少读天经地义,装配按位补空)。
    · 塌缩(格级)在读数层解析时诊断(非空格数<nc 才拆,见 _split_packed),不在此重复。
    EMPTY/FLAT 是原始输出的格式废品,构不成"读数",在读数层就地抢救,也不进本诊断。"""
    labels = set()
    if nc is not None and rows and _wstar(rows, nc) > nc + 1:
        labels.add("COL_OVER")
    if inkw is not None and rows and _wstar(rows, inkw) < inkw - 1:
        labels.add("COL_UNDER")
    if E is not None:
        n = len(rows)
        nz = sum(1 for x in rows if any(s.strip() for s in x))
        if n < len(E) or nz < len(E):
            labels.add("ROW_UNDER")
        if n > len(E) or nz > len(E):
            labels.add("ROW_OVER")
    return labels


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


def _read_tiles(im, tiles, meta, cell_ink, timeout):
    """批量调用 + 解析 + **格式废品就地抢救**(不进诊断——EMPTY/FLAT 是输出格式问题,
    构不成"读数",谈不上与骨架一致):
    · EMPTY(有墨tile实读0行)= API重试耗尽后的静默空响应(空不入缓存,直接重调)。
    · FLAT(方案B'):tile有内容却返回【无<table>】=VLM放弃表结构竖排输出(稀疏tile左上角
      几个数被拍扁成一竖列,丢列位置)。只打这类废tile(全A榜~17个),正常tile一律不动、
      零副作用。判据看 raw 有内容(非rows)——单值flat如'71557.9'被parse_tile解析成空rows,
      但raw明明有值,若看rows会漏救(71557.9在v2/v3都丢)。raw非空+无table即抢救:
      单列/单行墨=几何完全确定→直拼零调用;真2D歧义→裁有墨包围盒+画网格逼它结构化;
      仍失败→几何兜底按包围盒放值保内容。
    · 塌缩行在此拆格(_split_packed 自带"非空格数<nc"诊断门控)。
    返回 parsed[(r,c)] = (caption, rows)。"""
    up = meta["upsample"]
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    rb, cb = meta["rb"], meta["cb"]
    flat = [(r, c) for r in range(len(tiles)) for c in range(len(tiles[r]))
            if tiles[r][c] is not None]
    outs = dict(zip(flat, call_tiles([tiles[r][c] for r, c in flat],
                                     timeout=timeout, upsample=up)))

    def _E_n(r, c):
        ri, rj = row_bands[r]
        ci, cj = col_bands[c]
        return int(sum(1 for i in range(ri, rj) if cell_ink[i, ci:cj].any()))

    # EMPTY 批量重调:有墨tile实读0行 = API重试耗尽后的静默空响应(空不入缓存,直接重调)
    empt = [(r, c) for (r, c) in flat
            if not _parse_cap(outs.get((r, c)))[1] and _E_n(r, c) > 0]
    if empt:
        for (r, c), o in zip(empt, api.call_many(
                [upscale(tiles[r][c], up) for r, c in empt], timeout=timeout)):
            if o:
                outs[(r, c)] = o

    # 截断批量修复(自愈对半同款,列界对半+无条件采纳):截断=输出撞12k上限的**确定性**
    # 事件,同图重打无用,拆小输出减半才有解(B榜实测51个tile,bc0ccbea 64414.74 这类
    # 尾值就是这么没的)。对半为真,行列不符交装配补空/裁多收束 + audit 记录
    trunc = [(rc, row_bands[rc[0]][0], row_bands[rc[0]][1],
              col_bands[rc[1]][0], col_bands[rc[1]][1]) for rc in flat
             if outs.get(rc) and api.is_truncated(outs[rc])]
    fixes = _reread_halved(im, rb, cb, trunc, up, timeout, meta) if trunc else {}

    parsed = {}
    need_grid = []                                  # flat抢救待重调: (r,c,r0,r1,c0,c1,ci,cj)
    for r in range(len(row_bands)):
        ri, rj = row_bands[r]
        for c in range(len(col_bands)):
            ci, cj = col_bands[c]
            raw = outs.get((r, c))
            cap, rows = _parse_cap(raw)
            if (r, c) in fixes:
                joined = fixes[(r, c)]
                E_n = _E_n(r, c)
                if joined:
                    nz2 = sum(1 for x in joined if any(s.strip() for s in x))
                    rows = joined                # 列对半为真,无条件采纳;cap 保留原读的
                    html = (cap + "\n" if cap else "") + rows_to_html(rows)
                    api.write_cache(upscale(tiles[r][c], up), html)
                    raw = html                   # 修复后是合法表输出,不再触发flat抢救
                    meta.setdefault("adopt" if nz2 == E_n else "audit", []).append(
                        f"tile[{r}][{c}] 截断→列对半为真(nz{nz2}/E{E_n},回填)"
                        + ("" if nz2 == E_n else " ⚠行数不符,装配收束"))
                else:
                    meta.setdefault("audit", []).append(
                        f"tile[{r}][{c}] 截断⚠ 对半空读,半截行进装配")
            if (tiles[r][c] is not None and raw and raw.strip()
                    and "<table" not in raw.lower()):
                ir = [i for i in range(ri, rj) if cell_ink[i, ci:cj].any()]
                ic = [j for j in range(ci, cj) if cell_ink[ri:rj, j].any()]
                if ir and ic:
                    if len(ic) == 1 or len(ir) == 1:
                        # 单列/单行:几何完全确定(墨只在一列→竖排,只在一行→横排),
                        # 直接拼接,不画网格不调API(省一次调用;行号列/表头行的主场)
                        vals = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                        body = [[v] for v in vals] if len(ic) == 1 else [vals]
                        off = ic[0] - ci
                        rows = [([""] * off + g + [""] * (cj - ci))[:cj - ci]
                                for g in body]
                        meta.setdefault("adopt", []).append(
                            f"tile[{r}][{c}] flat→几何直拼(单{'列' if len(ic) == 1 else '行'})")
                    else:                              # 多行×多列真2D歧义 → 画网格问API
                        need_grid.append((r, c, ir[0], ir[-1] + 1, ic[0], ic[-1] + 1, ci, cj, raw))
            parsed[(r, c)] = (cap, rows)
    if need_grid:                                   # flat重调批量并发(统一发送层 map_io;
        def gwork(a):                               # gwork 内部只用同步 call_safe,单层池无死锁)
            return a, _grid_reread(im, rb, cb, a[2], a[3], a[4], a[5], a[6], a[7], timeout)
        for a, grows in api.map_io(gwork, need_grid):
                r, c, r0, r1, c0, c1, ci, cj, raw = a
                if not grows:
                    # 网格失败(如1×1单值,API不把单数字当表)→ 兜底按包围盒几何放flat值,
                    # 至少保住内容(71557.9这类曾整个丢失)。竖排流按行/列布局:
                    vals = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                    nr, ncol = r1 - r0, c1 - c0
                    if not vals:
                        continue
                    if ncol == 1:                  # 单列: 每值一行(竖排列, 如序号列)
                        grows = [[v] for v in vals]
                    elif nr == 1:                  # 单行: 所有值一行(横排, 如列号表头)
                        grows = [vals]
                    else:                          # 多行多列且网格没救回: 按行填(保内容)。
                        # 数字密集行须按空格拆格(_cap_rows),否则整行并一格+右补空=塌缩
                        # (baf9a56f 单调区 '1000 ×14/行' 曾整行进单格,两seg~25%格子错)
                        grows = _cap_rows(raw)
                    grows = [[""] * (c0 - ci) + g + [""] * (cj - c1) for g in grows]
                    parsed[(r, c)] = (parsed[(r, c)][0], grows)
                    meta.setdefault("adopt", []).append(f"tile[{r}][{c}] flat→几何兜底")
                else:
                    parsed[(r, c)] = (parsed[(r, c)][0], grows)
                    meta.setdefault("adopt", []).append(f"tile[{r}][{c}] flat→画网格重读")
    # 读数层出口:塌缩拆格统一规整一次(覆盖正常解析/画网格重读/几何兜底所有行流来源;
    # 必须在列校准之前——投票看的行宽须是拆格后的)。caption 行流在装配中才诞生,
    # 其拆格在 _align_tile 回收处(全管线仅此两处)。inkws=各有墨骨架行的墨迹非空格数
    # (判据②的期望;rows 与有墨行按序对齐,与装配同一假设)。
    for c, (ci, cj) in enumerate(col_bands):
        for r, (ri, rj) in enumerate(row_bands):
            cap, rows = parsed[(r, c)]
            inkws = [int(cell_ink[i, ci:cj].sum())
                     for i in range(ri, rj) if cell_ink[i, ci:cj].any()]
            parsed[(r, c)] = (cap, _split_packed(rows, cj - ci, inkws))
    return parsed


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
                # 一致性铁律(用户原则:大部分tile行列必须一致):采纳必须≥2票——
                # 单票例外已删(deb8尾表被1票从15列改成31列,token在稀疏行膨胀,
                # 孤证=病态);幅度封顶+3(合法救援5fdf为+2,翻倍必是病)
                meta.setdefault("adopt", []).append(
                    f"列带{c} 列校准采纳 骨架{cj - ci}列→{top}列({n}票)")
                nc = top
        band_nc.append(nc)
    return band_nc


def _heal_col_over(im, tiles, parsed, meta, band_nc, timeout):
    """自愈(tile 级修复,只治 COL_OVER=列多读:劈裂/口吃/重叠 W*>nc+1)。
    列少读(W*<nc)对稀疏表正常——边缘空列VLM少读天经地义,切列救不了纯浪费,交装配
    补空;行数不符是行轴问题→投票(佐证加行)+E对齐。只留上界单向触发。
    顺序(用户设计):①空解释已由E对齐承担 ②一致性投票(列校准/佐证)已改完期望
    ③与【最终期望】仍不符的孤立tile=读错。**先投票后自愈**——5fdf带0(14→17,18票一致)
    此前被先送自愈白烧50次调用的教训。
    修法=列边界对半重读,读取即真值(用户终稿:3x不可靠,不做二次质检;半块+1行首空格
    由溢出弃空格消化;装配照常按E/期望收束)。"""
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    rb, cb = meta["rb"], meta["cb"]
    up = meta["upsample"]
    cands = []                                     # 先收集全部 COL_OVER tile…
    for r in range(len(row_bands)):
        ri, rj = row_bands[r]
        for c in range(len(col_bands)):
            ci, cj = col_bands[c]
            if tiles[r][c] is None:
                continue
            cap, rows = parsed[(r, c)]
            enc = band_nc[c]
            if "COL_OVER" in _diagnose(rows, nc=enc):
                cands.append((r, c, ri, rj, ci, cj, cap, rows, enc))
    if not cands:
        return
    fixes = _reread_halved(im, rb, cb,             # 共用列对半原语,一波齐发
                           [((r, c), ri, rj, ci, cj) for (r, c, ri, rj, ci, cj,
                                                          cap, rows, enc) in cands],
                           up, timeout, meta)
    for k, (r, c, ri, rj, ci, cj, cap, rows, enc) in enumerate(cands):
        joined = fixes.get((r, c)) or []
        if joined:
            parsed[(r, c)] = (cap, joined)     # 保留原caption!拆表双条件与行流回收
            w = _wstar(joined, enc)            # 都靠它(抹掉曾致deb8表数3→2)
            bad = "" if enc - 5 <= w <= enc + 1 else " ⚠未收敛"   # 唯一可能fail的信号:
            meta.setdefault("audit" if bad else "adopt", []).append(  # 自愈跑了仍不符
                f"tile[{r}][{c}] 自愈:对半为真 W*{w}/exp{enc}{bad}")
        else:
            meta.setdefault("audit", []).append(
                f"tile[{r}][{c}] 自愈:对半空读⚠ W*{_wstar(rows, enc)}/exp{enc}")


def _align_tile(cap, rows, E, band_idx, r, c, ci, cj, cell_ink, cell_gray, meta, band_row0):
    """单 tile 行轴修复(免 API)。诊断一致(实读行数与非空行数都=|E|)→ 原样返回不修;
    不一致才按固定顺序过修复链(顺序=等价性关键,不可重排):
    灰救回 → 空行条件丢弃 → caption 行流回收 → 斜线表头合并 → 拆行墨测试合并。
    返回修复后的 (E, rows)(灰救回会扩 E)。"""
    if not (_diagnose(rows, E=E) & {"ROW_UNDER", "ROW_OVER"}):
        return E, rows
    # 淡灰内容救回(API门控):d1752e16整张数字印浅灰,128判空→E欠数→真值被裁多丢。
    # 仅当【API实读非空行数 > E】(API确认有更多内容)时,从灰行(180判有墨、128判空)
    # 按位置补足差额。幻觉(落在128&180都空的行)无灰墨→补不进→仍裁,不误纳;框线
    # 灰边被cell_gray的frac>0.7+膨胀挡掉。灰行只在API确认时启用,平时零影响。
    nz_k = sum(1 for x in rows if any(s.strip() for s in x))
    if nz_k > len(E):
        gray = [i for i in band_idx if i not in E and cell_gray[i, ci:cj].any()]
        if gray:
            add = gray[:nz_k - len(E)]
            E = sorted(E + add)
            meta.setdefault("adopt", []).append(
                f"tile[{r}][{c}] 灰内容救回 +{len(add)}行(128判空/180有墨,API确认)")
    while len(rows) > len(E):              # 空行**条件丢弃**(仅实读超期望时):
        empt = [k for k, x in enumerate(rows)      # 白pad后残影幻觉空行已绝源,
                if not any(s.strip() for s in x)]  # 这里只兜画线后VLM老实输出的
        if not empt:                               # 空行(数量超出有墨行数的部分);
            break                                  # 真空行内容由骨架按位置补"",
        rows.pop(empt[0])                          # 不因丢弃而丢失
    if cap and len(rows) < len(E):         # 实读不足额 → caption行流回收补进
        crows = _cap_rows(cap)             # (跨列表头单行/前表尾行/节头行按行序回填)
        crows = _split_packed(crows, cj - ci,   # 文本表头挤一格('第40..第52保单年度')
                              [int(cell_ink[i, ci:cj].sum()) for i in E])
        rows = crows + rows                     # 按恰好补齐/墨迹宽判据拆(34821e6c)
        while len(rows) > len(E):          # 溢出(骨架在零缝junction并行,物理行
            for k in range(len(crows)):    # 多于骨架):优先裁cap里的**文本行**,
                if k < len(rows) and not any(                 # 但不丢弃——改道进
                        t.replace(".", "").replace(",", "").isdigit()  # 表间文本通道
                        for t in rows[k]):                    # (拆表时作节头输出);
                    meta.setdefault("split_txt", {}).setdefault(
                        band_row0, []).append(" ".join(x for x in rows[k] if x.strip()))
                    del rows[k]
                    break
            else:
                del rows[0]                # 全是数据行才裁最前(不得已)
    if r == 0 and len(rows) >= len(E) + 1 and len(rows) >= 2:
        a, b = rows[0], rows[1]            # 斜线表头:一个高格斜线分写两行,OCR
        ov = [t for t in range(min(len(a), len(b)))   # 拆成两行且仅第0格重叠
              if a[t].strip() and b[t].strip()]       # → 合并,格0='下\上'
        if ov == [0]:                                 # (GT 口径:保单年度末\投保年龄)
            m = [b[0] + "\\" + a[0]] + [x if x.strip() else y
                 for x, y in zip(a[1:] + [""] * (len(b) - len(a)),
                                 b[1:] + [""] * (len(a) - len(b)))]
            rows = [m] + rows[2:]
    if len(rows) == len(E) + 1:
        # 拆行合并(行线/墨测试,非打分):拆出的两行本是同一条骨架行——相邻互补对
        # 合并后的非空格数须**恰等**该骨架行墨格数;真表头两行合并后对不上。
        # 通过者唯一才合并,否则保守裁尾。行是真值,读数必须与骨架一致。
        hits = []
        for k in range(len(rows) - 1):
            a, b = rows[k], rows[k + 1]
            L = max(len(a), len(b))
            aa = a + [""] * (L - len(a))
            bb = b + [""] * (L - len(b))
            if not (any(x.strip() for x in aa) and any(x.strip() for x in bb)):
                continue
            if any(x.strip() and y.strip() for x, y in zip(aa, bb)):
                continue
            merged = [x if x.strip() else y for x, y in zip(aa, bb)]
            nz = sum(1 for x in merged if x.strip())
            if k < len(E) and nz == int(cell_ink[E[k], ci:cj].sum()):
                hits.append((k, merged))
        if len(hits) == 1:
            k, merged = hits[0]
            rows = rows[:k] + [merged] + rows[k + 2:]
    return E, rows


# ═════════════════════════ 装配层(纯摆放,零 API 零修复) ═════════════════════════

def _assemble(parsed, meta, band_nc, cell_ink, cell_gray):
    """只信骨架:行多裁少补、溢出弃空格、按位补空。修复层没治好的在此强制收束,
    并 audit 真损失上报(找问题,不藏问题):
      补空有墨位: 骨架该tile有E个有墨行,实读不足→末尾几个有墨行拿不到内容,补空丢行
      裁多有内容: 实读>E,多出的行有内容→被裁(佐证加行至多救1行),丢内容
    返回 (grid, cap_rows_global)。"""
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    grid = []
    cap_rows_global = []                       # (拆表条件1)非首带出现caption的带 → 其全局行范围
    for r, (ri, rj) in enumerate(row_bands):
        band_idx = list(range(ri, rj))
        aligned = {}
        for c, (ci, cj) in enumerate(col_bands):
            cap, rows = parsed[(r, c)]
            E = [i for i in band_idx if cell_ink[i, ci:cj].any()]
            if rows and E:                     # 列少读诊断(audit-only,暂无修复手段):
                ws = [int(np.where(cell_ink[i, ci:cj])[0].max()) + 1 for i in E]
                inkw = Counter(ws).most_common(1)[0][0]
                if "COL_UNDER" in _diagnose(rows, inkw=inkw):
                    meta.setdefault("audit", []).append(
                        f"tile[{r}][{c}] 列少读⚠ W*{_wstar(rows, inkw)}<墨迹宽{inkw}")
            E, rows = _align_tile(cap, rows, E, band_idx, r, c, ci, cj,
                                  cell_ink, cell_gray, meta, len(grid))
            aligned[c] = (E, rows)
            nz_rows = [x for x in rows if any(s.strip() for s in x)]
            pos = "首带" if r == 0 else ("末带" if r == len(row_bands) - 1 else "中带!")
            if len(nz_rows) > len(E) + 1:              # 裁多(+1留给佐证加行):在aligned阶段判,
                meta.setdefault("audit", []).append(   # 因多余行装配时就丢了,最终grid看不到
                    f"tile[{r}][{c}]({pos}) 裁多丢行 实读{len(nz_rows)}>期望{len(E)},多余被裁")
            if cap and r > 0:
                cap_rows_global.append((len(grid), len(grid) + len(band_idx)))
        extra_votes = sum(1 for c in range(len(col_bands))
                          if len(aligned[c][1]) == len(aligned[c][0]) + 1)
        #                 ^ 恰好期望+1 才有投票权(越界废品已在解析层清除,双保险)
        pos = "首带" if r == 0 else ("末带" if r == len(row_bands) - 1 else "中带!")
        # 稀疏首行门控(治整片行错位):band0 首骨架行只在部分 col tile 有墨(=跨列表头/
        # 角格行,如6dfcd28f的[1,0,1,0]),VLM 常把该行合并进列号行→有墨 tile 实读少一行
        # 且缺口在【顶部】。仅此情形对缺行 tile 底部对齐(缺口补顶),右侧列号沉回正确行。
        # 首行全有墨/单tile/无缺行 → 原顶部对齐,不动真缺尾行的表(945e8fe9/88c6dbb4
        # 无门控全量底部对齐时 -9.6/-4.3 的教训)。
        sparse_head = False
        if r == 0 and len(col_bands) > 1 and band_idx:
            ink0 = [bool(cell_ink[band_idx[0], ci:cj].any()) for ci, cj in col_bands]
            sparse_head = any(ink0) and not all(ink0)
        for i in band_idx:                         # 行以骨架为准:多裁少补;唯一例外见下
            rowcells = []
            for c, (ci, cj) in enumerate(col_bands):
                E, rows = aligned[c]
                if sparse_head and i in E and len(rows) < len(E):
                    idx = E.index(i) - (len(E) - len(rows))
                    cells = rows[idx] if 0 <= idx < len(rows) else []
                else:
                    cells = rows[E.index(i)] if i in E and E.index(i) < len(rows) else []
                nc = band_nc[c]
                if len(cells) > nc:                # 溢出弃空格:先丢空格格(零信息)再截尾
                    need = len(cells) - nc         # (bac0aeac 13格=11值+''+0.00,截尾杀
                    kept = []                      #  真值0.00;丢''零损失)。截掉的非空=tile
                    for s in cells:                #  列边界重叠/VLM末值口吃的冗余副本,
                        if need and not s.strip():  #  真值由拥属tile保留(3792d522 136格截断
                            need -= 1; continue     #  实测真丢0)——补空/截断都是正常操作,不报
                        kept.append(s)
                    cells = kept
                rowcells += list(cells[:nc]) + [""] * max(0, nc - len(cells))
            grid.append(rowcells)
            # 补空丢行=看【最终grid行】:用组装自己的判空(cell_ink,与E同一把尺,不另造阈值)
            # ——骨架判此行有墨(cell_ink任一列真),装配出的这行却全空 = 有墨位补成空,真丢。
            if cell_ink[i].any() and not any(s.strip() for s in rowcells):
                meta.setdefault("audit", []).append(
                    f"骨架行{i}({pos}) 补空丢行 cell_ink判有墨却装配成全空")
        if extra_votes >= 2 or (extra_votes == 1 and len(col_bands) == 1):
            meta.setdefault("adopt", []).append(
                f"带{r} 佐证加行 +1行({extra_votes}票)")
            # **佐证加行**(骨架级修复,一致性仲裁):拆行墨测试没吃掉的多余实读行,≥2 tile
            # 同票(或单tile带)=骨架真欠一行(微距表 <3px 缝并行,a1aaef73 列号行+年1行挤在
            # 一个骨架行,API 实读 6>骨架5,0.00 末行被裁)→带尾补一行,实读顺序本身即正确
            # 顺序。孤证=幻觉仍裁
            rowcells = []
            for c, (ci, cj) in enumerate(col_bands):
                E, rows = aligned[c]
                cells = rows[len(E)] if len(rows) > len(E) else []
                nc = band_nc[c]
                rowcells += list(cells[:nc]) + [""] * max(0, nc - len(cells))
            grid.append(rowcells)
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
    """单个 seg 的骨架 OCR — 四段式:读数 → 诊断 → 修复 → 装配。
    返回 (grid, ncalls, meta);grid=None 表示骨架不可信需回退(交上层 ocr_table 自由读)。

    · 骨架 = 行列估计(slice_grid);misaligned → 早退。
    · 读数层(_read_tiles):API 调用+解析+格式废品就地抢救(EMPTY 重调 / FLAT 直拼、
      画网格重读、几何兜底 / 塌缩拆格)。
    · 诊断层(_diagnose,纯函数):按墨迹算期望非空行列,与实读一致=没问题直接装配;
      不一致才贴标签进修复。
    · 修复层:孤立异常=tile 错→修 tile(COL_OVER 对半自愈 / 行轴 _align_tile 免API重排);
      多 tile 一致异常=骨架错→改期望(_calibrate_cols 列校准、_assemble 内佐证加行,
      一致性铁律≥2票)。先仲裁后自愈(5fdf 先自愈白烧50次调用的教训)。修不好→audit。
    · 装配层(_assemble):只信骨架,多裁少补;组装 = **骨架行级墨证据 × tile 读数逐行
      核销**(替代 tile 级补零/众数)。表边界 _find_splits 轴行拆表。"""
    tiles, meta = slice_grid(im)
    if meta["misaligned"]:
        return None, 0, meta
    rb, cb = meta["rb"], meta["cb"]
    cell_ink, cell_gray = _ink_evidence(im, rb, cb, meta["rows"], meta["cols"])
    parsed = _read_tiles(im, tiles, meta, cell_ink, timeout)
    band_nc = _calibrate_cols(parsed, meta)                    # 骨架仲裁在前…
    _heal_col_over(im, tiles, parsed, meta, band_nc, timeout)  # …tile 自愈在后
    grid, cap_rows_global = _assemble(parsed, meta, band_nc, cell_ink, cell_gray)
    meta["splits"] = _find_splits(grid, cap_rows_global)
    ncalls = sum(1 for row in tiles for t in row if t is not None)
    return grid, ncalls, meta
