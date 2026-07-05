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
import numpy as np
from collections import Counter
from PIL import Image

import api_client as api
from config import MAX_CONCURRENCY, API_USER_IDS, BIN_INK, BIN_FAINT
from geom import row_bnds, col_bnds, rows_misaligned, LINE_FULL
from slicer_table import MAX_TILE_ROWS, MAX_TILE_COLS, UP_EDGE, UP_TARGET, TILE_MAX
from stitch_table import parse_tile

_BLANK_TILE_INK = 0.001   # tile 墨率<此=空白,不调 API,按骨架补空 cell
_EDGE_PAD = 3             # tile 四周留白px(边界在缝中心,±3 不吃邻格墨)
_ASPECT_SAFE = 180   # API拒收>200:1(实测),安全边180(几何列带对半/_ocr_strip共用)
_UP_CAP = 3.0             # 上采样上限(密集表实验:15列+自适应放大,cap3 兜住最小格)
_UP_TARGET = 55           # 目标格边长(实证:edge27 在 2× 已 99.5~99.8%,55/27≈2.0 正中甜点;
#                           slicer 的 75 在 cap2 时代从未生效,cap3 下会把 2.78× 强加给
#                           27px 格,载荷×7.7 纯浪费。3× 只留给最小格(19px→2.9×)


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
    rb, rf = row_bnds(dark, dark180)
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
        return None, meta
    # 密集判据(沿用 slicer):每 cell 平均边长小 → 上采样
    edge = (g.shape[0] * g.shape[1] / max(1, R * C)) ** 0.5
    meta["upsample"] = round(min(_UP_CAP, _UP_TARGET / edge), 2) if edge < UP_EDGE else 1.0
    meta["cell_edge"] = round(edge, 1)
    # 硬上限 行25×列15(上榜版参数,让渡已废——双田实测密集表35列≈69%/15列≥99.5%,
    # 让渡为省调用把列放大到37~187是密集内容错误的第一元凶;矮宽表也不例外)
    row_bands = _chunk(rb, MAX_TILE_ROWS)
    col_bands = _chunk(cb, MAX_TILE_COLS)
    if edge >= UP_EDGE and g.size <= TILE_MAX * TILE_MAX:
        # 小段快路:整段像素 ≤ 单tile预算 且 格子大(非密集)——切分的两个理由(图太大/
        # 格太小)都不存在,整段一口读。标题条(4400×96,字缝被数成上百"列")不再被
        # 15列上限切成8个tile;密集段(1de69d49尾表edge≈31)不走快路,读并防护不破。
        # 超比例仍由下方180:1对半环节兜
        row_bands, col_bands = [(0, R)], [(0, C)]
    # 像素长宽比约束:API 拒收 >200:1(400)。矮表单tile可到 241:1(1de69d49 尾表 2行
    # 7000×29px)——列带宽 > 180×最矮行带高 时对半加密列带(切在列边界上,骨架拼接原生
    # 支持多tile;不垫白,内容无损)
    min_bh = min(rb[j] - rb[i] for i, j in row_bands)
    while col_bands:
        wmax = max(cb[j] - cb[i] for i, j in col_bands)
        if wmax <= _ASPECT_SAFE * max(1, min_bh) or all(j - i <= 1 for i, j in col_bands):
            break
        col_bands = _chunk(cb, max(1, -(-max(j - i for i, j in col_bands) // 2)))
    meta["row_bands"], meta["col_bands"] = row_bands, col_bands
    H, W = g.shape
    tiles = []
    for (ri, rj) in row_bands:
        row = []
        for (ci, cj) in col_bands:
            y0, y1 = rb[ri], rb[rj]
            x0, x1 = cb[ci], cb[cj]
            # 空白判据(用户设计:真空⟺①每行墨投影基本一样②每行都很低,合并实现为
            # "去线后每行墨<3px"):先扣竖线列(线墨=每行常数基线)与横线行(frac>0.5),
            # 剩余行墨 max<3px ⟺ 真空。旧均值判据尺度相关,孤零"0.00"被110万px稀释
            # 误杀(实测17tile/54格);有框表线墨也不再撑爆判据 → 框内空区照跳,省调用
            band = dark[y0:y1, x0:x1]
            band180 = dark180[y0:y1, x0:x1]
            vkeep = band180.mean(axis=0) <= LINE_FULL   # 线=贯通;密字行0.5~0.7有字缝,
            b2 = band[:, vkeep] if vkeep.any() else band   # 0.5阈会把整行密字当线剔掉
            rfrac = b2.mean(axis=1) if b2.size else np.zeros(1)   # (0e8a501f 30格被误跳)
            hkeep = rfrac <= LINE_FULL
            rowink = b2[hkeep].sum(axis=1) if hkeep.any() else np.zeros(1)
            if rowink.size == 0 or rowink.max() < 3:
                row.append(None)
                continue
            # **白pad**:精确按边界裁剪,贴到四周留白的画布上——实pad(±3px)会带进邻带行
            # 的底边残影,VLM 对残影输出幻觉空行,整tile内容错移一行(9c7857f3 0.78 病根)。
            # 边界都在缝中心/框线上,字不贴边,白边只提供视觉余量,邻带墨物理进不来。
            core = im.crop((x0, y0, x1, y1)).convert("RGB")
            t = Image.new("RGB", (core.width + 2 * _EDGE_PAD, core.height + 2 * _EDGE_PAD),
                          (255, 255, 255))
            t.paste(core, (_EDGE_PAD, _EDGE_PAD))
            row.append(t)
        tiles.append(row)
    return tiles, meta


def _up(t, factor):
    if t is not None and factor and factor > 1:
        return t.resize((round(t.width * factor), round(t.height * factor)), Image.LANCZOS)
    return t


def _tokens(cell):
    """格内空格分词+数字重组:'3, 464'类被空格切碎的数字回并
    (前token以,/./，结尾且后token以数字开头→并回)。"""
    out = []
    for p in cell.split():
        if out and out[-1][-1] in ",.，" and p[0].isdigit():
            out[-1] += p
        else:
            out.append(p)
    return out


def _unmerge_row(cells, nc):
    """行级并列修复(半线表波动的确定性解药):VLM'以线分列'时把无线数据区并成一格
    ('59.36 138.31 ...'),实读格数<骨架列数但空格俱在——按token展开,恰好对上骨架
    列数才采纳(1674392a 倔形态3格→11格;乖形态行宽=nc不触发,零影响)。"""
    if len(cells) >= nc:
        return cells
    ex = []
    for c in cells:
        ex += _tokens(c) if c.strip() else [""]
    return ex if len(ex) == nc else cells


def _parse_cap(raw):
    """解析 tile 输出 → (caption, rows)。tile 是从表内切出的,**不存在表外文字**——
    API 把跨列表头(colspan行,如"保单年度")当 caption 放在 <table> 前,必须回收。"""
    rows = parse_tile(raw) if raw else []
    cap = ""
    if raw and "<table" in raw.lower():
        head = raw[:raw.lower().index("<table")]
        cap = head.strip()
    return cap, rows


def _cap_rows(cap):
    """caption → 行流。API 会把 tile 顶部的**前表尾行/节头行**降格为 <table> 前的
    文本(deb8de95: '103 7455.02...'三行三角尾+'## 二年交/男性'节头全在 caption 里,
    表内25物理行只进了19行)。逐行解析回行:数字密集行→按空格拆格(前表尾数据),
    文本行→单格行(节头);markdown 井号剥掉。"""
    out = []
    for ln in cap.splitlines():
        ln = ln.strip().lstrip("#").strip()
        if not ln:
            continue
        toks = ln.split()
        num = sum(1 for t in toks if t.replace(".", "").replace(",", "").isdigit())
        out.append(toks if (len(toks) >= 2 and num >= len(toks) - 1) else [ln])
    return out


def ocr_seg(im, timeout=240):
    """单个 seg 的骨架 OCR。返回 (grid, ncalls, meta);grid=None 表示骨架不可信需回退。

    组装 = **骨架行级墨证据 × tile 读数逐行核销**(替代 tile 级补零/众数):
    · tile 的期望行 = 该 tile 列范围内**有文字墨**的骨架行(排除纯横线行——OCR 不输出
      空行/线行,右侧 tile 因 colspan 区顶部空白整体上移一行的错位由此消除,1674392a)
    · caption 回填:实读=期望-1 且有 caption → caption 是首个有墨行的内容(跨列表头)
    · 差额核销:实读<期望 → 信骨架补空(OCR漏行是常态);实读>期望 → 需同带≥2 tile
      佐证(或单tile带)才在带尾加一行,孤证=幻觉裁掉。"""
    tiles, meta = slice_grid(im)
    if meta["misaligned"]:
        return None, 0, meta
    up = meta["upsample"]
    row_bands, col_bands = meta["row_bands"], meta["col_bands"]
    rb, cb = meta["rb"], meta["cb"]
    dark = np.asarray(im.convert("L")) < BIN_INK
    flat = [(r, c) for r in range(len(tiles)) for c in range(len(tiles[r]))
            if tiles[r][c] is not None]
    from concurrent.futures import ThreadPoolExecutor
    def call(args):
        i, (r, c) = args
        return api.call_safe(_up(tiles[r][c], up), timeout=timeout,
                             user_id=API_USER_IDS[i % len(API_USER_IDS)])
    outs = {}
    if flat:
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENCY, len(flat))) as ex:
            for (r, c), o in zip(flat, ex.map(call, list(enumerate(flat)))):
                outs[(r, c)] = o

    # 格级墨证据:cell_ink[i][j] = 骨架格(i,j)内部(收缩2px避开框线)文字墨≥3px。
    # 行对齐(哪些行有内容)和列摆放(哪些格有内容)共用——空白判定要求极低墨。
    R, C = meta["rows"], meta["cols"]
    cell_ink = np.zeros((R, C), dtype=bool)
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

    # 先解析全部 tile;**行数越界=废品**(行期望偏差≤1 的先验即废品检测器:实读 >
    # 期望+1 物理上不可能,必是错误页/幻觉)→绕过缓存重读一次(垃圾可能已污染缓存),
    # 重读好则回写缓存;仍废 → 置空(补空,不投票不入表)
    parsed = {}
    for r in range(len(row_bands)):
        ri, rj = row_bands[r]
        for c in range(len(col_bands)):
            ci, cj = col_bands[c]
            cap, rows = _parse_cap(outs.get((r, c)))
            E_n = int(sum(1 for i in range(ri, rj) if cell_ink[i, ci:cj].any()))
            if not rows and tiles[r][c] is not None and E_n > 0:
                # 有墨tile实读0行 = API重试耗尽后的静默空响应(空不入缓存,直接重调)。
                # 废品检测只抓行数超界,实读0是另一半洞——老slicer的空tile重试丢失于
                # 重写,全量夜跑抽风受害十余张的病根
                raw2 = api.call_safe(_up(tiles[r][c], up), timeout=timeout)
                cap, rows = _parse_cap(raw2)
            parsed[(r, c)] = (cap, rows)
            if tiles[r][c] is not None and len(rows) != E_n:
                pos = "首带" if r == 0 else ("末带" if r == len(row_bands) - 1 else "中带!")
                badc = sum(1 for x in rows if len(x) != cj - ci)
                meta.setdefault("audit", []).append(
                    f"tile[{r}][{c}]({pos}) 行:期望{E_n}实读{len(rows)} 列宽不符行数:{badc}/{len(rows)}")
                # 行整体精确的先验:只有首带(表头拆行/斜线)和末带(稀疏)允许偏差,
                # 中带偏差=异常(骨架错/API抽风),审计日志供人工核查

    # **列校准**(行列职责不对称:行估计=真值,列在稀疏区/标签区可能少):非空 tile 的行
    # 格数众数若一致 = 骨架列数+k(k>0),且 ≥2 个行带的 tile 同票(或该列带只有 1 个非空
    # tile) → 采纳 nc+k(5fdf46b0 三标签列被并 1 列,429 行每行一致多读 2 格=最强信号;
    # 稀疏空 tile 不投票)。
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

    # ── 统一自愈(用户设计,顺序:①空解释已由E对齐承担 ②一致性投票(上面的列校准/
    # 下面的佐证)已改完期望 ③与【最终期望】仍不符的孤立tile=读错 → 升采样3×重读
    # → 对半重读 → 放弃。行列同构:列看行宽众数 vs band_nc;行看非空行数 vs E(欠读
    # <E-1 也走此梯,封 1829 欠读洞)。先投票后自愈——5fdf带0(14→17,18票一致)此前被
    # 先送自愈白烧50次调用的教训。
    def _wstar(rws, dft):
        cs = [len(x) for x in rws if any(s.strip() for s in x)]
        return Counter(cs).most_common(1)[0][0] if cs else dft
    for r in range(len(row_bands)):
        ri, rj = row_bands[r]
        for c in range(len(col_bands)):
            ci, cj = col_bands[c]
            if tiles[r][c] is None:
                continue
            cap, rows = parsed[(r, c)]
            enc = band_nc[c]
            E_n = int(sum(1 for i in range(ri, rj) if cell_ink[i, ci:cj].any()))
            nz = sum(1 for x in rows if any(s.strip() for s in x))
            col_bad = rows and not (enc - 5 <= _wstar(rows, enc) <= enc + 1)
            row_bad = rows and (nz < E_n - 1 or nz > E_n + 1)   # 行越界±1(欠读/超读统一)
            if not (col_bad or row_bad):
                continue
            # 自愈=列边界对半重读,读取即真值(用户终稿:3x不可靠,不做二次质检;
            # 半块+1行首空格由溢出弃空格消化;装配照常按E/期望收束)
            mid = (ci + cj) // 2
            halves = []
            for lo, hi in ((ci, mid), (mid, cj)):
                if hi <= lo:
                    halves.append([]); continue
                core = im.crop((cb[lo], rb[ri], cb[hi], rb[rj])).convert("RGB")
                tt = Image.new("RGB", (core.width + 2 * _EDGE_PAD,
                                       core.height + 2 * _EDGE_PAD), (255, 255, 255))
                tt.paste(core, (_EDGE_PAD, _EDGE_PAD))
                _, hrows = _parse_cap(api.call_safe(_up(tt, up), timeout=timeout))
                halves.append(hrows)
            hl, hr_ = halves
            n_ = max(len(hl), len(hr_)) if (hl or hr_) else 0
            joined = [(hl[i] if i < len(hl) else [""] * (mid - ci))
                      + (hr_[i] if i < len(hr_) else [""] * (cj - mid))
                      for i in range(n_)]
            if joined:
                parsed[(r, c)] = (cap, joined)     # 保留原caption!拆表双条件与行流回收
                meta.setdefault("adopt", []).append(    # 都靠它(抹掉曾致deb8表数3→2)
                    f"tile[{r}][{c}] 自愈:对半为真 W*{_wstar(joined, enc)}/exp{enc}")
            else:
                meta.setdefault("audit", []).append(
                    f"tile[{r}][{c}] 对半空读,以骨架为真 W*{_wstar(rows, enc)}/exp{enc} 行{nz}/E{E_n}")

    grid = []
    cap_rows_global = []                       # (条件1)非首带出现caption的带 → 其全局行范围
    for r, (ri, rj) in enumerate(row_bands):
        band_idx = list(range(ri, rj))
        aligned = {}
        for c, (ci, cj) in enumerate(col_bands):
            cap, rows = parsed[(r, c)]
            E = [i for i in band_idx if cell_ink[i, ci:cj].any()]
            while len(rows) > len(E):              # 空行**条件丢弃**(仅实读超期望时):
                empt = [k for k, x in enumerate(rows)      # 白pad后残影幻觉空行已绝源,
                        if not any(s.strip() for s in x)]  # 这里只兜画线后VLM老实输出的
                if not empt:                               # 空行(数量超出有墨行数的部分);
                    break                                  # 真空行内容由骨架按位置补"",
                rows.pop(empt[0])                          # 不因丢弃而丢失
            if cap and len(rows) < len(E):         # 实读不足额 → caption行流回收补进
                crows = _cap_rows(cap)             # (跨列表头单行/前表尾行/节头行按行序回填)
                rows = crows + rows
                while len(rows) > len(E):          # 溢出(骨架在零缝junction并行,物理行
                    for k in range(len(crows)):    # 多于骨架):优先裁cap里的**文本行**,
                        if k < len(rows) and not any(                 # 但不丢弃——改道进
                                t.replace(".", "").replace(",", "").isdigit()  # 表间文本通道
                                for t in rows[k]):                    # (拆表时作节头输出);
                            meta.setdefault("split_txt", {}).setdefault(
                                len(grid), []).append(" ".join(x for x in rows[k] if x.strip()))
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
            aligned[c] = (E, rows)
            if cap and r > 0:
                cap_rows_global.append((len(grid), len(grid) + len(band_idx)))
        extra_votes = sum(1 for c in range(len(col_bands))
                          if len(aligned[c][1]) == len(aligned[c][0]) + 1)
        #                 ^ 恰好期望+1 才有投票权(越界废品已在解析层清除,双保险)
        for i in band_idx:                         # 行以骨架为准:多裁少补;唯一例外见下
            rowcells = []
            for c, (ci, cj) in enumerate(col_bands):
                E, rows = aligned[c]
                cells = rows[E.index(i)] if i in E and E.index(i) < len(rows) else []
                nc = band_nc[c]
                if len(cells) > nc:                # 溢出弃空格:先丢空格格(零信息)再截尾
                    need = len(cells) - nc         # (bac0aeac 13格=11值+''+0.00,截尾杀
                    kept = []                      #  真值0.00;丢''零损失)
                    for s in cells:
                        if need and not s.strip():
                            need -= 1; continue
                        kept.append(s)
                    cells = kept
                rowcells += list(cells[:nc]) + [""] * max(0, nc - len(cells))
            grid.append(rowcells)
        if extra_votes >= 2 or (extra_votes == 1 and len(col_bands) == 1):
            meta.setdefault("adopt", []).append(
                f"带{r} 佐证加行 +1行({extra_votes}票)")
            # **佐证加行**:拆行墨测试没吃掉的多余实读行,≥2 tile 同票(或单tile带)=骨架
            # 真欠一行(微距表 <3px 缝并行,a1aaef73 列号行+年1行挤在一个骨架行,API 实读
            # 6>骨架5,0.00 末行被裁)→带尾补一行,实读顺序本身即正确顺序。孤证=幻觉仍裁
            rowcells = []
            for c, (ci, cj) in enumerate(col_bands):
                E, rows = aligned[c]
                cells = rows[len(E)] if len(rows) > len(E) else []
                nc = band_nc[c]
                rowcells += list(cells[:nc]) + [""] * max(0, nc - len(cells))
            grid.append(rowcells)
    # 表边界(用户双条件): ①非首带tile出现caption ②该带行流含轴行(连续整数序列>10)
    # → 在轴行处拆表(deb8de95 '二年交'新块自带[0..14]轴行;1674392a 轴行在首带且
    #   caption带无轴行,双条件互锁不误拆)
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
    meta["splits"] = sorted(set(splits))
    return grid, len(flat), meta
