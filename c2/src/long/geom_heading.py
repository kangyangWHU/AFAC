# -*- coding: utf-8 -*-
"""几何标题层级修正:在 API/pipeline 的 Markdown 上重定级标题 H1–H6。

主逻辑靠**编号格式关系**(不靠字号):
  - 栈跟踪祖先深度:新格式=父级+1、同格式=兄弟同级、dec(x.y)按点数绝对深度
  - 缺父级 orphan 才用中文编号约定 rank 兜底(章<条<一、<（一）<1.…)
图像证据:
  - 淡横线上方=H1 章(救回 API 漏标的章);dec 不被横线覆盖(格式优先)
  - 无编号中段标题按字号判:≥正文中位 1.25× = 真篇名保 H1,否则 API 误报删
其它:目录区两层分级、伪定义项(冒号引出+正文兄弟)降级、连续枚举兄弟一致性。

字号 = 行内 core 高度(墨量≥中位×0.6 的行数),相对本文正文中位、每文档独立。
需 rapidocr(格级 rec ~9ms/行,结果按图像 hash 存盘)。表格/目录区按规则排除。

与 heading_norm 的关系:pipeline 里 heading_norm 先给条带内相对层级,本模块最后
在整篇 + 图像证据上重定级,**以此为准**。
"""
import re
import os
import pickle
import hashlib
import unicodedata
import collections
import numpy as np
from rapidfuzz import fuzz
from long.slicer_long import table_bands
from long.heading_norm import (_CHAP1, parse_marker as _pm, _is_pseudo_heading,
                               _LEADIN_END, _SENT_END)

_H = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def _segment(im):
    a = np.asarray(im.convert("L")); dark = a < 160; ink = dark.sum(1)
    out, s = [], None
    for y in range(len(ink)):
        if ink[y] > 2 and s is None:
            s = y
        elif ink[y] <= 2 and s is not None:
            if y - s >= 8:
                seg = ink[s:y]
                med = np.median(seg[seg > 0]) if (seg > 0).any() else 0
                core = int((seg >= med * 0.6).sum()) if med else (y - s)
                cols = np.where(dark[s:y].any(0))[0]
                out.append((s, y, int(cols.min()), int(cols.max()), core))
            s = None
    return out


def light_rules(im, bands):
    """全宽浅灰横线(章标题下方的分隔线)的 y 区间。

    判据:整行 <245 覆盖 >55% 宽,且几乎无 <160 的文字墨(df<0.05),薄(≤8px)。
    表格区排除;目录区的分栏横线是**半宽**,被全宽条件天然排除(目录是双栏排版)。
    用户裁决:淡横线上方的那一行 = H1(章)。三份样张验证 6/6、11/11、7/7 全中。
    """
    a = np.asarray(im.convert("L"))
    light = (a < 245).mean(1)
    dark = (a < 160).mean(1)
    rows = np.where((light > 0.55) & (dark < 0.05))[0]
    out, s0, prev = [], None, None
    for y in rows:
        if s0 is None:
            s0 = prev = y
        elif y - prev <= 2:
            prev = y
        else:
            if prev - s0 <= 8:
                out.append((int(s0), int(prev)))
            s0 = prev = y
    if s0 is not None and prev - s0 <= 8:
        out.append((int(s0), int(prev)))
    return [(r0, r1) for r0, r1 in out
            if not any(b0 - 10 < r0 < b1 + 10 for b0, b1 in bands)]


def _align(heads_norm, geo):
    """NW monotonic fuzzy alignment: heads idx -> geo idx."""
    na, nb = len(heads_norm), len(geo)
    pair = {}
    if not na or not nb:
        return pair
    dp = np.zeros((na + 1, nb + 1)); bt = np.zeros((na + 1, nb + 1), dtype=np.int8)
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            sc = fuzz.partial_ratio(heads_norm[i - 1], geo[j - 1][0]) / 100.0
            sc = sc if sc >= 0.6 else -1.0
            d, g, h = dp[i-1][j-1] + sc, dp[i][j-1], dp[i-1][j] - 0.01
            b = max(d, g, h); dp[i][j] = b
            bt[i][j] = 0 if b == d else (1 if b == g else 2)
    i, j = na, nb
    while i > 0 and j > 0:
        if bt[i][j] == 0:
            if fuzz.partial_ratio(heads_norm[i-1], geo[j-1][0]) / 100.0 >= 0.6:
                pair[i-1] = j-1
            i, j = i-1, j-1
        elif bt[i][j] == 1:
            j -= 1
        else:
            i -= 1
    return pair


_GEO_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "cache_geo")


def _image_features(im, ocr, cache_key=None):
    """图像相关特征(逐行 OCR + core + 淡横线),只依赖图像 → 存盘缓存。

    99% 的耗时在"全文每行 OCR"(大文档 741 行 × 230ms ≈ 3min),而图像不变则结果不变。
    改定级逻辑重跑时命中缓存 → 12min 降到几秒。key = 图像内容哈希。
    返回 (L, line_ocr, rule_yranges) —— L 每行 (y0,y1,x0,x1,core),line_ocr 同序 OCR。
    """
    cpath = None
    if cache_key:
        os.makedirs(_GEO_CACHE, exist_ok=True)
        h = hashlib.sha1((cache_key + str(im.size)).encode()).hexdigest()
        cpath = os.path.join(_GEO_CACHE, h + ".pkl")
        if os.path.exists(cpath):
            with open(cpath, "rb") as f:
                return pickle.load(f)
    bands = table_bands(im)
    L = [l for l in _segment(im) if not any(a < l[1] and b > l[0] for a, b in bands)]
    line_ocr = [_norm(ocr(im.crop((x0, max(0, y0 - 3), x1 + 1, y1 + 3))))
                for y0, y1, x0, x1, core in L]
    rule_yranges = light_rules(im, bands)
    feat = (L, line_ocr, rule_yranges)
    if cpath:
        with open(cpath, "wb") as f:
            pickle.dump(feat, f)
    return feat


def correct(md, im, ocr, cache_key=None):
    """返回 (new_md, changes)。ocr(pil)->str 是格级识别函数。
    cache_key(图片名)给定时,逐行 OCR 结果存盘复用 —— 迭代定级逻辑不重跑 OCR。
    """
    lines = md.split("\n")

    # 封面标题被 API 拆成多行:开头连续的无编号 heading 合成一个 H1。
    # 遇编号标题 / 正文实体 / 目录 即停;空行占位保行号不变。
    _tidx = []
    for _i, _l in enumerate(lines):
        _m = _H.match(_l)
        if _m and _m.group(2):
            if _pm(_m.group(2))[0] != "title" or re.search(r"目\s*录", _m.group(2)):
                break                       # 编号标题 / 目录 → 停
            _tidx.append(_i)
        elif _l.strip():
            break                           # 无编号正文实体 → 停
    if len(_tidx) >= 2:
        _merged = " ".join(_H.match(lines[i]).group(2).strip() for i in _tidx)
        lines[_tidx[0]] = "# " + _merged
        for _i in _tidx[1:]:
            lines[_i] = ""

    heads = [(i, len(m.group(1)), m.group(2)) for i, l in enumerate(lines)
             for m in [_H.match(l)] if m and m.group(2)]
    if len(heads) < 4:
        return md, []
    L, line_ocr, rule_yranges = _image_features(im, ocr, cache_key)
    if len(L) < 20:
        return md, []
    body = collections.Counter(l[4] for l in L).most_common(1)[0][0]

    geo = [(line_ocr[k], L[k][4], L[k][0])     # (norm_text, core, y0)
           for k in range(len(L)) if len(line_ocr[k]) >= 2]

    hnorm = [_norm(h[2]) for h in heads]
    pair = _align(hnorm, geo)
    core_of = {hi: geo[pair[hi]][1] for hi in pair}
    line_core = {heads[hi][0]: c for hi, c in core_of.items()}   # 行号 → 字号(core height)
    changes = []

    # 目录区行号范围:「目录」标题起,到章编号第二次从 1 重启为止(区内单独两层分级,
    # 之后各 pass 跳过此区)。
    toc_end_line = -1
    ti = next((i for i, l in enumerate(lines)
               if _H.match(l) and re.search(r"目\s*录", l)), None)
    if ti is not None:
        seen = False
        for i in range(ti + 1, len(lines)):
            mm = _CHAP1.match(lines[i])
            if mm:
                if seen:
                    toc_end_line = i
                    break
                seen = True
        else:
            toc_end_line = len(lines)

    # 目录分级(用户裁决:目录需分级,简单两层):bare-int 空格章 = H1,其余编号条目 = H2。
    # 跟踪章序号:值 = 上一章+1 的 bare-int(即便 API 漏标成正文)也补成 H1(如 9→10 附表)。
    # 目录整段仍被后续 pass 跳过,只在这里就地定级,不改内容/顺序。
    if ti is not None:
        last_chap = 0
        for i in range(ti, min(toc_end_line, len(lines))):
            m = _H.match(lines[i])
            is_head = bool(m and m.group(2))
            t = m.group(2) if is_head else lines[i].strip()
            if not t:
                continue
            if i == ti:
                lines[i] = "# " + t; continue             # 目录标题
            k, p = _pm(t)
            if k == "int" and p:                          # 单整数(1/1./1、)= 章 → H1
                if is_head or p[0] == last_chap + 1:       #   已标 或 延续序列的漏标章(9→10 附表)
                    lines[i] = "# " + t; last_chap = p[0]
            elif k == "dec" and is_head:                  # x.y → 点数=深度(1.1→H2,1.1.1→H3)
                lines[i] = "#" * min(6, len(p)) + " " + t
            elif is_head and k != "title":                # 第X条 / 一、名词 等 → H2
                lines[i] = "## " + t
            # 无编号标题(封面题名)保持原级,不降 —— 目录区里夹的题名是 H1

    # pass 0(用户裁决:淡横线上方 = H1):横线上方最近文本行匹配到预测行,强制为 H1
    # —— 正文行加 `# `、已是标题的改成一个 `#`。顺带救回 API 整章漏标(如 e33f4bfb)。
    line_bottoms = [(l[1], k) for k, l in enumerate(L)]
    rule_lines = set()                      # 被淡横线锚定的预测行号(确认或提升为 H1)
    for r0, r1 in rule_yranges:
        above = [k for b, k in line_bottoms if b <= r0 + 2 and r0 - b <= 300]
        if not above:
            continue
        k = above[-1]
        t = line_ocr[k]
        if len(t) < 3:
            continue
        best, bs = None, 0
        for li, ln in enumerate(lines):
            if li <= toc_end_line or not ln.strip():
                continue
            sc = fuzz.partial_ratio(t, _norm(re.sub(r"^#+\s*", "", ln)))
            if sc > bs:
                best, bs = li, sc
        if best is None or bs < 75:
            continue
        m = _H.match(lines[best])
        body_txt = m.group(2) if m else lines[best].strip()
        # 横线上方偶尔是上一节末行/伪标题/HTML,不是章标题;且格式优先:dec(x.y)
        # 点数已定死深度,横线不覆盖(2.1 永远是子节)。
        if (body_txt.lstrip().startswith("<") or _is_pseudo_heading(body_txt)
                or _pm(body_txt)[0] == "dec"):
            continue
        rule_lines.add(best)
        if not m or len(m.group(1)) != 1:
            lines[best] = "# " + body_txt
            changes.append({"op": "rule_h1", "text": body_txt[:60],
                            "was": (len(m.group(1)) if m else 0)})

    # 编号格式判定(_fmt/_val)+ 约定 rank,供下方栈循环定级。
    def _fmt(txt):
        k, p = _pm(txt)
        if k == "title":
            return None
        if k == "int":
            mm = re.match(r"^\s*\d+\s*([.、])", txt)
            return "int" + (mm.group(1) if mm else "")
        if k == "dec":
            return "dec%d" % len(p)
        return k

    def _val(txt):
        _, p = _pm(txt); return p[0] if p else None

    # 约定 rank:小=浅。章(bare-int/第X章)最浅,条次之,汉字枚举浅于数字枚举,括号浅于同族。
    _RANK = {"chap": 0, "int": 0, "art": 1, "cndun": 2, "cnpar": 3,
             "int、": 4, "int.": 5, "numpar": 6, "circ": 7}
    _ENUM_SUB = {"cnpar", "cndun", "numpar", "circ"}

    def _erank(e):                                 # 栈项的 rank;dec/横线章 = -1(结构父级,不被弹)
        f0 = e[0]
        return -1 if (f0 == "__chap__" or f0.startswith("dec")) else _RANK.get(f0, 5)

    dec_comps = [len(_pm(t)[1]) for li, lv, t in heads if _pm(t)[0] == "dec"]
    dec_floor = min(dec_comps) if dec_comps else 0
    first_numbered = next((li for li, lv, t in heads if _pm(t)[0] != "title"), len(lines))
    # 文档出现的非 dec 格式的 rank 集合:无父级 orphan 定级的地板 —— 最浅格式=H1,
    # 更深格式按"比它浅的 rank 种类数"下沉。仅在缺父级关系时用 rank 兜底(如 c463 四/100)。
    present_ranks = {_RANK.get(_fmt(t), 5) for li, lv, t in heads
                     if li > toc_end_line and _fmt(t) and not _fmt(t).startswith("dec")}

    # ── 伪定义项降级(一致性):带编号标题以「：」引出定义,而其同格式 ±1 兄弟是**正文**
    #   → 它也是正文。如 `## 2. 医学必需:指…条件:` 与正文 `1. 符合通常惯例:指…。` 并列。
    #   GT 里 `96.胆道重建手术:` 这类真标题成串出现(邻项亦标题、无正文兄弟),不受影响。
    _LI = _LEADIN_END + _SENT_END                     # 冒号/句号/分号结尾 = 定义引出或句末 → 正文信号
    meta = []                                         # (li, fmt, val, is_heading)
    for i, ln in enumerate(lines):
        if i <= toc_end_line:
            continue
        m = _H.match(ln)
        t = m.group(2) if (m and m.group(2)) else ln.strip()
        f = _fmt(t) if t else None; v = _val(t) if t else None
        if f and not f.startswith("dec") and v is not None:
            meta.append((i, f, v, bool(m and m.group(2))))
    for i, ln in enumerate(lines):
        m = _H.match(ln)
        if not m or not m.group(2) or i <= toc_end_line:
            continue
        t = m.group(2)
        if t.rstrip()[-1:] not in _LI:
            continue
        f = _fmt(t); v = _val(t)
        if not f or f.startswith("dec") or v is None:
            continue
        nxt = next((_H.match(lines[j]) for j in range(i + 1, len(lines)) if _H.match(lines[j])), None)
        nf = _fmt(nxt.group(2)) if nxt else None      # 子级判断看**格式约定**,不看 API 的 # 数
        if nf and not nf.startswith("dec") and _RANK.get(nf, 5) > _RANK.get(f, 5):
            continue                                  # 真子级=更深枚举(一)挂1.);dec 是平级章节,非子级
        if any(f2 == f and not h2 and abs(v2 - v) == 1 for j, f2, v2, h2 in meta if j != i):
            lines[i] = t
            changes.append({"op": "drop_pseudo", "text": t[:60]})

    # ── 层级 = 栈(父级深度)+ 约定 rank(同级/更深判定)+ dec 绝对 ──────────────────
    #   栈跟踪当前祖先链。新格式:弹掉 rank 更深的祖先,遇同 rank=兄弟同级,否则=父级+1。
    #   dec(x.y) 按点数绝对深度。横线章 push __chap__@1(内容 nest 到 H2);封面标题 = H1
    #   且清栈(内容与标题平级、从 H1 起,不被压深)。无父级碎片枚举按 dec_floor 兜底。
    stack = []                                    # [fmt, level, value]
    for li, ln in enumerate(lines):
        if li <= toc_end_line:
            continue
        m = _H.match(ln)
        if not m or not m.group(2):
            continue
        txt = m.group(2); f = _fmt(txt); v = _val(txt)
        if li in rule_lines:                      # 横线锚点 → H1 章,后续内容纳入其下
            lines[li] = "# " + txt; stack = [["__chap__", 1, None]]; continue
        if f is None:                             # 无编号 → 只可能是题名(H1)或 API 误报
            # 判据 = 字号:大字号(≥正文 1.25×)是真标题(封面/rider 篇名),保留 H1 并清栈;
            # 正常字号的无编号标题 = API 把强调句读成标题 → 删。中段题名只可能是 H1。
            big = line_core.get(li, 0) >= body * 1.25
            if li < first_numbered or re.search(r"目\s*录", txt) or big:
                lines[li] = "# " + txt; stack = []          # 题名:H1,内容从 H1 起
            else:
                lines[li] = txt                             # 正常字号无编号 → API 误报,删
                changes.append({"op": "drop_unnum", "text": txt[:60]})
            continue
        if f.startswith("dec"):
            level = int(f[3:])
            stack = [e for e in stack if e[1] < level]       # dec 点数=绝对深度,关闭更深/同深
            stack.append([f, level, v])
        else:
            r = _RANK.get(f, 5)
            while stack and _erank(stack[-1]) > r:           # 弹掉更深的同类祖先
                stack.pop()
            if stack and _erank(stack[-1]) == r:             # 同 rank = 兄弟 → 同级
                level = stack[-1][1]; stack[-1][2] = v
            else:
                if stack:                                    # 有父级 → 纯关系,父级 +1
                    level = stack[-1][1] + 1
                else:                                        # 无父级(orphan)→ rank 地板兜底
                    rank_floor = 1 + sum(1 for rr in present_ranks if rr < r)
                    dec_fl = dec_floor + 1 if (f in _ENUM_SUB and dec_floor) else 1
                    level = max(rank_floor, dec_fl)          # rank深沉 与 dec碎片 取深者
                stack.append([f, level, v])
        lines[li] = "#" * min(6, level) + " " + txt

    # ── 兄弟一致性(跨区):同格式连续编号的长枚举 → 少数派拉齐到多数派层级 ──────
    #   如 36、~66、共 31 个 int、疾病,若首项被误检横线顶成 H1,其余 30 个 H2,
    #   本 pass 把 36、拉回 H2(同格式同级)。仅作用于 ≥3 项的连续段,不碰短组。
    hd = []                                            # (li, fmt, val, level)
    for li, ln in enumerate(lines):
        m = _H.match(ln)
        if not m or not m.group(2) or li <= toc_end_line:
            continue
        f = _fmt(m.group(2))
        if f is None or f.startswith("dec"):
            continue
        hd.append((li, f, _val(m.group(2)), len(m.group(1))))
    i = 0
    while i < len(hd):
        j = i + 1
        while (j < len(hd) and hd[j][1] == hd[i][1]
               and hd[j][2] is not None and hd[j - 1][2] is not None
               and hd[j][2] == hd[j - 1][2] + 1):
            j += 1
        if j - i >= 3:                                 # 连续同格式 ≥3 项
            lv = [e[3] for e in hd[i:j]]
            mode = collections.Counter(lv).most_common(1)[0][0]
            for li, f, v, l in hd[i:j]:
                if l != mode:
                    lines[li] = "#" * min(6, mode) + " " + _H.match(lines[li]).group(2)
        i = j

    return "\n".join(lines), changes


# 独立驱动:只在已有 pipeline 文本产出上单独施加几何定级(不重跑 API)。主链路
# 已把 correct() 内置于 main.py 的 long 分支,此 CLI 仅供单独调定级时用。

_IMAGES = None          # 子进程经 fork 继承;由 _cli 设定
_PREDS = None


def _apply_one(name):
    from PIL import Image
    from common.preprocess import prep
    from main import _local_ocr
    im = prep(Image.open(os.path.join(_IMAGES, name)))
    new, ch = correct(_PREDS[name], im, _local_ocr(), cache_key=name)
    return name, new, ch


def _cli():
    import sys
    import csv
    import argparse
    from multiprocessing import Pool
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    csv.field_size_limit(sys.maxsize)

    ap = argparse.ArgumentParser(description="在 *_txt.csv 上施加几何标题定级")
    ap.add_argument("--txt", required=True, help="pipeline 文本产出 *_txt.csv")
    ap.add_argument("--images", required=True, help="对应原图目录")
    ap.add_argument("--out", required=True, help="输出 *_raw.csv")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    global _IMAGES, _PREDS
    _IMAGES = a.images
    with open(a.txt, encoding="utf-8") as f:
        _PREDS = {r["file_name"]: r["ground_truth"] for r in csv.DictReader(f)}

    names = [n for n in sorted(os.listdir(_IMAGES)) if n in _PREDS]
    with Pool(a.workers) as p:
        res = p.map(_apply_one, names)

    out = {n: new for n, new, _ in res}
    nch = sum(1 for _, _, ch in res if ch)
    ndrop = sum(len(ch) for _, _, ch in res)
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["file_name", "ground_truth"])
        for n in sorted(_PREDS):
            w.writerow([n, out.get(n, _PREDS[n])])
    print(f"geom 施加 {len(names)} 篇;{ndrop} 处标题改动 across {nch} 篇 → {a.out}")


if __name__ == "__main__":
    _cli()
