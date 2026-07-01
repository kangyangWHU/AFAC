# -*- coding: utf-8 -*-
"""LONG Markdown 标题层级校正。

FinixDoc-VL 是按横条识别的。每个横条里的 Markdown `#` 是局部判断，拼成整篇后
常出现同一编号序列在条带边界被重置成更浅层级的问题。本模块只改标题行的 `#`
数量，不改标题文本和正文。

策略保持保守：
  - 第一条出现的标题作为锚点，保留它的 API 层级。
  - 只在条带边界处，根据上一条带已确认的编号序列推断新条带第一标题的全局层级。
  - 条带内部保留 API 给出的相对层级，只整体平移，避免用全局“编号格式->绝对层级”
    表去覆盖 GT 中真实存在的局部层级差异。
"""
import re
import unicodedata

_H = re.compile(r"^([#＃]{1,6})\s+(.*)$")
_CN = "一二三四五六七八九十百零两〇"
_CN_VAL = {c: i for i, c in enumerate("零一二三四五六七八九", 0)}
_CN_VAL.update({"两": 2, "〇": 0})


def _cn2int(s):
    """简易中文数字转 int，覆盖保险条款里常见的一到一百多。"""
    s = (s or "").strip()
    if not s:
        return None
    if s == "十":
        return 10

    total = 0
    if "百" in s:
        a, s = s.split("百", 1)
        total += (_CN_VAL.get(a, 1) or 1) * 100
    if "十" in s:
        a, s = s.split("十", 1)
        total += (_CN_VAL.get(a, 1) or 1) * 10
    for ch in s:
        if ch in _CN_VAL:
            total += _CN_VAL[ch]
    return total or None


_PATTERNS = [
    (re.compile(r"^第\s*([" + _CN + r"]+)\s*章"), "chap", lambda m: (_cn2int(m.group(1)),)),
    (re.compile(r"^第\s*([" + _CN + r"]+)\s*条"), "art", lambda m: (_cn2int(m.group(1)),)),
    (re.compile(r"^([" + _CN + r"]+)\s*[、.]"), "cndun", lambda m: (_cn2int(m.group(1)),)),
    (re.compile(r"^[（(]\s*([" + _CN + r"]+)\s*[)）]"), "cnpar", lambda m: (_cn2int(m.group(1)),)),
    (re.compile(r"^[（(]\s*(\d+)\s*[)）]"), "numpar", lambda m: (int(m.group(1)),)),
    (re.compile(r"^(\d+(?:\.\d+)+)"), "dec", lambda m: tuple(int(x) for x in m.group(1).split("."))),
    (re.compile(r"^(\d+)\s*[.、\s]"), "int", lambda m: (int(m.group(1)),)),
    (re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]"), "circ", lambda m: ("①②③④⑤⑥⑦⑧⑨⑩".index(m.group(0)) + 1,)),
]


def parse_marker(text):
    """返回 (kind, path)。非编号标题返回 ("title", ())。"""
    t = unicodedata.normalize("NFKC", text or "").strip()
    for rx, kind, value_fn in _PATTERNS:
        m = rx.match(t)
        if m:
            path = value_fn(m)
            if path and path[0] is not None:
                return kind, path
    return "title", ()


def _is_next(prev, cur):
    """同一编号序列的相邻项：1.1 -> 1.2、(五) -> (六)。"""
    pk, pp = prev
    ck, cp = cur
    return (
        pk == ck and
        pp and cp and
        len(pp) == len(cp) and
        pp[:-1] == cp[:-1] and
        cp[-1] == pp[-1] + 1
    )


def _is_child(parent, child):
    """十进制编号的父子关系：2.3 -> 2.3.1。"""
    pk, pp = parent
    ck, cp = child
    if ck != "dec" or not cp:
        return False
    if pk == "dec" and len(cp) == len(pp) + 1 and cp[:-1] == pp:
        return True
    return pk == "int" and len(pp) == 1 and len(cp) == 2 and cp[:-1] == pp


def _series_key(marker):
    kind, path = marker
    if kind == "title" or not path:
        return None
    if kind == "dec":
        return kind, path[:-1]
    return kind, ()


def _clamp_level(level):
    return max(1, min(6, int(level)))


def _is_doc_title(text):
    t = unicodedata.normalize("NFKC", text or "")
    return bool(re.search(r"(条款|保险|公司|合同|附加|目录)", t))


def _same_series(prev, cur):
    return _series_key(prev) is not None and _series_key(prev) == _series_key(cur)


def _find_stack_level(stack, marker):
    for old_marker, old_level in reversed(stack):
        if old_marker == marker:
            return old_level
    return None


def _find_series_level(series_levels, marker):
    sk = _series_key(marker)
    if sk is None:
        return None
    return series_levels.get(sk)


def _find_stack_sibling(stack, marker):
    """栈上仍打开的同序列帧:若当前编号 = 该帧编号+1 → 是它的兄弟,返回其层级。
    栈只含祖先(已关闭的深层兄弟不在),故能避免深层同 scheme 的层级污染。
    例:顶层 int 2 仍在栈上,新来 int 3 → 接回顶层,而非深层列表的 int。"""
    sk = _series_key(marker)
    if sk is None or not marker[1]:
        return None
    cur = marker[1][-1]
    for m, lvl in reversed(stack):
        if _series_key(m) == sk:
            return lvl if (m[1] and m[1][-1] == cur - 1) else None
    return None


def _heading_positions(md):
    lines = (md or "").split("\n")
    matches = []
    for i, line in enumerate(lines):
        m = _H.match(line)
        if m:
            matches.append((i, m))
    return lines, matches


def relevel_markdown(md, offset=0):
    """给单篇 Markdown 标题整体平移层级。

    默认 offset=0，所以对 GT 是严格幂等；传入 offset 时只改 `#` 数量。
    """
    if not offset:
        return md
    lines, matches = _heading_positions(md)
    for i, m in matches:
        old = len(m.group(1))
        lines[i] = "#" * _clamp_level(old + offset) + " " + m.group(2)
    return "\n".join(lines)


def predict_heading_levels(headings, anchor_level, anchor_index=0):
    """用锚点绝对层级预测整篇标题层级。

    headings: [(raw_level, text), ...]，raw_level 来自 API 原始 Markdown `#`。
    anchor_level: 给定锚点的绝对层级，例如总标题=1，首个 section=1/2。
    anchor_index: 锚点在 headings 中的位置，默认第一个标题。

    返回与 headings 等长的新 level。除锚点外，raw_level 不参与计算；API 的 `#`
    只负责告诉我们哪些行是标题。
    """
    if not headings:
        return []
    anchor_index = max(0, min(anchor_index, len(headings) - 1))
    texts = [text for _, text in headings]
    levels = [None] * len(headings)

    stack = []
    marker_levels = {}
    series_levels = {}
    series_last = {}                 # sk -> 该序列最近一次出现的 marker
    last_marker = None
    last_level = None

    for idx, text in enumerate(texts):
        marker = parse_marker(text)
        kind, path = marker
        sk = _series_key(marker)

        if idx == anchor_index:
            level = anchor_level
        elif kind == "title":
            level = anchor_level
        elif (_sib := _find_stack_sibling(stack, marker)) is not None:
            # 接回栈上仍打开的同序列前一项(顶层 int 2 → int 3),
            # 避免被已关闭的深层同 scheme 列表污染层级
            level = _sib
        elif (sk is not None and sk in series_last
              and _is_next(series_last[sk], marker)):
            # 回到已确立的编号序列(十七…深层游走…十八 / (5)…漂移…(6)):
            # 用该序列历史层级,而非相邻 last_level±1
            level = series_levels[sk]
        elif last_marker is not None and _is_child(last_marker, marker):
            level = last_level + 1
        elif last_marker is not None and _is_next(last_marker, marker):
            if _same_series(last_marker, marker):
                level = last_level
            else:
                level = _find_series_level(series_levels, marker) or last_level
        elif kind == "dec":
            parent_level = None
            if len(path) > 1:
                parent_level = _find_stack_level(stack, ("dec", path[:-1]))
                if parent_level is None and len(path) == 2:
                    parent_level = _find_stack_level(stack, ("int", (path[0],)))
            if parent_level is not None:
                level = parent_level + 1
            else:
                seen = _find_series_level(series_levels, marker)
                level = seen if seen is not None else anchor_level + max(0, len(path) - 1)
        elif kind == "int":
            if (path and path[0] == 1 and last_marker
                    and last_marker[0] in ("cndun", "cnpar", "numpar", "circ", "dec")):
                # 值==1 紧跟列表项/小数子项 = 起新子列表 → 当前项的子级
                # (避免深层 "1、住院" 被当成顶层 "1." 的层级)
                level = last_level + 1
            else:
                seen = _find_series_level(series_levels, marker)
                level = seen if seen is not None else anchor_level
        elif kind == "art":
            seen = _find_series_level(series_levels, marker)
            if seen is not None:
                level = seen
            elif last_marker and last_marker[0] in ("int", "dec"):
                level = last_level + 1
            elif idx > 0 and parse_marker(texts[idx - 1])[0] == "title" and not _is_doc_title(texts[idx - 1]):
                level = last_level + 1
            else:
                level = anchor_level
        elif kind == "cndun":
            if path and path[0] == 1 and last_marker and last_marker[0] in ("int", "dec", "art"):
                level = last_level + 1
            elif last_marker and last_marker[0] in ("cnpar", "numpar", "circ"):
                level = max(anchor_level, last_level - 1)
            else:
                seen = _find_series_level(series_levels, marker)
                level = seen if seen is not None else anchor_level
        elif kind in ("cnpar", "numpar", "circ"):
            if path and path[0] == 1 and last_marker and last_marker[0] in ("cndun", "dec", "int", "art"):
                level = last_level + 1
            else:
                seen = _find_series_level(series_levels, marker)
                level = seen if seen is not None else ((last_level + 1) if last_level is not None else anchor_level)
        else:
            level = last_level if last_level is not None else anchor_level

        level = _clamp_level(level)
        levels[idx] = level

        while stack and stack[-1][1] >= level:
            stack.pop()
        stack.append((marker, level))
        if kind != "title":
            marker_levels[marker] = level
            if sk is not None:
                series_levels[sk] = level
                series_last[sk] = marker
        last_marker, last_level = marker, level

    return levels


def relevel_markdown_from_anchor(md, anchor_level=None, anchor_index=0):
    """按给定锚点重算整篇 Markdown 的所有标题 `#`。

    anchor_level=None 时，默认保留锚点原始层级；用于 GT 幂等测试。
    """
    lines, matches = _heading_positions(md)
    if not matches:
        return md

    anchor_index = max(0, min(anchor_index, len(matches) - 1))
    heads = [(len(m.group(1)), m.group(2)) for _, m in matches]
    if anchor_level is None:
        anchor_level = heads[anchor_index][0]
    levels = predict_heading_levels(heads, anchor_level, anchor_index)

    for (i, m), level in zip(matches, levels):
        if level != len(m.group(1)):
            lines[i] = "#" * level + " " + m.group(2)
    return "\n".join(lines)


# 封面大标题常被 VLM 拆成几行普通段落(无 `#`)。识别并提升为 `# ` L1。
_TITLE_KW = re.compile(r"(公司|保险|条款|附加|目录|合同|银行|基金|年金|信托)")
_REG_NOTE = re.compile(r"(注册编号|备案|编号\s*[：:])")


def _promote_leading_title(md):
    """把文档开头被漏标的描述性大标题块(公司名/产品/条款)提升成 `# ` 标题。

    只动第一个已有 `#` 之前的开头行:取连续的【无编号、短、无句末标点、非注册编号】
    行(跳空行),合并成一行 `# `。遇编号行/注册编号/长句即停(避免吃到正文和 (X) 病种项)。
    返回 (新md, 是否提升)。
    """
    lines = md.split("\n")
    first_h = next((i for i, l in enumerate(lines) if _H.match(l)), len(lines))
    block, idxs = [], []
    for i in range(first_h):
        s = lines[i].strip()
        if not s:
            continue
        if _REG_NOTE.search(s):
            break
        if parse_marker(s)[0] != "title":     # 编号行 → Type B,停
            break
        if len(s) > 40 or "。" in s:           # 长句/句末标点 → 正文,停
            break
        block.append(s)
        idxs.append(i)
        if len(block) >= 5:
            break
    joined = " ".join(block)
    if not block or not _TITLE_KW.search(joined):
        return md, False
    lines[idxs[0]] = "# " + joined
    for j in reversed(idxs[1:]):
        del lines[j]
    return "\n".join(lines), True


def _anchor_for(text):
    """首标题 → 锚点绝对层级:无编号描述性标题=L1;任何编号首标题(B/C)=L2
    (视作隐含文档标题之下的大标题)。"""
    return 1 if parse_marker(text)[0] == "title" else 2


def relevel_strips(strips, anchor_level=None):
    """按条带顺序校正标题层级。

    输入/输出都是 API 每个横条返回的 Markdown 字符串列表。API 的 `#` 仅用于
    识别标题行；层级按全局标题文本序列重新预测。先补提升被漏标的封面大标题。
    """
    strips = list(strips)
    for i, md in enumerate(strips):             # Type A:只在首个有内容的条带补标题
        if md and md.strip():
            strips[i], _ = _promote_leading_title(md)
            break

    docs = []
    headings = []
    for doc_i, md in enumerate(strips):
        lines, matches = _heading_positions(md)
        docs.append((lines, matches))
        for i, m in matches:
            headings.append((doc_i, i, len(m.group(1)), m.group(2)))

    if not headings:
        return strips

    if anchor_level is None:
        anchor_level = _anchor_for(headings[0][3])   # 首标题定锚(L1 / 大标题 L2)
    levels = predict_heading_levels([(raw, text) for _, _, raw, text in headings], anchor_level)
    for (doc_i, line_i, raw, text), level in zip(headings, levels):
        if level != raw:
            docs[doc_i][0][line_i] = "#" * level + " " + text

    return ["\n".join(lines) for lines, _ in docs]


# ---------------------------------------------------------------------------
# 目录列表项 → 标题
# ---------------------------------------------------------------------------
_BULLET = re.compile(r"^\s*[\*\-\+]\s+(.*)$")


def toc_bullets_to_headings(md):
    """把「目录」区里的带编号列表项(`* 1.1 合同构成`)转成标题(占位 `# `),
    层级留给 relevel 按编号定(1.1 → 1. 的子级)。只在目录区内改,遇正文段落即停。"""
    lines = md.split("\n")
    start = next((i for i, l in enumerate(lines)
                  if _H.match(l) and "目录" in l), None)
    if start is None:
        return md
    for i in range(start + 1, len(lines)):
        s = lines[i].strip()
        if not s or _H.match(lines[i]):        # 空行/已是标题 → 继续
            continue
        m = _BULLET.match(lines[i])
        if not m:                              # 普通段落 → 目录结束
            break
        text = m.group(1).strip()
        if parse_marker(text)[0] == "title":   # 无编号列表项 → 非目录条目,停
            break
        lines[i] = "# " + text                 # 转标题(占位级)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 标点半角化(官方口径:GT/VLM 均为半角)
# ---------------------------------------------------------------------------
def _build_punct_map():
    m = {}
    for cp in range(0xFF01, 0xFF5F):           # 全角 ASCII 标点(排除字母/数字)
        half = chr(cp - 0xFEE0)
        if not half.isalnum():
            m[chr(cp)] = half
    m["　"] = " "                          # 全角空格
    m.update({
        "。": ".", "、": ",", "，": ",", "：": ":", "；": ";",
        "？": "?", "！": "!", "（": "(", "）": ")",
        "「": '"', "」": '"', "『": "'", "』": "'",
        "【": "[", "】": "]", "〔": "[", "〕": "]",
        "《": "<", "》": ">", "〈": "<", "〉": ">",
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "—": "-", "－": "-", "～": "~",
    })
    return m


_PUNCT_TRANS = str.maketrans(_build_punct_map())


def to_halfwidth_punct(md):
    """所有标点转半角。不动字母/数字/圈号(①)/`#`。"""
    return (md or "").translate(_PUNCT_TRANS).replace("…", "...")


# ---------------------------------------------------------------------------
# 罗马数字 ASCII → unicode(GT 用 Ⅲ/Ⅳ,VLM 常吐 III/IV;仅在 级/期/型/度/区 语境转)
# ---------------------------------------------------------------------------
_ROMAN_MAP = {"I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ", "V": "Ⅴ",
              "VI": "Ⅵ", "VII": "Ⅶ", "VIII": "Ⅷ", "IX": "Ⅸ", "X": "Ⅹ",
              "XI": "Ⅺ", "XII": "Ⅻ"}
_ROMAN_RE = re.compile(
    r"(?<![A-Za-z])(VIII|XII|VII|III|XI|VI|IV|IX|II|V|X|I)(?![A-Za-z])")
_ROMAN_GRADE = set("级期型度区类")           # 医学分级/分区/分型语境
_ROMAN_ENUM = set(",，、和或") | _ROMAN_GRADE  # 再加枚举分隔(如 Ⅰ,Ⅱ,Ⅲ或Ⅳ区)


def roman_to_unicode(s):
    s = s or ""

    def repl(m):
        tok = m.group(1)
        nxt = s[m.end()] if m.end() < len(s) else ""
        prev = s[m.start() - 1] if m.start() > 0 else ""
        if tok == "X":                       # X线/X光/X片 是字母,仅分级语境才转
            return _ROMAN_MAP[tok] if nxt in _ROMAN_GRADE else tok
        if nxt in _ROMAN_ENUM or prev == "素":
            return _ROMAN_MAP[tok]
        return tok

    return _ROMAN_RE.sub(repl, s)


# ---------------------------------------------------------------------------
# 下标 → LaTeX(GT 用 ${T}_{1}{N}_{0}{M}_{0}$,VLM 常吐 unicode 下标 T₁N₀M₀)
# ---------------------------------------------------------------------------
_SUB = "₀₁₂₃₄₅₆₇₈₉"
_SUBMAP = {c: str(i) for i, c in enumerate(_SUB)}
_SUBRUN = re.compile(r"(?:[A-Za-z]+[" + _SUB + r"]+)+")
_SUBPAIR = re.compile(r"([A-Za-z]+)([" + _SUB + r"]+)")


def subscript_to_latex(s):
    def repl(m):
        body = "".join(
            "{%s}_{%s}" % (base, "".join(_SUBMAP[c] for c in sub))
            for base, sub in _SUBPAIR.findall(m.group(0)))
        return "$" + body + "$"
    return _SUBRUN.sub(repl, s or "")
