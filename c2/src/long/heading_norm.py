# -*- coding: utf-8 -*-
"""LONG Markdown 标题层级校正。

FinixDoc-VL 是按横条识别的。每个横条里的 Markdown `#` 是局部判断，拼成整篇后
常出现同一编号序列在条带边界被重置成更浅层级的问题。本模块只改标题行的 `#`
数量，不改标题文本和正文。

策略保守：首个标题作锚点保留其 API 层级；条带边界处按上一条带已确认的编号序列推断
新条带首标题的全局层级；条带内部保留 API 相对层级只整体平移，不用全局“编号格式->绝对
层级”表覆盖 GT 真实的局部层级差异。
"""
import re
import unicodedata
import collections

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
    # 单边括号 一)/（一 :OCR 常丢半个括号,丢了就解析成无编号→被当误报删,
    # 本该按 cnpar 定级。
    (re.compile(r"^([" + _CN + r"]+)\s*[)）]"), "cnpar", lambda m: (_cn2int(m.group(1)),)),
    (re.compile(r"^[（(]\s*(\d+)\s*[)）]"), "numpar", lambda m: (int(m.group(1)),)),
    (re.compile(r"^(\d+(?:\.\d+)+)"), "dec", lambda m: tuple(int(x) for x in m.group(1).split("."))),
    (re.compile(r"^(\d+)\s*[.、\s]"), "int", lambda m: (int(m.group(1)),)),
]

# 圈号必须在 NFKC 之前匹配:NFKC 把 ① 归一成 "1"(① 床位费→"1 床位费" 误判成顶层 int
# 章编号)。圈号与阿拉伯数字是两套并存编号体系,非宽度变体,归一即信息销毁。
_CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_CIRC_RE = re.compile("^[" + _CIRC + "]")
# 章节重启标记:章号 1(1空格 / 1. / 1、),排除 1.1 这类 dec(1 后跟"可选.、+数字")—— 目录护栏用
_CHAP1 = re.compile(r"^#*\s*1(?![.、]?\d)[.、\s]")


def parse_marker(text):
    """返回 (kind, path)。非编号标题返回 ("title", ())。"""
    s = (text or "").strip()
    m = _CIRC_RE.match(s)
    if m:
        return "circ", (_CIRC.index(m.group(0)) + 1,)
    t = unicodedata.normalize("NFKC", s)
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


def _find_stack_level(stack, marker):
    for old_marker, old_level in reversed(stack):
        if old_marker == marker:
            return old_level
    return None


def _find_chapter_level(stack):
    """栈上最深的「章」(顶层 int 编号)的层级;没有则 None。

    栈只含祖先,所以命中即意味着"当前标题在这一章之内"。
    只认单层 int(1/2/3…),不认 dec(1.1) —— 后者是小节不是章。
    """
    for m, lvl in reversed(stack):
        if m[0] == "int" and len(m[1]) == 1:
            return lvl
    return None


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


def predict_heading_levels(headings, anchor_level, anchor_index=0):
    """用锚点绝对层级预测整篇标题层级。

    headings: [(raw_level, text, strip_id), ...]，raw_level 来自 API 原始 Markdown `#`，
              strip_id 标识该标题出自哪个横条(用于判断两个 raw_level 是否可比)。
    anchor_level: 给定锚点的绝对层级，例如总标题=1，首个 section=1/2。
    anchor_index: 锚点在 headings 中的位置，默认第一个标题。

    返回与 headings 等长的新 level。层级主要由编号文本推出；raw_level 只在
    编号推不出层级时(无编号标题)作为**同条带内的相对**证据使用 —— API 的绝对
    层级跨条带会漂(同文档 strip0 章=`##`、strip1 章=`###`)，不可跨条带比较。
    """
    if not headings:
        return []
    anchor_index = max(0, min(anchor_index, len(headings) - 1))
    texts = [text for _, text, _ in headings]
    raws = [raw for raw, _, _ in headings]
    sids = [sid for _, _, sid in headings]
    levels: list = [None] * len(headings)

    stack = []
    series_levels = {}
    series_last = {}                 # sk -> 该序列最近一次出现的 marker
    last_marker = None
    last_level = None
    numbered_seen = False            # 是否已出现过带编号的标题(封面题名区判据)
    toc_level = None                 # 「目录」标题层级；目录编号不污染正文编号状态
    toc_seen_first = False           # 目录中已见过顶层 1；再次出现 1 视为正文重启

    for idx, text in enumerate(texts):
        marker = parse_marker(text)
        kind, path = marker
        sk = _series_key(marker)

        # 目录是一个独立编号作用域：「目录」=L1时，1=L2、1.1=L3。
        # 若直接复用下方的全局 series_levels，目录中的 1..8 会把正文
        # 再次出现的 1..8 锁在错误层级。新文档标题或顶层 1 重启时退出目录。
        in_toc = toc_level is not None
        toc_restart = (in_toc and kind == "int" and path == (1,) and toc_seen_first)
        toc_ended_by_title = in_toc and kind == "title" and "目录" not in text
        if toc_restart or toc_ended_by_title:
            toc_level = None
            toc_seen_first = False
            stack.clear()
            series_levels.clear()
            series_last.clear()
            last_marker = None
            last_level = None
            in_toc = False

        if in_toc and kind != "title":
            if kind == "dec":
                level = toc_level + max(1, len(path))
            else:
                level = toc_level + 1
            levels[idx] = _clamp_level(level)
            if kind == "int" and path == (1,):
                toc_seen_first = True
            # 目录内部不更新正文的 stack / series_levels / last_marker。
            continue

        if idx == anchor_index:
            level = anchor_level
        elif kind == "title":
            # 无编号标题没编号可推。用 API 同条带内的相对层差(唯一可用证据,跨条带不可比
            # → 退回锚点),而非一律赋 anchor_level(会浅于其带编号父级)。
            # 例外:首个编号标题之前是封面题名区(公司名/产品名常被拆成多行 `#`),GT 整块
            # 算 L1,不按版式层差分级(同 _promote_leading_title 立场)。
            if (numbered_seen and idx > 0 and last_level is not None
                    and sids[idx] == sids[idx - 1]):
                level = last_level + (raws[idx] - raws[idx - 1])
            else:
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
        elif last_marker is not None and last_level is not None and _is_child(last_marker, marker):
            level = last_level + 1
        # (相邻 _is_next(last_marker,·) 分支不需要:_is_next 成立 ⇒ 同 series key,而
        #  series_last 每个编号项都更新 ⇒ 上面的 series_last 分支必先命中,且给值相同)
        elif kind == "dec":
            parent_level = None
            if len(path) > 1:
                parent_level = _find_stack_level(stack, ("dec", path[:-1]))
                if parent_level is None and len(path) == 2:
                    parent_level = _find_stack_level(stack, ("int", (path[0],)))
            if parent_level is not None:
                level = parent_level + 1
            else:
                seen = series_levels.get(sk)
                level = seen if seen is not None else anchor_level + max(0, len(path) - 1)
        elif kind == "int":
            if (path and path[0] == 1 and last_marker and last_level is not None
                    and last_marker[0] in ("cndun", "cnpar", "numpar", "circ", "dec")):
                # 值==1 紧跟列表项/小数子项 = 起新子列表 → 当前项的子级
                # (避免深层 "1、住院" 被当成顶层 "1." 的层级)
                level = last_level + 1
            else:
                seen = series_levels.get(sk)
                level = seen if seen is not None else anchor_level
        elif kind == "art":
            seen = series_levels.get(sk)
            # 章在栈上(当前条的祖先)→ 条 = 章+1,**优先于序列历史**。
            # GT:章后出现的条 762 处 89.4% 深一级、8.7% 同级。
            # 序列历史会把条钉死在首次层级:开头章被 API 漏标时第一条落 L1,此后真章
            # 出现 `seen` 仍先命中 → 章条一路同级。
            chap = _find_chapter_level(stack)
            if chap is not None and (seen is None or seen <= chap):
                level = chap + 1
            elif seen is not None:
                level = seen
            elif last_marker and last_level is not None and last_marker[0] in ("int", "dec"):
                level = last_level + 1
            elif (last_level is not None and idx > 0
                  and parse_marker(texts[idx - 1])[0] == "title" and not _is_doc_title(texts[idx - 1])):
                level = last_level + 1
            else:
                level = anchor_level
        elif kind == "cndun":
            if (path and path[0] == 1 and last_marker and last_level is not None
                    and last_marker[0] in ("int", "dec", "art")):
                level = last_level + 1
            elif last_marker and last_level is not None and last_marker[0] in ("cnpar", "numpar", "circ"):
                level = max(anchor_level, last_level - 1)
            else:
                seen = series_levels.get(sk)
                level = seen if seen is not None else anchor_level
        elif kind in ("cnpar", "numpar", "circ"):
            if (path and path[0] == 1 and last_marker and last_level is not None
                    and last_marker[0] in ("cndun", "dec", "int", "art")):
                level = last_level + 1
            else:
                seen = series_levels.get(sk)
                level = seen if seen is not None else ((last_level + 1) if last_level is not None else anchor_level)
        else:
            level = last_level if last_level is not None else anchor_level

        # 后置约束:条不得与所在章(顶层 int)同级或更浅(GT 762 处:89.4% 深一级、8.7%
        # 同级、0 更浅)。必须后置——前面「回到已确立序列」分支会把条钉死在首次层级。
        if kind == "art":
            chap = _find_chapter_level(stack)
            if chap is not None and level <= chap:
                level = chap + 1

        level = _clamp_level(level)
        levels[idx] = level

        while stack and stack[-1][1] >= level:
            stack.pop()
        stack.append((marker, level))
        if kind != "title":
            if sk is not None:
                series_levels[sk] = level
                series_last[sk] = marker
        last_marker, last_level = marker, level
        if kind != "title":
            numbered_seen = True

        if kind == "title" and "目录" in text:
            toc_level = level
            toc_seen_first = False
            # 进入目录时也切断之前的编号历史。
            stack.clear()
            series_levels.clear()
            series_last.clear()
            last_marker = marker
            last_level = level

    return levels


_CODE_RUN = re.compile(r"[A-Za-z0-9]{8,}")   # 注册号/备案号那种长串编码 = 元数据行

# 封面大标题常被 VLM 拆成几行普通段落(无 `#`)。识别并提升为 `# ` L1。
# 判据全结构性:曾用行业词表(公司|保险|…)+注册号词表,前者实测死重、后者正则写错(漏「编」
# 字)。改用长串编码的**形态**判据后 level 63.42→63.52、recall 95.48→95.59,更准更通用。


def _promote_leading_title(md):
    """把文档开头被漏标的描述性大标题块提升成 `# ` 标题。

    只动第一个已有 `#` 之前的开头行:取连续的【无编号、短、无句末标点、无长串编码】
    行(跳空行),合并成一行 `# `。遇编号行/编码行/长句即停(避免吃到正文和 (X) 病种项)。
    """
    lines = md.split("\n")
    first_h = next((i for i, l in enumerate(lines) if _H.match(l)), len(lines))
    block, idxs = [], []
    for i in range(first_h):
        s = lines[i].strip()
        if not s:
            continue
        if _CODE_RUN.search(s):               # 注册号/备案号 → 元数据,停
            break
        if parse_marker(s)[0] != "title":     # 编号行 → Type B,停
            break
        if len(s) > 40 or "。" in s:           # 长句/句末标点 → 正文,停
            break
        block.append(s)
        idxs.append(i)
        if len(block) >= 5:
            break
    if not block:
        return md
    lines[idxs[0]] = "# " + " ".join(block)
    for j in reversed(idxs[1:]):
        del lines[j]
    return "\n".join(lines)




def merge_wrapped_title(md, max_lines=3, max_len=44):
    """封面大标题被换行拆开时合回一条。

    版面上「XX保险股份有限公司 / 附加学生、幼儿住院医疗保险 / (互联网专属)条款」是
    同一个标题的折行，GT 写成一条；而 API 只把第一行标成 `#`，其余留作正文 ——
    训练集 100 篇里 31 篇的文档标题因此被截短(首标题与 GT 一致仅 51/100)。

    _promote_leading_title 治不了这种:它只合并**第一个 `#` 之前**的行,
    首行已被标成标题时那个循环是空的。

    停止条件全是结构判据,不认任何具体词:
      - 遇到下一个标题行(`#`) —— 那是另一个分节点,不是折行
      - 行内出现 ≥8 位字母数字长串 —— 注册号/备案号这类元数据行
      - 带编号 / 过长 / 含句号 —— 正文特征
    """
    lines = md.split("\n")
    h = next((i for i, l in enumerate(lines) if _H.match(l)), None)
    if h is None:
        return md
    head = _H.match(lines[h]).group(2).strip()
    # 首标题带编号 → 这页是文档中段片段(如某疾病清单项),不是封面题名。
    # 中段页的首标题后面跟的就是正文,吞进来会把整句释义并进标题。
    if parse_marker(head)[0] != "title":
        return md

    acc, take = [head], []
    for j in range(h + 1, len(lines)):
        s = lines[j].strip()
        if not s:
            continue
        if (_H.match(s) or _CODE_RUN.search(s) or parse_marker(s)[0] != "title"
                or len(s) > max_len or "。" in s or len(take) >= max_lines):
            break
        acc.append(s)
        take.append(j)

    if not take:
        return md
    lines[h] = _H.match(lines[h]).group(1) + " " + " ".join(acc)
    for k in reversed(take):
        del lines[k]
    return "\n".join(lines)


def _anchor_for(text):
    """首标题 → 锚点绝对层级。

    能看到描述性文档标题时它是 L1。页面从正文中途开始时不能再
    凭空假设一个「隐含文档标题」：8 释义/第十四条应从 L1 锚定，
    8.1/8.1.1 则按编号深度锚定。其他无法从文本确定绝对层级的
    列表型编号保持原来的 L2 保守值。"""
    kind, path = parse_marker(text)
    if kind == "title":
        return 1
    if kind == "dec":
        return _clamp_level(len(path))
    if kind in ("int", "art", "chap"):
        return 1
    return 2


# ---------------------------------------------------------------------------
# 伪标题降级：标题是「命名」，不是「陈述句」
# ---------------------------------------------------------------------------
_SENT_END = "；;"                     # 分号结尾:GT 7723 个标题里 0 个
_LEADIN_END = "：:。"                  # 冒号/句号结尾 + 无编号:GT 里同样 0 个
MAX_HEAD_LEN = 40                     # 标题长度上限,见 _is_pseudo_heading


def _is_pseudo_heading(text):
    """加粗强调句被 API 读成了标题。

    标题是给一段内容命名,不会以句末标点收尾。GT 全量 7723 个标题:
      - 以 `；` 结尾 0 个
      - 无编号且以 `：`/`。` 结尾 0 个
    「无编号」这个限定不能去:GT 里有 100 个**带编号**的冒号结尾标题
    (`（三）疾病全残保险金申请：`、`96.胆道重建手术：`),那些是真标题。
    无编号 + 冒号 = 引出下文列表的正文引导句(`下列疾病不在保障范围内：`)。
    """
    t = (text or "").strip()
    if not t:
        return False
    if t[-1] in _SENT_END:
        return True
    return t[-1] in _LEADIN_END and parse_marker(t)[0] == "title"


def demote_pseudo_headings(strips):
    """把伪标题行的 `#` 去掉,变回正文。

    豁免有子级的:后面紧跟着 API 层级更深的标题,说明它确实是个分节点
    (如 `一) 必选责任:` 下面挂着 1./2./3.),降级反而更糟。
    子级判断用 API 的原始层级,且只在同一条带内比 —— 绝对层级跨条带会漂。
    """
    out = []
    for md in strips:
        lines, matches = _heading_positions(md)
        for k, (i, m) in enumerate(matches):
            if not _is_pseudo_heading(m.group(2)):
                continue
            nxt = matches[k + 1][1] if k + 1 < len(matches) else None
            if nxt is not None and len(nxt.group(1)) > len(m.group(1)):
                continue                       # 有子级 → 是真分节点,留着
            lines[i] = m.group(2).strip()
        out.append("\n".join(lines))
    return out


def relevel_strips(strips, anchor_level=None):
    """按条带顺序校正标题层级。

    输入/输出都是 API 每个横条返回的 Markdown 字符串列表。API 的 `#` 仅用于
    识别标题行；层级按全局标题文本序列重新预测。先补提升被漏标的封面大标题。
    """
    strips = list(strips)
    for i, md in enumerate(strips):             # Type A:只在首个有内容的条带补标题
        if md and md.strip():
            strips[i] = merge_wrapped_title(_promote_leading_title(md))
            break

    # 伪标题先降级:它们既占错层级,又会污染下方编号序列的 last_marker/last_level
    strips = demote_pseudo_headings(strips)

    docs = []
    headings = []
    for doc_i, md in enumerate(strips):
        lines, matches = _heading_positions(md)
        docs.append((lines, matches))
        for i, m in matches:
            headings.append((doc_i, i, len(m.group(1)), m.group(2)))

    if not headings:
        return strips

    # 无文档标题/父级上下文，却从「(四十八)…」这类中途括号列表
    # 开始时，文本本身无法判断它是 L2/L3 还是普通列表。训练集上强制
    # 全局重排稳定退化，因此只对「非一起始」的孤立括号序列保留 API 层级。
    first_marker = parse_marker(headings[0][3])
    if (first_marker[0] == "cnpar" and first_marker[1]
            and first_marker[1][0] != 1):
        return strips

    if anchor_level is None:
        anchor_level = _anchor_for(headings[0][3])   # 首标题定锚(L1 / 大标题 L2)
    levels = predict_heading_levels(
        [(raw, text, doc_i) for doc_i, _, raw, text in headings], anchor_level)
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
# unicode 上/下标 → 普通字符：GT 里这类字符出现 0 次
# ---------------------------------------------------------------------------
# GT 全量 200 篇:unicode 上/下标 0 次,一律压平(×10⁹/L→"109/L");VLM 常吐 unicode
# (B 榜 16/50 份、79 字符),每个纯失分。
_SCRIPT_MAP = str.maketrans({
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")",
    "ₐ": "a", "ₑ": "e", "ₒ": "o", "ₓ": "x", "ₕ": "h",
    "ₖ": "k", "ₗ": "l", "ₘ": "m", "ₙ": "n", "ₚ": "p", "ₛ": "s", "ₜ": "t",
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")", "ⁿ": "n",
})


def flatten_scripts(s):
    """unicode 上/下标压成普通字符。"""
    return (s or "").translate(_SCRIPT_MAP)


# API 对公式排版区会吐 $$ \text{未满期保险费} = \text{保费} \times (…) $$。
# GT 全量 200 篇:公式一律纯文本(87 处「未满期保险费=保险费×[1-…]」),$$ 出现 0 次。
_TEXT_CMD = re.compile(r"\\text\s*{([^{}]*)}")
_FRAC_CMD = re.compile(r"\\[df]?frac\s*{([^{}]*)}\s*{([^{}]*)}")


def flatten_math(md):
    """$$…$$ / $…$ 压成 GT 风格的纯文本公式。"""
    def conv(m):
        e = m.group(1)
        e = _TEXT_CMD.sub(r"\1", e)
        e = _FRAC_CMD.sub(r"\1/\2", e)
        e = (e.replace(r"\times", "×").replace(r"\div", "÷")
               .replace(r"\cdot", "·").replace(r"\%", "%"))
        return re.sub(r"\s+", "", e)
    md = re.sub(r"\$\$(.+?)\$\$", conv, md or "", flags=re.S)
    md = re.sub(r"\$([^$\n]+)\$", conv, md)
    return md


# ---------------------------------------------------------------------------
# 编号序列补标题:两个同级同体系标题之间的编号空洞,由夹在中间的正文行补上
# ---------------------------------------------------------------------------
def infer_missing_headings(md):
    """(三)…(七) 两个 H3 之间缺 (四)(五)(六),而它们正以正文形式躺在两者之间
    → 升为同级标题。API 漏标标题是随机的,但编号序列是连续的 —— 用序列约束反推。

    三重门槛,全部满足才补(B 榜实测 90 处命中、零可见误报;此前不带位置约束的
    全文搜索版本只敢补 1 处):
      - 位置:候选行严格位于左右两个锚点标题之间
      - 取值:恰好是缺失的编号,同体系(cnpar/cndun/numpar/int),每个缺号唯一
      - 顺序:候选行的出现顺序与编号顺序一致
    run-in 长行(标题+释义连排)照常升级 —— 同序列的兄弟标题本就是这种排版,
    文档内格式一致优先。
    """
    lines = md.split("\n")
    # 目录区护栏(用户裁决:目录不修改):目录 heading 起,到章编号第二次从 1 重启为止
    toc_end = -1
    ti = next((i for i, l in enumerate(lines) if _H.match(l) and re.search(r"目\s*录", l)), None)
    if ti is not None:
        seen = False
        for i in range(ti + 1, len(lines)):
            mm = _CHAP1.match(lines[i])
            if mm:
                if seen:
                    toc_end = i
                    break
                seen = True
        else:
            toc_end = len(lines)
    heads = []
    for i, l in enumerate(lines):
        if i <= toc_end:
            continue
        m = _H.match(l)
        if m and m.group(2):
            k, p = parse_marker(m.group(2))
            if k != "title" and p and len(p) == 1:
                heads.append((i, len(m.group(1)), k, p[0]))
    changed = False
    for (i1, lv1, k1, v1), (i2, lv2, k2, v2) in zip(heads, heads[1:]):
        if lv1 != lv2 or k1 != k2 or v2 <= v1 + 1:
            continue
        missing = list(range(v1 + 1, v2))
        cand = {}
        for j in range(i1 + 1, i2):
            s = lines[j].strip()
            if not s or _H.match(lines[j]):
                continue
            kk, pp = parse_marker(s)
            if kk == k1 and pp and len(pp) == 1 and pp[0] in missing:
                cand.setdefault(pp[0], []).append(j)
        if missing and all(len(cand.get(v, [])) == 1 for v in missing):
            idxs = [cand[v][0] for v in missing]
            if idxs == sorted(idxs):
                for j in idxs:
                    lines[j] = "#" * lv1 + " " + lines[j].strip()
                changed = True
    return "\n".join(lines) if changed else md


def demote_stray_enum(md):
    """孤立编号标题、其序列兄弟全是正文 → 它也是正文(序列推断的镜像)。

    案例:责任免除列举 1、…5、全为正文,唯独 6、脊椎间盘疾病。被 API 标了 #,
    编号豁免让它躲过句号降级,孤例让它躲过几何整组规则。
    判据(全部满足才降,GT 200 篇误杀 0):
      - 紧邻前两个非空行是同 kind、值 v-1/v-2 的**正文**行(它长在正文序列的尾巴上)
      - 全文档不存在 (kind, v-1) 或 (kind, v+1) 的**标题**(序列内没有标题邻居;
        疾病清单 (4) 前面恰有 (3) 子项正文的碰撞由此护栏挡住 —— 那里 (3)(5) 都是标题)
    """
    lines = md.split("\n")
    nb = [(i, l) for i, l in enumerate(lines) if l.strip()]
    head_vals = set()
    for _, l in nb:
        m = _H.match(l)
        if m and m.group(2):
            k, p = parse_marker(m.group(2))
            if k != "title" and p and len(p) == 1:
                head_vals.add((k, p[0]))
    changed = False
    for j, (i, l) in enumerate(nb):
        m = _H.match(l)
        if not m or not m.group(2) or j < 2:
            continue
        k, p = parse_marker(m.group(2))
        if k == "title" or not p or len(p) != 1 or p[0] < 3:
            continue
        v = p[0]
        if (k, v - 1) in head_vals or (k, v + 1) in head_vals:
            continue
        a, b = nb[j - 1][1], nb[j - 2][1]
        if _H.match(a) or _H.match(b):
            continue
        ka, pa = parse_marker(a.strip())
        kb, pb = parse_marker(b.strip())
        if (ka == k and kb == k and pa and pb
                and pa[0] == v - 1 and pb[0] == v - 2):
            lines[i] = m.group(2)
            changed = True
    return "\n".join(lines) if changed else md


def enforce_enum_consistency(md, gate_len=False):
    """编号兄弟序列内多数派表决:全标题或全正文,不能混(用户原则:一致性优先)。

    同 kind 且值递增的成员链(标题行与正文行都算成员,值回落即断链、目录区跳过),
    ≥3 成员且状态混杂时,少数派服从多数派:
      多数为标题 → 正文成员升为标题层级众数(gate_len 时 run-in >MAX_HEAD_LEN 不升)
      多数为正文 → 标题成员降为正文
    GT 依据:序列内部 GT 几乎全一致(释义 183:0、疾病清单 528:1、正文列举 0:N)。
    我们混排时输的是少数派那半;向多数派对齐是期望收益最大的选择。
    取代原 demote_stray_enum 的孤例特判(它只覆盖"紧邻前两行"一种形态)。
    """
    lines = md.split("\n")
    # 目录区
    toc_end = -1
    ti = next((i for i, l in enumerate(lines)
               if _H.match(l) and re.search(r"目\s*录", l)), None)
    if ti is not None:
        seen = False
        for i in range(ti + 1, len(lines)):
            mm = _CHAP1.match(lines[i])
            if mm:
                if seen:
                    toc_end = i
                    break
                seen = True
        else:
            toc_end = len(lines)

    items = []                                  # (line_idx, kind, v, level|0)
    for i, l in enumerate(lines):
        if i <= toc_end or not l.strip():
            continue
        m = _H.match(l)
        txt = m.group(2) if (m and m.group(2)) else l.strip()
        k, p = parse_marker(txt)
        if k != "title" and p and len(p) == 1:
            items.append((i, k, p[0], len(m.group(1)) if m else 0))

    per_kind = {}
    out_chains = []
    for it in items:
        i, k, v, lv = it
        cur = per_kind.get(k)
        if cur and v > cur[-1][2] and v - cur[-1][2] <= 5:
            cur.append(it)
        else:
            if cur and len(cur) >= 3:
                out_chains.append(cur)
            per_kind[k] = [it]
    for cur in per_kind.values():
        if len(cur) >= 3:
            out_chains.append(cur)

    changed = False
    for chain in out_chains:
        hs = [it for it in chain if it[3] > 0]
        bs = [it for it in chain if it[3] == 0]
        if not hs or not bs:
            continue
        if len(hs) >= len(bs):
            lvl = collections.Counter(x[3] for x in hs).most_common(1)[0][0]
            for i, k, v, _ in bs:
                if gate_len and len(lines[i].strip()) > MAX_HEAD_LEN:
                    continue
                lines[i] = "#" * lvl + " " + lines[i].strip()
                changed = True
        else:
            for i, k, v, lv in hs:
                m = _H.match(lines[i])
                lines[i] = m.group(2)
                changed = True
    return "\n".join(lines) if changed else md
