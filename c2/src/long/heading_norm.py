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
]

# 圈号必须在 NFKC 之前匹配:NFKC 把 ① 归一成 "1",圈号编号会因此
#   ① 床位费 → "1 床位费" → 误判成顶层 int 章编号(与「1 总则」串成一个序列)
#   ①床位费  → "1床位费"  → 无分隔符,连 int 都不匹配 → 退化成无编号 title
# 圈号与阿拉伯数字是两套并存的编号体系,不是同一字符的宽度变体,归一即信息销毁。
_CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
_CIRC_RE = re.compile("^[" + _CIRC + "]")


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
            # 无编号标题没有编号可推。原先一律赋 anchor_level，等于把它放到全文
            # 最浅层 —— 于是「## (三十一)严重类风湿性关节炎」下面的说明性小标题
            # 反而浅于它的父级。改用 API 在同条带内给出的相对层差:那是它实际
            # 看到的版式，也是这里唯一可用的证据。跨条带不可比 → 退回锚点。
            # 例外:第一个编号标题出现之前是封面题名区(公司名/产品名/条款名常被
            # API 拆成 `#`/`##` 多行),GT 惯例整块算 L1,不按版式层差分级 ——
            # 与 _promote_leading_title 的既有立场一致。
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
            if seen is not None:
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
# 判据全部是结构性的 —— 曾经这里靠两张词表:
#   _TITLE_KW = (公司|保险|条款|附加|目录|合同|银行|基金|年金|信托)  要求标题含行业词
#   _REG_NOTE = (注册编号|备案|编号[：:])                        识别注册号行
# 前者实测是死重(去掉后指标一字不变),后者还写错了 ——「注册号：」里没有「编」字,
# 那个正则根本匹配不上,于是注册号被当成标题的一部分吞进去。改用长串编码的**形态**
# 判据后 level 63.42→63.52、recall 95.48→95.59。词表换形态,更准也更不挑数据集。


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
    # 首标题带编号 → 这页是文档中段片段(如「(三十六) 严重冠心病」),不是封面题名。
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
# 下标 → LaTeX(GT 用 ${T}_{1}{N}_{0}{M}_{0}$,VLM 常吐 unicode 下标 T₁N₀M₀)
# ---------------------------------------------------------------------------
_SUB = "₀₁₂₃₄₅₆₇₈₉"
_SUBMAP = {c: str(i) for i, c in enumerate(_SUB)}
_SUBRUN = re.compile(r"(?:[A-Za-z]+[" + _SUB + r"]+)+")
_SUBPAIR = re.compile(r"([A-Za-z]+)([" + _SUB + r"]+)")
_TNM_BASE = re.compile(r"^(?:T|N|M|pT|pN|pM|cT|cN|cM|ypT|ypN|ypM)$")


# ---------------------------------------------------------------------------
# unicode 上/下标 → 普通字符：GT 里这类字符出现 0 次
# ---------------------------------------------------------------------------
# GT 全量 200 篇统计:unicode 上标/下标字符一个都没有,写法一律是压平的普通数字
#   ×10⁹/L → "109/L"、PaO₂ → "PaO2"
# 而 VLM 常吐 unicode 形式(B 榜 16/50 份、79 个字符),每个都是纯失分。
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


def subscript_to_latex(s):
    def repl(m):
        pairs = _SUBPAIR.findall(m.group(0))
        # GT uses LaTeX for TNM cancer staging (T₁N₀M₀, pT₃...), but keeps
        # ordinary medical notation such as PaO₂ / SaO₂ / FiO₂ as Unicode.  The old
        # global conversion damaged 31/100 training documents to help only eight.
        if not pairs or any(not _TNM_BASE.match(base) for base, _ in pairs):
            return m.group(0)
        body = "".join(
            "{%s}_{%s}" % (base, "".join(_SUBMAP[c] for c in sub))
            for base, sub in pairs)
        return "$" + body + "$"
    return _SUBRUN.sub(repl, s or "")
