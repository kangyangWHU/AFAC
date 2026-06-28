# -*- coding: utf-8 -*-
"""本地三指标复现评测器。

官方综合分：
    Overall = [ (1−TextEdit)×100 + TableTEDS + (1−ReadOrderEdit)×100 ] / 3   (满分 100)

三项含义（官方释义）：
  - TextEdit       字级 Levenshtein 编辑距离 ÷ 标准文本总长度（归一化损失，越小越好）
  - TableTEDS      表格树编辑距离相似度（越大越好，满分 100；本模块内部用 [0,1] 再 ×100）
  - ReadOrderEdit  逻辑块级阅读流顺序的归一化编辑距离（越小越好）

⚠️ 官方未公开评测脚本，本实现为**合理复现**，用于本地相对排序与迭代。
   关键假设见下方注释，待 error_profiling 阶段拿到 API 真实输出后与线上分校准。
"""
import re
import unicodedata
from rapidfuzz.distance import Levenshtein

from config import TABLE_OPEN_RE, TABLE_CLOSE_RE
from teds import teds_score, _first_table

# 同时匹配整段 <table>...</table>
_TABLE_BLOCK_RE = re.compile(TABLE_OPEN_RE + r".*?" + TABLE_CLOSE_RE,
                             re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# 文本归一化
# ---------------------------------------------------------------------------
def normalize_text(s):
    """轻量归一化：统一全角/半角、去多余空白。
    不做激进清洗，避免掩盖真实的漏字/错字（评测要如实反映）。
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)          # 全角→半角、兼容字符统一
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 行内多空格压成一个；保留换行（换行参与阅读流，但文本编辑距离里影响很小）
    s = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


# ---------------------------------------------------------------------------
# 文档拆块：文本块 / 表格块
# ---------------------------------------------------------------------------
def split_blocks(doc):
    """把文档按 <table> 边界切成有序块列表：
    [{'type':'table'|'text', 'raw':..., 'norm':...}, ...]
    顺序即阅读流顺序。
    """
    blocks = []
    pos = 0
    for m in _TABLE_BLOCK_RE.finditer(doc or ""):
        if m.start() > pos:
            txt = doc[pos:m.start()]
            if txt.strip():
                blocks.append({"type": "text", "raw": txt})
        blocks.append({"type": "table", "raw": m.group(0)})
        pos = m.end()
    if pos < len(doc or ""):
        txt = doc[pos:]
        if txt.strip():
            blocks.append({"type": "text", "raw": txt})
    return blocks


def _text_only(doc):
    """移除所有表格块后的纯文本（用于不含表的文本编辑距离）。"""
    return normalize_text(_TABLE_BLOCK_RE.sub(" ", doc or ""))


# ---------------------------------------------------------------------------
# 指标 1：文本编辑距离
# ---------------------------------------------------------------------------
def text_edit_loss(pred, gt, include_tables=True):
    """归一化字级编辑距离损失，∈[0,1]。
    include_tables=True：对整篇（含表格 HTML 源码）计算——最贴近官方"标准文本总长度"字面。
    include_tables=False：仅对去表后的正文计算——用于诊断正文质量。
    """
    if include_tables:
        p, g = normalize_text(pred), normalize_text(gt)
    else:
        p, g = _text_only(pred), _text_only(gt)
    if len(g) == 0:
        return 0.0 if len(p) == 0 else 1.0
    d = Levenshtein.distance(p, g)
    return min(1.0, d / len(g))


# ---------------------------------------------------------------------------
# 指标 2：表格 TEDS
# ---------------------------------------------------------------------------
def table_teds(pred, gt):
    """按出现顺序匹配 GT 与 pred 的表格，逐个算 TEDS 后求均值，∈[0,1]。
    GT 无表格 → 返回 None（该样本不计入表格分，例如 LONG 文档）。
    """
    gt_tables = _TABLE_BLOCK_RE.findall(gt or "")
    if not gt_tables:
        return None
    pred_tables = _TABLE_BLOCK_RE.findall(pred or "")
    scores = []
    for i, gt_t in enumerate(gt_tables):
        pred_t = pred_tables[i] if i < len(pred_tables) else ""
        s = teds_score(pred_t, gt_t)
        scores.append(s if s is not None else 0.0)
    return sum(scores) / len(scores) if scores else None


# ---------------------------------------------------------------------------
# 指标 3：阅读流顺序编辑距离（块级）
# ---------------------------------------------------------------------------
def _doc_blocks(doc):
    """把文档切成逻辑块序列（用于阅读流）：
    - 表格整体作为一个块，签名 'TBL#<出现序号>'（按位置区分,贴近官方"逻辑块序列编辑距离"）
    - 文本按空行分段，每段归一化后作为一个块（取前 80 字符做匹配键）

    表签名为何带序号:旧版所有表都用同一个 '__TABLE__'。文档里【多表与文字块交错】时,
    pred 的每个表都 fuzzy 匹配到 GT 的第一个 __TABLE__(同签名),matched 里反复出现同一
    index 把 LIS 打断 → ro 被严重低估(a4924b6d 真值92、旧版给48;2358594b 88→50)。改成
    'TBL#0/1/2…' 后,第 k 个表匹配第 k 个表(exact=100 > 邻号≈80,extractOne 选对),按位置
    对齐 → 贴近官方。单表文档(gt/pred 都只 TBL#0)行为不变。"""
    blocks = []
    nt = 0
    for b in split_blocks(doc):
        if b["type"] == "table":
            blocks.append("TBL#%d" % nt)
            nt += 1
        else:
            txt = normalize_text(b["raw"])
            for para in re.split(r"\n\s*\n", txt):
                para = " ".join(para.split())
                if para:
                    blocks.append(para[:80])
    return blocks


def _lnds(seq):
    """最长非降子序列长度（容忍分段差异导致的重复索引）。"""
    import bisect
    tails = []
    for x in seq:
        i = bisect.bisect_right(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def read_order_loss(pred, gt, match_thresh=75.0):
    """阅读流损失 ∈[0,1]，**与分段方式解耦**：
    1) 把 pred / gt 切成逻辑块；
    2) 每个 pred 块用模糊匹配找最相似的 gt 块索引（达阈值才算命中）；
       —— 用 token_set_ratio：对"标题被切成两段/合成一段"等分段差异鲁棒
          （只看词集合重叠），避免把"顺序正确但分段不同"误判为乱序；
    3) 命中的 gt 索引按 pred 顺序排列，取最长非降子序列 L（**顺序由 LIS 严格把关**）；
    4) read_score = L / len(gt块)，同时惩罚乱序与缺块；loss = 1 − score。
    """
    from rapidfuzz import process, fuzz
    gb = _doc_blocks(gt)
    pb = _doc_blocks(pred)
    if len(gb) == 0:
        return 0.0 if len(pb) == 0 else 1.0
    if len(pb) == 0:
        return 1.0
    matched = []
    for blk in pb:
        hit = process.extractOne(blk, gb, scorer=fuzz.token_set_ratio,
                                 score_cutoff=match_thresh)
        if hit is not None:
            matched.append(hit[2])               # gt 块索引
    if not matched:
        return 1.0
    L = _lnds(matched)
    return max(0.0, 1.0 - L / len(gb))


# ---------------------------------------------------------------------------
# 综合
# ---------------------------------------------------------------------------
def evaluate_one(pred, gt, text_include_tables=True):
    """返回单样本各项指标与综合分（满分 100）。"""
    te = text_edit_loss(pred, gt, include_tables=text_include_tables)
    teds = table_teds(pred, gt)              # None 表示无表格
    ro = read_order_loss(pred, gt)

    text_score = (1 - te) * 100
    read_score = (1 - ro) * 100
    teds_score_100 = (teds * 100) if teds is not None else None

    # 综合分：无表格样本（LONG）按官方三项均值；其 TEDS 项缺失时如何处理官方未明确。
    # 这里给两种口径：
    #   overall_3term  —— 严格三项平均（无表样本 TEDS 记 100，即"没有表格=表格满分"，偏乐观）
    #   overall_2term  —— 无表样本只平均文本与阅读流两项（更保守，推荐看这个做 LONG 调参）
    teds_for_3 = teds_score_100 if teds_score_100 is not None else 100.0
    overall_3term = (text_score + teds_for_3 + read_score) / 3
    if teds_score_100 is not None:
        overall_2term = overall_3term
    else:
        overall_2term = (text_score + read_score) / 2

    return {
        "text_edit": te,
        "teds": teds,
        "read_order": ro,
        "text_score": text_score,
        "teds_score": teds_score_100,
        "read_score": read_score,
        "overall_3term": overall_3term,
        "overall_2term": overall_2term,
    }


def evaluate_batch(pairs, text_include_tables=True):
    """pairs: List[(file_name, pred, gt)] → (per_sample_list, summary_dict)。"""
    rows = []
    for name, pred, gt in pairs:
        r = evaluate_one(pred, gt, text_include_tables)
        r["file_name"] = name
        rows.append(r)

    def _avg(key, subset=None):
        vals = [r[key] for r in rows if r[key] is not None
                and (subset is None or subset(r))]
        return sum(vals) / len(vals) if vals else None

    has_table = lambda r: r["teds"] is not None
    no_table = lambda r: r["teds"] is None
    summary = {
        "n": len(rows),
        "n_table": sum(1 for r in rows if has_table(r)),
        "n_long": sum(1 for r in rows if no_table(r)),
        "mean_text_score": _avg("text_score"),
        "mean_read_score": _avg("read_score"),
        "mean_teds_score": _avg("teds_score", has_table),
        "mean_overall_3term": _avg("overall_3term"),
        "mean_overall_2term": _avg("overall_2term"),
        # 分组诊断
        "long_text_score": _avg("text_score", no_table),
        "long_read_score": _avg("read_score", no_table),
        "table_text_score": _avg("text_score", has_table),
        "table_teds_score": _avg("teds_score", has_table),
    }
    return rows, summary


# ---------------------------------------------------------------------------
# 自检：用 GT 对 GT（应满分）+ 轻微扰动（应略降）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import glob
    import json

    ap = argparse.ArgumentParser(description="本地三指标评测")
    ap.add_argument("--pred_dir", help="预测 .md 目录（文件名与 GT 同名）")
    ap.add_argument("--gt_dir", help="GT .md 目录")
    ap.add_argument("--selftest", action="store_true",
                    help="用训练集 GT 做自检：GT vs GT 应满分，GT vs 扰动应略降")
    args = ap.parse_args()

    if args.selftest:
        from config import TRAIN_LONG_DIR, TRAIN_TABLE_DIR
        for tag, d in [("LONG", TRAIN_LONG_DIR), ("TABLE", TRAIN_TABLE_DIR)]:
            mds = sorted(glob.glob(os.path.join(d, "mds", "*.md")))[:5]
            pairs_same, pairs_pert = [], []
            for f in mds:
                gt = open(f, encoding="utf-8").read()
                pairs_same.append((os.path.basename(f), gt, gt))
                # 扰动：随机删 2% 字符
                import random
                random.seed(0)
                chars = list(gt)
                keep = [c for c in chars if random.random() > 0.02]
                pairs_pert.append((os.path.basename(f), "".join(keep), gt))
            _, s_same = evaluate_batch(pairs_same)
            _, s_pert = evaluate_batch(pairs_pert)
            print(f"\n===== {tag} (n={len(mds)}) =====")
            print("  GT vs GT     overall_3term =",
                  round(s_same["mean_overall_3term"], 3),
                  "| teds =", None if s_same["mean_teds_score"] is None
                  else round(s_same["mean_teds_score"], 3))
            print("  GT vs 扰动2%  overall_3term =",
                  round(s_pert["mean_overall_3term"], 3),
                  "| text =", round(s_pert["mean_text_score"], 3),
                  "| teds =", None if s_pert["mean_teds_score"] is None
                  else round(s_pert["mean_teds_score"], 3))
    elif args.pred_dir and args.gt_dir:
        pairs = []
        for f in sorted(glob.glob(os.path.join(args.gt_dir, "*.md"))):
            name = os.path.basename(f)
            gt = open(f, encoding="utf-8").read()
            pf = os.path.join(args.pred_dir, name)
            pred = open(pf, encoding="utf-8").read() if os.path.exists(pf) else ""
            pairs.append((name, pred, gt))
        rows, summary = evaluate_batch(pairs)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        ap.print_help()
