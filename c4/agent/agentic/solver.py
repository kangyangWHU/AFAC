"""简化编排（无 judge / 无 verdict）：规则 decompose → 逐 fact 直接检索 → 合并全部 chunk
→ 一次领域 prompt 答题（legacy 式合并推理）。decompose 的价值=驱动逐选项/逐篇把证据召回；
judge/loop 仅留给 eval 测检索效果, 不进生产线。
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from .. import config
from ..index.bm25 import BM25Index
from ..llm.qwen import QwenClient
from ..postprocess.answer import normalize_answer, is_valid, parse_option_verdicts
from .decompose import Decomposer

# "第N份/前者/text0N"是定篇标签, 留在 ask 里污染 BM25 query → 检索前剥掉。
_DOC_REF = re.compile(r"第[一二三四五六七八九十\d]+份(文档|报告)?之?的?|前者的?|后者的?|两份(文档|报告)?[中里的]*"
                      r"|(?:文档|文件|在)?\s*[（(]?\s*(?:fc_)?text[ _]?0*\d+\s*[）)]?\s*[中里内]?之?的?")

# ---- 领域 prompt（复用 legacy reasoner 文案）----
_BASE = """你是金融长文档问答专家。严格依据【证据】作答，证据没有的不要用常识臆测。
先看清题干在问什么"条件/范围"。判断某选项是否入选，依据是【该选项是否满足题干所问的条件】，
不是【选项里那句描述本身是否属实】。若选项自带的说明已表明它不满足题干要求，
那么即使这句说明属实，该选项对本题也应判为不入选。
对每个选项，按固定格式判断，避免"事实算对但结论填反"：
  选项X：陈述了「<选项主张>」｜证据事实「<从证据得到的事实/数值>」｜两者是否一致 → 真/假
判定"是否一致"的规则：只有当选项与证据在【主体/年份/单位/数值/方向】上确有不符时才判假；
若这些都对，仅仅是选项表述更概括、或省略了不影响真假的限定词，**不算不符，应判真**。
多选题逐项独立判断；选项主张与证据事实一致就选，不要因"措辞不如证据具体"或"没找到反证"而漏掉本应为真的项。
**务必**在最后单独一行输出最终结论：答案：<字母>（这一行不能省略）。"""

_DOMAIN = {
    "regulatory": "这是监管法规题。必须依据法条原文判断，注意施行日期、时限（X日/X个月）、"
                  "义务主体、普通决议vs特别决议、阈值（比例/金额）。区分相近条款。",
    "insurance": "这是保险条款题。注意责任触发条件与计算公式（身故保险金、退保金/现金价值、"
                 "账户价值、已领年金）。涉及计算时逐产品列式计算再比较大小，不要跳步。",
    "financial_reports": "这是财报题。多为跨年/跨公司数值比较（营收、净利润、现金流、研发投入、"
                         "分红）。先从证据表格取准确数值（注意年份列对应），再判断增长/下滑/高低。"
                         "注意同义口径：研发投入≈研发费用、现金分红≈股息≈派息。",
    "financial_contracts": "这是债券/金融合同题。核对发行人、发行规模、信用评级、受托管理人、"
                           "违约责任、增信安排等要素，逐项与证据比对。",
    "research": "这是行业研报题。核验具体数字（市场规模、增速、份额、年份）。数字须在证据中有"
                "明确出处；趋势方向类可据图表描述判断。张冠李戴只指'数值对但主体/国家/年份/单位/方向被换掉'；"
                "选项若与证据的主体年份单位数值方向都一致、只是少写个限定词，应判真，不要疑心判假。",
}

_FMT_HINT = {
    "mcq": "单选题：只有一个正确选项，输出单个字母。",
    "tf": "判断题：根据选项含义输出 A 或 B。",
    "multi": "多选题：选出所有正确选项，按字母序输出（如 ABC）。漏选错选均算错，逐项严格判断。",
}
_FMT_ANS = {"mcq": "单个字母", "tf": "A或B", "multi": "所有正确字母按字母序如ABC"}


def _build_messages(q: dict, evidence: str) -> list[dict]:
    sys = _BASE + "\n" + _DOMAIN.get(q.get("domain", ""), "") + "\n" + _FMT_HINT.get(q["answer_format"], "")
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    user = (f"【证据】\n{evidence}\n\n【题目】\n{q['question']}\n\n【选项】\n{opts}\n\n"
            f"请逐项判断并在最后一行给出：答案：<字母>")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


@dataclass
class AgenticAnswer:
    qid: str
    answer: str
    path: str
    archetype: str
    sub: list[dict] = field(default_factory=list)
    verdicts: dict[str, dict] = field(default_factory=dict)
    selection_rule: dict[str, str] = field(default_factory=dict)


class AgenticSolver:
    def __init__(self):
        variant = config.load().get("agentic", {}).get("index_variant", "")  # ""=旧, "_vl"=新parse
        idx = BM25Index.from_file(os.path.join(config.path("index_dir"), f"bm25{variant}.pkl"))
        self.index = idx
        self.llm = QwenClient()
        self.decomposer = Decomposer(idx)
        a = config.load().get("agentic", {})
        self.synth_max_tokens = a.get("synth_max_tokens", 600)
        self.fact_workers = a.get("fact_workers", 6)
        self.pool_max_chars = a.get("synth_pool_chars", 18000)
        self.retrieve_k = a.get("retrieve_k", 8)            # 每 fact 每篇取多少 chunk
        self.final_thinking = config.get("llm.reasoning_effort") is not None
        self._cid2text = {ch["chunk_id"]: ch["text"] for ch in idx.chunks}
        self._cid2chunk = {ch["chunk_id"]: ch for ch in idx.chunks}
        self._ids = self.decomposer.ids                     # doc_id → 产品/主体名(给证据块打可读标签)

    def _backfill(self, cids: list[str]) -> list[str]:
        """命中块 + 前后各1邻块(small-to-big, 治答案跨块/'详见'指针), 按 seq 连贯, 去重。"""
        out: list[str] = []
        seen: set[str] = set()
        for cid in cids:
            ch = self._cid2chunk.get(cid)
            grp = [cid]
            if ch and ch.get("seq") is not None:
                try:
                    grp = sorted({cid} | {nb["chunk_id"] for nb in
                                 self.index.neighbors(ch["doc_id"], ch["seq"], 1)},
                                 key=lambda c: self._cid2chunk.get(c, {}).get("seq", 0))
                except Exception:
                    pass
            for c in grp:
                if c not in seen:
                    seen.add(c); out.append(c)
        return out

    def _gather_evidence(self, d: dict, doc_ids) -> list[dict]:
        """逐 fact 在其【每篇】各取 top-k chunk(保证 ≤2 篇都有代表) + 邻块回填(无 judge)。"""
        cand = [str(x) for x in doc_ids]

        def run(f: dict) -> dict:
            docs = [x for x in (f.get("doc") or cand) if x in cand] or cand
            ask = _DOC_REF.sub("", f["ask"]).strip() or f["ask"]
            cids: list[str] = []
            for dd in docs:                                  # 每篇各取, 不让一篇挤掉另一篇
                cids += [h.chunk_id for h in self.index.search_local(ask, [dd], k=self.retrieve_k)]
            f["chunks"] = self._backfill(cids)
            return f

        facts = d["facts"]
        if self.fact_workers > 1 and len(facts) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.fact_workers) as ex:
                return list(ex.map(run, facts))
        return [run(f) for f in facts]

    def _evidence_pool(self, facts: list[dict]) -> str:
        """合并全部 fact 的 chunk → 去重 → 带 [id|doc|breadcrumb] 标签 → 截到 pool 上限。"""
        seen: set[str] = set()
        out, chars = [], 0
        for s in facts:
            for cid in s.get("chunks", []):
                if cid in seen:
                    continue
                seen.add(cid)
                t = self._cid2text.get(cid)
                if not t:
                    continue
                ch = self._cid2chunk.get(cid, {})
                did = str(ch.get("doc_id"))
                name = (self._ids.get(did) or did)[:40]      # 用产品/主体名而非不透明 doc_id, 让模型把证据归到对应选项
                bc = ch.get("breadcrumb") or ch.get("article_no") or ch.get("type") or ""
                block = f"[{name} | {bc}] {t}"
                out.append(block)
                chars += len(block)
                if chars >= self.pool_max_chars:
                    return "\n\n".join(out)
        return "\n\n".join(out)

    def _answer_llm(self, q: dict, facts: list[dict]) -> str:
        fmt = q.get("answer_format", "mcq")
        pool = self._evidence_pool(facts)
        msgs = _build_messages(q, pool)
        out = self.llm.complete(msgs, max_tokens=self.synth_max_tokens,
                                enable_thinking=self.final_thinking)
        ans = normalize_answer(out, fmt) or parse_option_verdicts(out, fmt)
        if not is_valid(ans, fmt):                          # 没"答案："行→强制再问一次只输出字母
            out2 = self.llm.complete(
                msgs + [{"role": "assistant", "content": out[-1500:]},
                        {"role": "user", "content":
                         f"基于以上分析直接下结论，只输出一行 答案：<{_FMT_ANS.get(fmt)}>"}],
                max_tokens=30, enable_thinking=False)
            a2 = normalize_answer(out2, fmt) or parse_option_verdicts(out2, fmt)
            if is_valid(a2, fmt):
                ans = a2
        return ans if is_valid(ans, fmt) else "A"           # 绝不交空卷

    def answer(self, q: dict, doc_ids) -> AgenticAnswer:
        d = self.decomposer.decompose(q, doc_ids)
        facts = self._gather_evidence(d, doc_ids)
        ans = self._answer_llm(q, facts)
        return AgenticAnswer(q["qid"], ans, "merge", d["archetype"], facts)
