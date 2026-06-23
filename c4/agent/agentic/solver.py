"""v4 Agentic 编排器（plan.md §v4）：decompose → 子问题 loop → synthesize。
不做确定性合并、不降级旧管线：把"原题 + 题型 + 各子问题结论(含证据/或无证据)"交给
synthesize LLM 直接给最终答案。子问题检索不到 → 标"无证据"，强行让 LLM 继续作答。
子问题检索质量后续单独优化。
"""
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from .. import config
from ..index.bm25 import BM25Index
from ..llm.qwen import QwenClient
from ..postprocess.answer import normalize_answer, is_valid, parse_option_verdicts
from .decompose import Decomposer
from .loop import SubQLoop
from .route import route

# "第N份文档/前者/后者"是给路由用的, 但会毒化 judge(它去块里找"第二份文档"找不到→判没证据)。
# 路由用完, 从 ask 里剥掉, judge 只看裸事实。注: 公司名/年份不剥(它们在原文里, 不毒化)。
_DOC_REF = re.compile(r"第[一二三四五六七八九十\d]+份(文档|报告)?之?的?|前者的?|后者的?|两份(文档|报告)?[中里的]*")

_ARCH_HINT = {
    "value_compare": "各子问题给出了每个实体的数值。按选项里陈述的数值与排序，选与这些值最一致的选项。",
    "option_verdict": "已查到各选项所依赖的事实值。逐选项把事实与该选项主张比对：一致则该选项为真。多选题选入所有为真的选项（有正向证据就倾向选入，宁多勿漏）；判断题/单选选为真的那个。",
    "single_fact": "子问题给出了关键事实。选与该事实一致的选项。",
    "fallback": "依据子问题结论与原题直接判断。",
}

_SYN_SYS = """你根据【检索到的原文证据】对原题作答（像人读资料一样，自己从原文里找答案）。
- 【以原文为准】：另给的"关键值线索"是自动抽取的、可能抽错或单位不一致，务必回原文核对（比大小先对齐口径单位）。
- 逐选项把主张与原文比对：原文支持才算真。多选选入所有为真的选项；判断/单选选为真的那个。
- 原文没明说的，用线索+常识尽力判断，别空答。
- 最后单独一行输出：答案：<字母>（多选按字母序如 ABD）。"""


@dataclass
class AgenticAnswer:
    qid: str
    answer: str
    path: str
    archetype: str
    sub: list[dict] = field(default_factory=list)


class AgenticSolver:
    def __init__(self):
        idx = BM25Index.from_file(os.path.join(config.path("index_dir"), "bm25.pkl"))
        outlines = json.load(open(os.path.join(config.path("index_dir"), "outlines.json"),
                                  encoding="utf-8"))
        self.index = idx
        self.llm = QwenClient()
        self.decomposer = Decomposer(self.llm, outlines)
        self.loop = SubQLoop(self.llm, idx)
        a = config.load().get("agentic", {})
        self.synth_max_tokens = a.get("synth_max_tokens", 600)
        self.fact_workers = a.get("fact_workers", 6)
        self.pool_max_chars = a.get("synth_pool_chars", 16000)   # B: 喂 synth 的原文证据池上限
        self._cid2text = {ch["chunk_id"]: ch["text"] for ch in idx.chunks}

    def _run_one_fact(self, f: dict, doc_ids) -> dict:
        dids = route(self.index, f["ask"], list(doc_ids))   # 路由用原 ask(含"第N份")
        ask = _DOC_REF.sub("", f["ask"]).strip()            # 剥 doc-ref 再喂 loop/judge
        r = self.loop.solve(ask, dids)
        # B: 带回这条子问题【检索过的所有块】, 供 synth 池化读原文(不只压缩值)
        chunk_ids = [cid for t in r.trace for cid in t.get("chunks", [])]
        return {"ask": f["ask"], "doc": "/".join(dids),
                "value": r.value if r.found else "未查到",
                "found": r.found, "src": r.source_chunk_id, "evidence": r.source_text,
                "chunks": chunk_ids}

    def _run_facts(self, d: dict, doc_ids) -> list[dict]:
        """跑每条原子事实(全 compute=取值)。facts 互相独立 → 并行跑, 喂饱 GPU。"""
        facts = d["facts"]
        if self.fact_workers > 1 and len(facts) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.fact_workers) as ex:
                return list(ex.map(lambda f: self._run_one_fact(f, doc_ids), facts))
        return [self._run_one_fact(f, doc_ids) for f in facts]

    def _evidence_pool(self, facts: list[dict]) -> str:
        """B: 所有子问题召回块的并集原文(去重、截断)。一个 reasoner 整体读, 替代 per-fact 压缩值。"""
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
                out.append(t)
                chars += len(t)
                if chars >= self.pool_max_chars:
                    return "\n---\n".join(out)
        return "\n---\n".join(out)

    def _synthesize(self, q: dict, arch: str, facts: list[dict]) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        hints = "\n".join(f"- {s['ask']} = {s['value']}" for s in facts)   # 抽取值(线索, 以原文为准)
        pool = self._evidence_pool(facts)                                  # 原文证据池
        user = (f"【题目】{q['question']}\n\n【选项】\n{opts}\n\n"
                f"【自动抽取的关键值(线索, 可能抽错/单位不一, 以原文为准)】\n{hints}\n\n"
                f"【检索到的原文证据】\n{pool}\n\n"
                f"【题型】{_ARCH_HINT.get(arch, '')}\n"
                "请【以原文证据为准】，对每个选项逐一核对判断，给出最终答案。")
        out = self.llm.complete([{"role": "system", "content": _SYN_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.synth_max_tokens, enable_thinking=False)
        fmt = q.get("answer_format", "mcq")
        ans = normalize_answer(out, fmt) or parse_option_verdicts(out, fmt)
        return ans if is_valid(ans, fmt) else "A"   # 绝不空答

    def answer(self, q: dict, doc_ids) -> AgenticAnswer:
        d = self.decomposer.decompose(q, doc_ids)
        facts = self._run_facts(d, doc_ids)
        ans = self._synthesize(q, d["archetype"], facts)
        return AgenticAnswer(q["qid"], ans, "synth", d["archetype"], facts)
