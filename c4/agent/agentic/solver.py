"""v4 Agentic 编排器（plan.md §v4）：decompose → 子问题 loop → synthesize。
不做确定性合并、不降级旧管线：把"原题 + 题型 + 各子问题结论(含证据/或无证据)"交给
synthesize LLM 直接给最终答案。子问题检索不到 → 标"无证据"，强行让 LLM 继续作答。
子问题检索质量后续单独优化。
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from .. import config
from ..index.bm25 import BM25Index
from ..llm.qwen import QwenClient
from ..postprocess.answer import normalize_answer, is_valid, parse_option_verdicts
from .decompose import Decomposer
from .loop import SubQLoop
from .route import route

_ARCH_HINT = {
    "value_compare": "各子问题给出了每个实体的数值。按选项里陈述的数值与排序，选与这些值最一致的选项。",
    "option_verdict": "已查到各选项所依赖的事实值。逐选项把事实与该选项主张比对：一致则该选项为真。多选题选入所有为真的选项（有正向证据就倾向选入，宁多勿漏）；判断题/单选选为真的那个。",
    "single_fact": "子问题给出了关键事实。选与该事实一致的选项。",
    "fallback": "依据子问题结论与原题直接判断。",
}

_SYN_SYS = """你根据若干子问题的检索结论，对原题给出最终答案。
- 充分利用每个子问题的结论与其证据片段。
- 有的子问题标注"未检索到证据"——不要因此空答，用已有的子结论 + 文档常识尽力判断。
- 最后单独一行输出：答案：<字母>（单选/判断一个字母；多选按字母序如 ABD）。"""


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
        self.synth_max_tokens = config.load().get("agentic", {}).get("synth_max_tokens", 600)
        self.fact_workers = config.load().get("agentic", {}).get("fact_workers", 6)

    def _run_one_fact(self, f: dict, doc_ids) -> dict:
        dids = route(self.index, f["ask"], list(doc_ids))
        r = self.loop.solve(f["ask"], dids)
        return {"ask": f["ask"], "doc": "/".join(dids),
                "value": r.value if r.found else "未查到",
                "found": r.found, "src": r.source_chunk_id, "evidence": r.source_text}

    def _run_facts(self, d: dict, doc_ids) -> list[dict]:
        """跑每条原子事实(全 compute=取值)。facts 互相独立 → 并行跑, 喂饱 GPU。"""
        facts = d["facts"]
        if self.fact_workers > 1 and len(facts) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.fact_workers) as ex:
                return list(ex.map(lambda f: self._run_one_fact(f, doc_ids), facts))
        return [self._run_one_fact(f, doc_ids) for f in facts]

    def _synthesize(self, q: dict, arch: str, facts: list[dict]) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        lines = []
        for s in facts:
            doc = f"[{s['doc']}] " if s.get("doc") else ""
            ev = f"  (依据: {s['evidence']})" if s.get("evidence") else ""
            lines.append(f"- {doc}{s['ask']} = {s['value']}{ev}")
        user = (f"【题目】{q['question']}\n\n【选项】\n{opts}\n\n"
                f"【已查到的事实】\n" + "\n".join(lines) +
                f"\n\n【题型】{_ARCH_HINT.get(arch, '')}\n"
                "请综合以上事实，对每个选项逐一比对判断，给出最终答案。")
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
