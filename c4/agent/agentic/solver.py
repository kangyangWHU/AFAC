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

_ARCH_HINT = {
    "value_compare": "各子问题给出了每个实体的数值。按选项里陈述的数值与排序，选与这些值最一致的选项。",
    "option_verdict": "各子问题给出了每个选项主张的真/假。多选题：选入所有为真的选项（有正向证据就倾向选入，宁多勿漏）；判断题/单选：选为真的那个。",
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
        self.llm = QwenClient()
        self.decomposer = Decomposer(self.llm, outlines)
        self.loop = SubQLoop(self.llm, idx)
        self.synth_max_tokens = config.load().get("agentic", {}).get("synth_max_tokens", 600)

    def _run_subqs(self, d: dict, doc_ids) -> list[dict]:
        """跑每个子问题的 loop，收集结论 + 证据。value_compare 用 compute、其余用 verify。"""
        shape_default = "compute" if d["archetype"] in ("value_compare", "single_fact") else "verify"
        sub = []
        for s in d["sub_questions"]:
            shape = s.get("shape") or shape_default
            ent = str(s.get("entity") or "")
            narrow_ent = ent if d["archetype"] == "value_compare" else ""
            dids = [s["doc_hint"]] if s.get("doc_hint") else list(doc_ids)
            r = self.loop.solve(s["sq"], dids, entity=narrow_ent, shape=shape)
            if shape == "verify":
                concl = ("成立" if r.verdict else "不成立") if r.found else "未检索到证据"
            else:
                concl = r.value if r.found else "未检索到证据"
            sub.append({"sq": s["sq"], "entity": ent, "conclusion": concl,
                        "found": r.found, "src": r.source_chunk_id,
                        "evidence": r.source_text})
        return sub

    def _synthesize(self, q: dict, arch: str, sub: list[dict]) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        lines = []
        for i, s in enumerate(sub, 1):
            ev = f"\n   依据: {s['evidence']}" if s.get("evidence") else ""
            lines.append(f"{i}. {s['sq']}\n   结论: {s['conclusion']}{ev}")
        user = (f"【题目】{q['question']}\n\n【选项】\n{opts}\n\n"
                f"【题型】{_ARCH_HINT.get(arch, '')}\n\n"
                f"【子问题与检索结论】\n" + "\n".join(lines) +
                "\n\n请据此给出最终答案。")
        out = self.llm.complete([{"role": "system", "content": _SYN_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.synth_max_tokens, enable_thinking=False)
        fmt = q.get("answer_format", "mcq")
        ans = normalize_answer(out, fmt) or parse_option_verdicts(out, fmt)
        return ans if is_valid(ans, fmt) else "A"   # 绝不空答

    def answer(self, q: dict, doc_ids) -> AgenticAnswer:
        d = self.decomposer.decompose(q, doc_ids)
        sub = self._run_subqs(d, doc_ids)
        ans = self._synthesize(q, d["archetype"], sub)
        return AgenticAnswer(q["qid"], ans, "synth", d["archetype"], sub)
