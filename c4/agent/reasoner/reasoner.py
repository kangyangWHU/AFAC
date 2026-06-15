"""推理器：证据 + 题目 -> 答案字母 + 推理文本（供 evidence.json）。"""
from __future__ import annotations
from dataclasses import dataclass
from ..llm.base import LLMClient
from ..retriever.retriever import Evidence
from ..postprocess.answer import normalize_answer, is_valid
from . import prompts
from .. import config


@dataclass
class Answer:
    qid: str
    answer: str
    reasoning: str
    evidence_doc_ids: list[str]


def _format_evidence(evs: list[Evidence], max_chars: int) -> str:
    parts, total = [], 0
    for e in evs:
        tag = e.article_no or (f"p{e.page}" if e.page else e.type)
        block = f"[{e.doc_id} | {tag}] {e.text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


class Reasoner:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.max_ev_chars = config.get("reasoner.max_evidence_chars", 12000)

    def answer(self, q: dict, evidences: list[Evidence]) -> Answer:
        ev_text = _format_evidence(evidences, self.max_ev_chars)
        msgs = prompts.build_messages(q, ev_text)
        out = self.llm.complete(msgs)
        ans = normalize_answer(out, q["answer_format"])
        if not is_valid(ans, q["answer_format"]):
            # 没给出显式"答案："行（常因 max_tokens 截断）→ 追问让它只下结论。
            # 把已有分析作为上下文，强制输出答案行，避免"全文扫字母"过度选择。
            fmt = {"mcq": "单个字母", "tf": "A或B", "multi": "所有正确字母按字母序如ABC"}
            fix = self.llm.complete(
                msgs + [{"role": "assistant", "content": out},
                        {"role": "user", "content":
                         f"基于以上分析，直接给出最终答案（{fmt.get(q['answer_format'])}），"
                         f"格式严格为一行：答案：<字母>"}],
                max_tokens=20)
            ans = normalize_answer(fix, q["answer_format"])
        return Answer(qid=q["qid"], answer=ans, reasoning=out,
                      evidence_doc_ids=sorted({e.doc_id for e in evidences}))
