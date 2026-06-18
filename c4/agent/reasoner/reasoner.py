"""推理器：证据 + 题目 -> 答案字母 + 推理文本（供 evidence.json）。"""
from __future__ import annotations
from dataclasses import dataclass
from ..llm.base import LLMClient
from ..retriever.retriever import Evidence
from ..postprocess.answer import normalize_answer, is_valid, parse_option_verdicts
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
        self.verify_multi = config.get("reasoner.verify_multi", False)

    def _verify_multi_answer(self, q: dict, ev_text: str, first_out: str, first_ans: str) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        msgs = [
            {"role": "system", "content":
             "你是金融长文档多选题复核员。只依据证据复核初答是否漏选或错选。"
             "必须重新判断 A-D 每个选项；证据支持核心主张就选入，只有与证据明确冲突才排除。"
             "最后单独一行输出：答案：<所有正确字母按字母序>。"},
            {"role": "user", "content":
             f"【证据】\n{ev_text}\n\n"
             f"【题目】\n{q['question']}\n\n【选项】\n{opts}\n\n"
             f"【初答】{first_ans}\n\n【初答推理】\n{first_out[-2500:]}\n\n"
             "请重点检查初答是否漏选。按固定格式复核：\n"
             "选项A：证据支持/冲突点 -> 真/假\n"
             "选项B：证据支持/冲突点 -> 真/假\n"
             "选项C：证据支持/冲突点 -> 真/假\n"
             "选项D：证据支持/冲突点 -> 真/假\n"
             "最后输出：答案：<字母>"}]
        out = self.llm.complete(msgs, max_tokens=1200, enable_thinking=False)
        return normalize_answer(out, "multi") or parse_option_verdicts(out, "multi")

    def answer(self, q: dict, evidences: list[Evidence]) -> Answer:
        ev_text = _format_evidence(evidences, self.max_ev_chars)
        msgs = prompts.build_messages(q, ev_text)
        out = self.llm.complete(msgs)
        ans = normalize_answer(out, q["answer_format"])
        fmt = {"mcq": "单个字母", "tf": "A或B", "multi": "所有正确字母按字母序如ABC"}
        # 兜底1：无"答案："行（常因截断）→ 解析逐项真假拼答案（零额外调用）
        if not is_valid(ans, q["answer_format"]):
            ans = parse_option_verdicts(out, q["answer_format"])
        # 兜底2：仍无效 → 强制再问一次只输出字母
        if not is_valid(ans, q["answer_format"]):
            fix = self.llm.complete(
                msgs + [{"role": "assistant", "content": out[-1500:]},
                        {"role": "user", "content":
                         f"基于以上分析直接下结论，只输出（{fmt.get(q['answer_format'])}）："
                         f"一行 答案：<字母>"}],
                max_tokens=30, enable_thinking=False)
            a2 = normalize_answer(fix, q["answer_format"]) or parse_option_verdicts(fix, q["answer_format"])
            if is_valid(a2, q["answer_format"]):
                ans = a2
        # 兜底3：绝不交空卷（空=必错；任意合法猜测期望≥空）
        if not is_valid(ans, q["answer_format"]):
            ans = "A"
        if (self.verify_multi and q["answer_format"] == "multi"
                and is_valid(ans, q["answer_format"])):
            a2 = self._verify_multi_answer(q, ev_text, out, ans)
            if is_valid(a2, q["answer_format"]):
                ans = a2
        return Answer(qid=q["qid"], answer=ans, reasoning=out,
                      evidence_doc_ids=sorted({e.doc_id for e in evidences}))
