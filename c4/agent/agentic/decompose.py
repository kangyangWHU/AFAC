"""问题分解器（plan.md §v4 阶段2）：题目 -> 子问题 + 题型。只输出 JSON，离线可评。
不碰旧管线；供 v4 agentic loop 调用。设计要点（与用户对齐）：
- 子问题"自带答案"：value_compare 的子问题把题干数字写进去，块只需提供规则，LLM 直接吐值。
- option_verdict 每个选项一句判真；合并时收真+add-only 偏置。
- doc_hint 可空(null=该子问题在全部候选文档里搜)。
"""
from __future__ import annotations
import json
import re
from ..llm.base import LLMClient
from .. import config

ARCHETYPES = {"value_compare", "option_verdict", "single_fact", "fallback"}

SYS = """你是金融长文档问答的"问题分解器"。把题目拆成若干可独立检索作答的子问题。只输出 JSON，不要任何解释。

archetype（四选一）：
- value_compare：选项在比较/排序多个实体的某指标数值（如各保险产品身故金排序）。按【实体】拆；每个子问题必须把题干给出的全部相关数字写进 sq，答案是一个具体数值。shape=compute。
- option_verdict：多选题，每个选项是一句可独立判真假的主张（"下列哪些正确"）。按【选项 A/B/C/D】各拆一个；每个子问题问"该主张是否成立"。shape=verify。
- single_fact：取一个事实即可定答案。1 个子问题。shape=compute。
- fallback：不宜拆解，整体作答。sub_questions 置为空数组。

每个子问题字段：
- id："sq1","sq2"...
- sq：自包含的子问题文本。value_compare 要把题干数字写进去；option_verdict 直接复述该选项主张并问是否成立。
- shape："compute" 或 "verify"
- entity：实体名 或 选项字母（A/B/C/D），便于合并对齐
- doc_hint：该子问题最该查的 doc_id（只能从候选文档里选）；不确定就填 null

只输出这个 JSON（不要 markdown 代码块）：
{"archetype":"...","sub_questions":[{"id":"sq1","sq":"...","shape":"...","entity":"...","doc_hint":"2"}]}"""


def _fmt_docs(outlines: dict, doc_ids: list[str]) -> str:
    lines = []
    for d in map(str, doc_ids):
        o = outlines.get(d, {})
        nm = o.get("name") or "(无标题)"
        lines.append(f"  {d}: {nm}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict | None:
    """从模型输出里抠出第一个 JSON 对象（容忍 ```json 包裹/前后废话）。"""
    t = re.sub(r"```(?:json)?|```", "", text)
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class Decomposer:
    def __init__(self, llm: LLMClient, outlines: dict):
        self.llm = llm
        self.outlines = outlines
        a = config.load().get("agentic", {})
        self.max_tokens = a.get("decompose_max_tokens", 1000)
        self.retries = a.get("decompose_retries", 1)

    def build_messages(self, q: dict, doc_ids: list[str]) -> list[dict]:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        user = (f"【候选文档】\n{_fmt_docs(self.outlines, doc_ids)}\n\n"
                f"【题目】{q['question']}\n\n【选项】\n{opts}\n\n"
                f"【答案格式】{q.get('answer_format')}\n\n请输出分解 JSON。")
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": user}]

    def decompose(self, q: dict, doc_ids: list[str]) -> dict:
        msgs = self.build_messages(q, doc_ids)
        out = ""
        for _ in range(self.retries + 1):
            out = self.llm.complete(msgs, max_tokens=self.max_tokens, enable_thinking=False)
            obj = _parse_json(out)
            if obj is not None and self._valid(obj):
                return self._norm(obj, doc_ids)
            msgs = msgs + [{"role": "assistant", "content": out[:600]},
                           {"role": "user", "content": "上面不是合法 JSON。只输出符合规定格式的 JSON。"}]
        return {"archetype": "fallback", "sub_questions": [], "raw": out[:300]}

    @staticmethod
    def _valid(obj: dict | None) -> bool:
        if not isinstance(obj, dict) or obj.get("archetype") not in ARCHETYPES:
            return False
        sqs = obj.get("sub_questions")
        if not isinstance(sqs, list):
            return False
        return all(isinstance(s, dict) and "sq" in s for s in sqs)

    def _norm(self, obj: dict, doc_ids: list[str]) -> dict:
        dset = set(map(str, doc_ids))
        for i, s in enumerate(obj["sub_questions"]):
            s.setdefault("id", f"sq{i+1}")
            dh = s.get("doc_hint")
            s["doc_hint"] = str(dh) if (dh is not None and str(dh) in dset) else None
            s["shape"] = s.get("shape") or ("verify" if obj["archetype"] == "option_verdict" else "compute")
        return obj
