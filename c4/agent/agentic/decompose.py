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

SYS = """你是金融文档问答的"原子事实分解器"。把题目拆成若干【单篇文档 + 单个事实】的取值查询。只输出 JSON，不要解释。

铁律：
1. 每条 fact 只在【一篇】文档里取【一个】值/事实。doc_hint 填那一篇 doc_id；【无法确定是哪篇就填 null】（检索时在所有候选里找，比瞎猜一篇强）。
2. 问【值是什么】，不要问【是不是等于X】——比较/判真一律留到最后做：
   ✗ "发行人是否为广晟"   ✓ "发行人名称"
   ✗ "评级是否为AAA"      ✓ "主体信用评级"
3. 【跨文档比较 / 两份均为X】必须拆成每篇各一条：
   "第二份发行额低于第一份" → {"ask":"本期债券发行金额","doc_hint":"text01"} 和 {"ask":"本期债券发行金额","doc_hint":"text02"}
4. 算值排序题：每个实体一条，ask 把题干给的数字全写进去。
5. 不同选项依赖同一事实 → 合并成一条（去重）。

archetype（供最后推理用，不影响拆法）：value_compare | option_verdict | single_fact | fallback

每条 fact 字段：id；ask（单篇单值的取值查询）；doc_hint（单篇 doc_id，或 null）

只输出这个 JSON（不要 markdown 代码块）：
{"archetype":"...","facts":[{"id":"f1","ask":"...","doc_hint":"text01"}]}"""


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
        return {"archetype": "fallback", "facts": [], "raw": out[:300]}

    @staticmethod
    def _valid(obj: dict | None) -> bool:
        if not isinstance(obj, dict) or obj.get("archetype") not in ARCHETYPES:
            return False
        facts = obj.get("facts")
        if not isinstance(facts, list):
            return False
        return all(isinstance(s, dict) and "ask" in s for s in facts)

    def _norm(self, obj: dict, doc_ids: list[str]) -> dict:
        dset = set(map(str, doc_ids))
        for i, s in enumerate(obj["facts"]):
            s.setdefault("id", f"f{i+1}")
            dh = s.get("doc_hint")
            s["doc_hint"] = str(dh) if (dh is not None and str(dh) in dset) else None
        return obj
