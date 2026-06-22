"""问题分解器（plan.md §v4 阶段2）：题目 -> 子问题 + 题型。只输出 JSON，离线可评。
不碰旧管线；供 v4 agentic loop 调用。设计要点（与用户对齐）：
- 子问题"自带答案"：value_compare 的子问题把题干数字写进去，块只需提供规则，LLM 直接吐值。
- option_verdict 每个选项一句判真；合并时收真+add-only 偏置。
- doc_hint 可空(null=该子问题在全部候选文档里搜)。
"""
from __future__ import annotations
import json
import os
import re
import threading
from ..llm.base import LLMClient
from .. import config

ARCHETYPES = {"value_compare", "option_verdict", "single_fact", "fallback"}

SYS = """你是金融文档问答的"原子事实分解器"。把题目拆成若干【单篇文档 + 单个事实】的取值查询。只输出 JSON，不要解释。

铁律：
1. 每条 fact 只问【一个】值/事实（来自单篇文档；到底哪一篇由后续路由决定，你不用指定）。
2. 问【值是什么】，不要问【是不是等于X】——比较/判真一律留到最后做：
   ✗ "发行人是否为广晟"   ✓ "发行人名称"
   ✗ "评级是否为AAA"      ✓ "主体信用评级"
3. 【跨文档比较 / 两份均为X】必须拆成每篇各一条（按文档拆，但不用写文档名）：
   "第二份发行额低于第一份" → {"ask":"第一份文档的本期债券发行金额"} 和 {"ask":"第二份文档的本期债券发行金额"}
4. 算值排序题：每个实体一条，ask 把题干给的数字全写进去。
5. 【案例计算题】题干给的情形数据（某人、给定的金额/费用、发生的事件）是【计算输入】，文档里根本没有——【不要】为它建 fact：
   ✗ "王某本人的医疗费用总额"  ✗ "李某住院花了多少"（这些题干已给，查文档查不到）
   ✓ 只为文档里的【规则】建 fact，把情形数据写进该 fact 的 ask 供后续代入算：
     {"ask":"太保团体百万医疗对王某8万住院费(医保已报3万)的赔付金额(免赔额/给付比例规则)"}
6. 不同选项依赖同一事实 → 合并成一条（去重）。

archetype（供最后推理用，不影响拆法）：value_compare | option_verdict | single_fact | fallback

facts 不能为空：每道题都要拆出能定位答案的事实。多选/判断/计算/取数题【一律】要拆（每个选项/实体所依赖的事实都列出来）。fallback 只用于题目完全无法在文档里定位任何事实的极端情况（极罕见），别因为"要结合多篇/需要推理"就用 fallback——结合和推理是【最后一步】做的，分解阶段只管把事实查出来。

每条 fact 字段：id；ask（单个值/事实的取值查询）

只输出这个 JSON（不要 markdown 代码块）：
{"archetype":"...","facts":[{"id":"f1","ask":"..."}]}"""


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
        # 子问题缓存：同(题,候选文档)复用上次分解 → 跨运行固定子问题, 让"只有retrieve在变"可净测+省token。
        # 改 decompose prompt 时手动删 index/decompose_cache.json。
        self.cache_on = a.get("decompose_cache", True)
        self._cpath = os.path.join(config.path("index_dir"), "decompose_cache.json")
        self._clock = threading.Lock()
        self._cache = (json.load(open(self._cpath, encoding="utf-8"))
                       if self.cache_on and os.path.exists(self._cpath) else {})

    def build_messages(self, q: dict, doc_ids: list[str]) -> list[dict]:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        user = (f"【候选文档】\n{_fmt_docs(self.outlines, doc_ids)}\n\n"
                f"【题目】{q['question']}\n\n【选项】\n{opts}\n\n"
                f"【答案格式】{q.get('answer_format')}\n\n请输出分解 JSON。")
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": user}]

    def _facts_from_options(self, q: dict) -> list[dict]:
        """兜底拆解：LLM 给空/fallback 时, 每个选项(或题干)生成一条 fact, doc 全搜。"""
        opts = q.get("options", {})
        if q.get("answer_format") == "tf" or len(opts) < 2:
            return [{"id": "f1", "ask": q["question"]}]
        return [{"id": f"f{i+1}", "ask": v}
                for i, v in enumerate(opts.values()) if len(str(v)) > 2]

    def decompose(self, q: dict, doc_ids: list[str]) -> dict:
        key = f"{q.get('qid')}||{'/'.join(map(str, doc_ids))}"
        if self.cache_on and key in self._cache:
            return self._cache[key]
        obj = self._decompose_uncached(q, doc_ids)
        if self.cache_on:
            with self._clock:
                self._cache[key] = obj
                json.dump(self._cache, open(self._cpath, "w", encoding="utf-8"), ensure_ascii=False)
        return obj

    def _decompose_uncached(self, q: dict, doc_ids: list[str]) -> dict:
        msgs = self.build_messages(q, doc_ids)
        out = ""
        for _ in range(self.retries + 1):
            out = self.llm.complete(msgs, max_tokens=self.max_tokens, enable_thinking=False)
            obj = _parse_json(out)
            if obj is not None and self._valid(obj):
                obj = self._norm(obj, doc_ids)
                if not obj["facts"]:                    # LLM 给了 fallback/空 → 用选项兜底
                    obj["facts"] = self._facts_from_options(q)
                return obj
            msgs = msgs + [{"role": "assistant", "content": out[:600]},
                           {"role": "user", "content": "上面不是合法 JSON。只输出符合规定格式的 JSON。"}]
        return {"archetype": "option_verdict", "facts": self._facts_from_options(q),
                "raw": out[:300]}

    @staticmethod
    def _valid(obj: dict | None) -> bool:
        if not isinstance(obj, dict) or obj.get("archetype") not in ARCHETYPES:
            return False
        facts = obj.get("facts")
        if not isinstance(facts, list):
            return False
        return all(isinstance(s, dict) and "ask" in s for s in facts)

    def _norm(self, obj: dict, doc_ids: list[str]) -> dict:
        for i, s in enumerate(obj["facts"]):
            s.setdefault("id", f"f{i+1}")
        return obj                                   # 绑哪篇文档交给 route.py, 分解器不管
