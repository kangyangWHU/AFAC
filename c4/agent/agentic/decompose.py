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
1. 每条 fact 只问【一个】值/事实，且必须能【唯一定位到文档里的那一处】。同一指标文档常有多处值（多年份/报告期、多主体/公司、多口径如合并vs母公司、多张表）——ask 必须带区分限定（年份/日期/报告期、主体、口径、所属表），把那一处钉死；缺限定就会取到别年/别表的值。（该 fact 已能确定属于某一篇候选文档时，写上 doc=该篇原样id；不确定就留空，由后续路由按内容定篇。）
   ✗ "资产负债率数值"(多张表都有)  ✓ "截至2020年3月31日的资产负债率"
   选项若列举一组值(如"5.70%和35.83%") → ask 取【整行/整组(各报告期列全)】，别压成单个数。
2. 问【值是什么】，不要问【是不是等于X】——比较/判真一律留到最后做：
   ✗ "发行人是否为广晟"   ✓ "发行人名称"
   ✗ "评级是否为AAA"      ✓ "主体信用评级"
3. 【跨文档 claim：均/都/两份/分别/各/不一致】凡选项主张【横跨多篇候选文档】（"两份均…/都…/分别…/各自…/不一致"），【必须】按候选文档数拆成每篇各一条，每篇证据各检一份；绝不能合成一条——合成只会路由到一篇、另一篇证据缺失，就判不了"均/不一致"。
   每条拆出的 fact【必须带 doc 字段】＝该篇在【候选文档】里冒号前的【原样 id】（逐字复制，如 text04 / 9 / annual_cscec_2025_report），把这条钉死到那一篇；别把"第一份/textXX中"写进 ask 靠后续猜。
   ✗ 一条"两份均设有条件赎回条款"
   ✓ {"ask":"有条件赎回条款的设置情况","doc":"<候选里第1篇的原样id>"} + {"ask":"有条件赎回条款的设置情况","doc":"<候选里第2篇的原样id>"}
   数值比较同理：{"ask":"本期债券发行金额","doc":"<第1篇id>"} + {"ask":"本期债券发行金额","doc":"<第2篇id>"}
4. 算值排序题：每个实体一条，ask 把题干给的数字全写进去。
5. 【案例计算题】题干给的情形数据（某人、给定的金额/费用、发生的事件）是【计算输入】，文档里根本没有；选项里【算出来的金额/合计】文档里也查不到。两者都【不要】建成 fact：
   ✗ "某人住院花了多少"  ✗ "X赔a元…合计c元"（选项算出来的数，查文档查不到）
   ✓ 识别标志：选项是"各产品分别赔多少、合计多少"这类算出来的结果 = 案例计算题。则为【题干涉及的每个产品/条款】各建一条"赔付规则"fact，把题干费用情形写进 ask，让 verdict 代入规则算、再与各选项比对：
     {"ask":"<题干涉及的某产品>对[题干给的费用/免赔额/事件情形]的赔付规则(免赔额/给付比例/赔付范围/分摊方式)"}
6. 不同选项依赖同一事实 → 合并成一条（去重）。

archetype（供最后推理用，不影响拆法）：value_compare | option_verdict | single_fact | fallback

facts 不能为空：每道题都要拆出能定位答案的事实。多选/判断/计算/取数题【一律】要拆（每个选项/实体所依赖的事实都列出来）。fallback 只用于题目完全无法在文档里定位任何事实的极端情况（极罕见），别因为"要结合多篇/需要推理"就用 fallback——结合和推理是【最后一步】做的，分解阶段只管把事实查出来。

每条 fact 字段：id；ask（单个值/事实的取值查询）；doc（可选＝该 fact 所属候选文档的原样id；能确定属于某篇时填、跨文档比较时必填，不确定留空由路由定）

只输出这个 JSON（不要 markdown 代码块）：
{"archetype":"...","facts":[{"id":"f1","ask":"...","doc":"候选原样id或省略"}]}"""


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
            return [{"id": "f1", "option_id": "shared", "ask": q["question"]}]
        return [{"id": f"f{i+1}", "option_id": k, "ask": v}
                for i, (k, v) in enumerate(opts.items()) if len(str(v)) > 2]

    def decompose(self, q: dict, doc_ids: list[str]) -> dict:
        key = f"{q.get('qid')}||{'/'.join(map(str, doc_ids))}"
        if self.cache_on and key in self._cache:
            return self._norm(json.loads(json.dumps(self._cache[key])), q, doc_ids)
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
                obj = self._norm(obj, q, doc_ids)
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

    def _norm(self, obj: dict, q: dict, doc_ids: list[str]) -> dict:
        opts = list(k for k in q.get("options", {}) if k in "ABCD")
        valid = set(opts) | {"shared"}
        cand_set = set(map(str, doc_ids))
        facts = obj.get("facts") or []
        infer_by_order = (obj.get("archetype") == "option_verdict"
                          and q.get("answer_format") == "multi"
                          and len(facts) == len(opts))
        for i, s in enumerate(facts):
            s.setdefault("id", f"f{i+1}")
            oid = s.get("option_id")
            if isinstance(oid, list):
                oid = oid[0] if oid else None
            oid = str(oid).upper() if oid is not None else ""
            if oid not in valid:
                oid = opts[i] if infer_by_order and i < len(opts) else "shared"
            s["option_id"] = oid
            d = s.get("doc")                          # 仅保留【精确命中候选id】的篇绑定; 否则 None → 丢给 route 按内容定篇
            s["doc"] = str(d).strip() if (d is not None and str(d).strip() in cand_set) else None
        return obj
