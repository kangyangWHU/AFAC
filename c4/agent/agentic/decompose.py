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
1. 【一条 fact 一个值，ask 带全身份】ask 信息尽量全、把"取哪一处"钉死：主体/公司/产品全称 + 年份/报告期 + 口径(合并vs母公司、全年vs年末) + 场景。同一指标常有多版(多年/多主体/多表)，缺限定就取错。选项给一组值(如"5.70%和35.83%") → ask 取整组(各报告期列全)，别压成单个。
2. 【定篇：只在题干【自己点名了某篇】时才写 doc，否则一律留空——你只拿到候选 id 和顺序、【没有标题】，【绝不按内容/公司/产品名猜 doc】】
   写 doc【仅】两种情形：
   ① 题干用序数(第一份/第二份/前者/后者) → doc=候选里【对应顺序】那篇 id(第一份=第1篇)。
   ② 题干用文档名/原样 id(如"文档fc_text_006""text06") → doc=该 id(fc_text_006 会自动归一到候选 text06)。
   其余【全部留空 doc】——尤其"哪些产品/公司…"这种【选项点名实体】的题：实体→篇由 route/idrouter 按内容定(它有准确身份库、比你猜准)，你只需把【主体全名】写进 ask。
      ✓ 留空+带主体:"平安e生保的施救费用赔偿上限是?"   ✓ 题干点名篇:{"ask":"是否有转股价格向下修正条款","doc":"text06"}   ✗ 题干没点名却按产品名猜 doc
3. 【问值，不问是否=X；问值别用表头泛词】比较/判真留到最后。"发行人是谁""主体信用评级"这类【要找的就是身份本身】、身份未知写不进 ask 的，必须靠【篇】定位(题干点名就按铁律2②写 doc；没点名则该 fact 天然属某篇)。问值时直奔答案、别加"名称/类型/情况"这类表头泛词(会把"子公司名称/债务人名称"等无关表顶上来)：
   ✗"发行人是否为「某公司」"(别问是否) ✗"发行人名称"(表头泛词) ✓{"ask":"发行人是?","doc":"text01"}　✓{"ask":"主体信用评级是?","doc":"text01"}
4. 【跨多篇主张拆开】选项横跨多篇(均/都/两份/分别/各/不一致) → 按候选篇数拆成每篇各一条(绝不合成,否则只检到一篇)，每条按铁律2定篇、ask 各带自己主体的全名。
5. 【案例计算题别查算出来的数】题干情形(某人/金额/事件)和选项算出的合计，文档里都没有 → 别建 fact；改为题干涉及的【每个产品】各建一条赔付规则 fact(把情形写进 ask、留给 verdict 代入算)：ask 形如「某产品对[题干费用/免赔额/事件]的赔付规则(免赔额/给付比例/范围/分摊)」。
6. 不同选项依赖同一事实 → 合并去重。

archetype（供最后推理用，不影响拆法）：value_compare | option_verdict | single_fact | fallback

facts 不能为空：每道题都要拆出能定位答案的事实。多选/判断/计算/取数题【一律】要拆（每个选项/实体所依赖的事实都列出来）。fallback 只用于题目完全无法在文档里定位任何事实的极端情况（极罕见），别因为"要结合多篇/需要推理"就用 fallback——结合和推理是【最后一步】做的，分解阶段只管把事实查出来。

每条 fact 字段：id；ask；doc（可选，仅【铁律2②题干点名了文档】时填候选原样id，否则省略、由 route 定篇）

只输出这个 JSON（不要 markdown 代码块）：
{"archetype":"...","facts":[{"id":"f1","ask":"...","doc":"候选原样id或省略"}]}"""


def _fmt_docs(doc_ids: list[str]) -> str:
    # 只给候选 id + 顺序(供"第N份"映射)，【不给标题】——outlines.name 多为公司/文档类型【泛名(多篇撞名)】或缺失，
    # 给了只会诱导模型【按内容猜 doc】并猜错；内容→篇交 route/idrouter(用准确的 doc_identities)。
    return "\n".join(f"  第{i + 1}篇: {d}" for i, d in enumerate(map(str, doc_ids)))


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


def _match_doc(ref, cand_set: set[str]) -> str | None:
    """铁律2②: 把题干式文档名归一到候选原样id。先精确; 否则按【末尾数字组】唯一匹配
    (fc_text_006→text06)；歧义(如多篇 ..._att1 末尾同号)则不匹配, 留给 route。"""
    if ref is None:
        return None
    ref = str(ref).strip()
    if ref in cand_set:
        return ref
    nums = re.findall(r"\d+", ref)
    if not nums:
        return None
    rn = int(nums[-1])
    hits = [c for c in cand_set if (m := re.findall(r"\d+", c)) and int(m[-1]) == rn]
    return hits[0] if len(hits) == 1 else None


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
        user = (f"【候选文档(只列 id 和顺序, 无标题)】\n{_fmt_docs(doc_ids)}\n\n"
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
            s["doc"] = _match_doc(s.get("doc"), cand_set)  # 铁律2②: 题干名→候选id(fc_text_006→text06); 配不上→None→route
        return obj
