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
from ..chunker.base import normalize
from .. import config

ARCHETYPES = {"value_compare", "option_verdict", "single_fact", "fallback"}

SYS = """你是金融文档问答的"原子事实分解器"。

【先想清楚你的产物拿来干嘛——想通了怎么拆自然就对，别背规则】：
系统会把你产出的每条 ask 拿去【某一篇文档】里检索一小段、读出【一个值】；最后另一步用这些值跟选项逐一比对、判对错。
所以你的任务 = 把题目变成【验证各选项所需的、一条条能被检索器找到的"取值查询"】——不是复述题目、不是下判断。

由此推出怎么拆(每条都讲"为什么"，按理解判断；遇到下面没列举到的写法，照同样的道理办)：
1. 一条 ask 只问【一个值】，带全限定(主体 + 年份/报告期 + 口径如合并vs母公司 + 场景)把"读哪一处"钉死。
   为什么：检索器一次只读一小段、抽一个值；同一指标多篇/多年/多表都有，限定不全就读到别处。
2. 问【值是多少】，别抄选项原话、别把待验证的那个数写进 ask；也别用"名称/类型/情况"这类表头泛词。
   为什么：检索靠你的词去匹配【文档原文里的指标】(原文写"初始转股价19.59元")——文档里没有你的判断词("明确记载了")，那个数(19.59)正是要去找的、对错最后才判；"名称/类型"是大量表头词、会把无关表顶上来。
   ✗"明确记载初始转股价为19.59元" / ✗"发行人名称" → ✓"初始转股价是多少?" / ✓"发行人是?"
3. 每条都要让检索器知道【去哪一篇读】，二选一：
   - 题干自己点了篇(序数"第一份/前者"、或文档名如"fc_text_003"、或某标题) → 把题干用的【那个名字原样写进 doc 字段】(代码会自动对到候选篇，你不用管它写成 fc_text_003 还是 text03)。
   - 没点篇 → 把【主体全名】(公司/产品/发行人)写进 ask，让检索器按内容定篇。
   为什么：同一指标很多篇都有，不说是谁的/哪篇就落错篇。注意你只看得到候选的【id 和顺序、没有标题】，所以【别自己按内容猜哪篇是哪篇】(猜常错)——除非题干点名，一律靠主体名交给检索器(它有准确的文档身份库)。
4. 只要【一句话指向不止一篇候选】(像"两份均…""其中一份…"这种，任何跨多篇的表述都算)，→ 按候选篇数拆成【每篇各一条】，每条按第3条定篇。
   为什么：要核"都有X""其中一份的X=19.59"，必须【对每篇分别去查 X】；合成一条只会查到一篇、漏掉另一篇。
5. 个案情景值【绝不建检索 fact】：题面给的个案数(某人花了8000、住院5天、本人/配偶各自的费用、三家合计13.5万)是【题面已知、文档里查不到】——别把这个已知值重新问一遍(✗"王某本人医疗费用是多少"、✗"李某自费金额是多少")。改为只给【每个涉及的产品/条款】建"赔付规则/免赔额/比例/给付条件"这类 ask(这些文档才查得到)，个案数留到最后一步代入算。
   为什么：检索器只在【文档】里找；个案人物/数字文档根本没有，为它建 fact 必然查空、白费一轮，还可能让 judge 误判。
6. 不同选项依赖【同一个值】→ 合并成一条。

archetype（供最后推理用，不影响拆法）：value_compare | option_verdict | single_fact | fallback

facts 不能为空：每道题都要拆出能定位答案的事实。多选/判断/计算/取数题【一律】要拆（每个选项/实体所依赖的事实都列出来）。fallback 只用于题目完全无法在文档里定位任何事实的极端情况（极罕见），别因为"要结合多篇/需要推理"就用 fallback——结合和推理是【最后一步】做的，分解阶段只管把事实查出来。

每条 fact 字段：id；ask；doc（可选，仅【铁律2②题干点名了文档】时填候选原样id；否则省略，代码自动置为【整组】、由 solver 全组齐检）

只输出这个 JSON（不要 markdown 代码块）：
{"archetype":"...","facts":[{"id":"f1","ask":"...","doc":"候选原样id或省略"}]}"""


def _fmt_docs(doc_ids: list[str]) -> str:
    # 给候选 id + 顺序(供"第N份"映射)，【不给标题】(标题/泛名会诱导按内容猜 doc)。
    # 即便给了 id, 模型仍可能【按公司/年/名字猜哪篇】——这类【无题面显式依据】的猜在 _norm 里被丢弃,
    # 只认 ① ask 里位置词(第N份/前者) ② 题干/选项点了的文档名; 其余按内容→篇交 solver 路由。
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


def _doc_in_ask(ask, cand_set: set[str]) -> str | None:
    """code 兜底: ask 文本里若出现【候选篇的名字】(候选原样id, 或 "fc_text_002" 这类题干式写法)→自动 pin。
    治"模型把文档名写进了 ask 却没填 doc 字段"。只认够独特的引用, 不碰裸数字(防误绑年份)。"""
    if not ask:
        return None
    for c in cand_set:                                            # a) 候选原样id(≥4字符才够独特)作子串出现
        if len(c) >= 4 and c in ask:
            return c
    for m in re.findall(r"(?:fc_)?text[_ ]?0*\d+", ask, re.I):    # b) fc_text_002 / text02 这类引用→归一到候选
        d = _match_doc(m, cand_set)
        if d:
            return d
    return None


# 题干【显式位置词】：只有 ask 真带这些, 模型按位置指派某篇才算数(治"按年份/选项序猜位置")。
_POS_RE = re.compile(r"第[一二三四五六七八九十两\d]+[份篇章]|前者|后者|前一[份篇]|后一[份篇]")


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
        obj = None
        for _ in range(self.retries + 1):
            out = self.llm.complete(msgs, max_tokens=self.max_tokens, enable_thinking=False)
            cand = _parse_json(out)
            if cand is not None and self._valid(cand):
                obj = cand
                break
            msgs = msgs + [{"role": "assistant", "content": out[:600]},
                           {"role": "user", "content": "上面不是合法 JSON。只输出符合规定格式的 JSON。"}]
        if obj is None:                                  # 始终没拿到合法 JSON
            obj = {"archetype": "option_verdict", "facts": [], "raw": out[:300]}
        obj = self._norm(obj, q, doc_ids)
        if not obj["facts"]:                             # 空/fallback → 选项兜底, 并【同样过 _norm 以 pin 文档名】
            obj["facts"] = self._facts_from_options(q)
            obj = self._norm(obj, q, doc_ids)
        return obj

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
            s["ask"] = normalize(s.get("ask", ""))    # 去 PDF/模型伪空格(2024 年→2024年), 对齐已归一的块/身份
            oid = s.get("option_id")
            if isinstance(oid, list):
                oid = oid[0] if oid else None
            oid = str(oid).upper() if oid is not None else ""
            if oid not in valid:
                oid = opts[i] if infer_by_order and i < len(opts) else "shared"
            s["option_id"] = oid
            ask = s.get("ask", "")
            opt = q.get("options", {}).get(oid, "")
            pin = _match_doc(s.get("doc"), cand_set) or _doc_in_ask(ask, cand_set)
            # 只保留【题面显式信号】支撑的 pin: ask 带位置词(第N份/前者/后者), 或 题干/选项点了这篇的文档名;
            # 否则是模型【按公司/年/名字猜的篇】(无显式依据)→ 丢弃 → [整组], 交 solver 按内容路由(idrouter 字面/全组齐检)
            if pin and not (_POS_RE.search(ask) or _doc_in_ask(f"{ask} {opt}", cand_set) == pin):
                pin = None
            s["doc"] = [pin] if pin else [str(x) for x in doc_ids]
        return obj
