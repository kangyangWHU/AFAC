"""v4 Agentic 编排器（plan.md §v4）：decompose → 子问题 loop → verdict/synthesize。
原则：LLM 负责抽取/核验，代码只做结构化解析与合法性校验；不做基于错题的答案补丁。
"""
from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from .. import config
from ..index.bm25 import BM25Index
from ..llm.qwen import QwenClient
from ..postprocess.answer import normalize_answer, is_valid, parse_option_verdicts
from .decompose import Decomposer
from .loop import SubQLoop
from .route import route
from .idroute import IdRouter

# "第N份文档/前者/后者"是给路由用的, 但会毒化 judge。
_DOC_REF = re.compile(r"第[一二三四五六七八九十\d]+份(文档|报告)?之?的?|前者的?|后者的?|两份(文档|报告)?[中里的]*")

_ARCH_HINT = {
    "value_compare": "各子问题给出了每个实体的数值。按选项里陈述的数值与排序，选与这些值最一致的选项。",
    "option_verdict": "已查到各选项所依赖的事实值。逐选项把事实与该选项主张比对：一致则该选项为真。多选题选入所有为真的选项（有正向证据就倾向选入，宁多勿漏）；判断题/单选选为真的那个。",
    "single_fact": "子问题给出了关键事实。选与该事实一致的选项。",
    "fallback": "依据子问题结论与原题直接判断。",
}

_STATEMENT_TRUTH_RE = re.compile(
    r"(下列|以下).{0,12}(说法|表述|陈述|描述|判断).{0,12}(正确|准确|符合|一致|成立)|"
    r"(哪些|哪项).{0,12}(说法|表述|陈述|描述|判断).{0,12}(正确|准确|符合|一致|成立)"
)

_ENTITY_CRITERION_PATTERNS = [
    (re.compile(r"(哪些|哪几|下列哪些|以下哪些).{0,18}(产品|保险|条款).{0,18}(可以|可|能够|能).{0,8}(赔付|赔偿|给付|报销)"),
     "选择在题目给定场景下可以赔付/赔偿/给付/报销的产品。"),
    (re.compile(r"(哪些|哪几|下列哪些|以下哪些).{0,24}(给出|载明|明确).{0,12}(计算方法|计算公式|公式|方法)"),
     "选择条款中明确给出具体计算方法或公式的对象。"),
    (re.compile(r"(哪些|哪几|下列哪些|以下哪些).{0,20}(属于|构成).{0,12}(免责|除外责任|责任免除)"),
     "选择确实属于免责范围/除外责任/责任免除的情形。"),
    (re.compile(r"(哪些|哪几|下列哪些|以下哪些).{0,24}(满足|符合|适用|具有|包含|包括)"),
     "选择满足原题条件的对象。"),
]

_SYN_SYS = """你根据【检索到的原文证据】对原题作答（像人读资料一样，自己从原文里找答案）。
- 【以原文为准】：逐字逐格回原文核对，比数值先对齐日期/报告期/主体/口径/单位，别拿相邻列或别期的数顶替。
- 逐选项把主张与原文比对：原文逐字支持才算真。多选选入所有为真的选项；判断/单选选为真的那个。
- 原文没明说的，尽力判断别空答。
- 最后单独一行输出：答案：<字母>（多选按字母序如 ABD）。"""

_VERDICT_SYS = """你是金融长文档问答的选项核验器。只依据给定【原文证据】（文档原文/表格）判断每个选项是否应【选入答案】。多选漏选=整题错，故判 false 要保守：只在【硬冲突】或【主张范围未被证据证实】时判 false。
逐项判断 A-D，输出结构化 verdict：
- true：原文证据支持该选项【完整主张】。【措辞宽松】——选项把原文概括/同义改写、省略无关紧要的修辞，只要核心事实与方向对得上，就判 true，别因"不够字面"而否。
- false：与原文【硬冲突】判 false——数值不相等、日期/报告期不符、方向相反(增vs减/高vs低)、单位或口径错、主体/义务主体写错、条件/时限矛盾。数值类务必找到【对得上那个日期/报告期/主体的那一格】再比，别拿相邻列或别期顶替(如 35.83% ≠ 35.90%)。
  下列【主张范围词】是选项核心、不是"无关紧要的限定词"，证据未逐一证实就判 false(宁缺勿滥选)：
  ·全称量词(均/都/两份均/所有/任一)：须对它点名的【每个主体/文档】都在证据里证实该谓词，只要一个未证实→false。
  ·具体数值/比例/日期/口径(全年vs年末、合并vs母公司等)：须与证据【同口径同期】精确对上，口径或期间不同→false。
  ·适用场景/范围/对象(某险种承保情形、某条款适用主体等)：选项情形须落在证据所述范围内，情形不符→false，别因"名字沾边"就选。
- unknown：原文证据里【确实找不到】相关依据(是真没有，不是"不够逐字")。
evidence 只能填写【原文证据】中出现的 chunk_id；没有直接证据则填空数组。
最后只输出 JSON，不要 markdown，不要解释。格式：
{"A":{"verdict":"true|false|unknown","evidence":["chunk_id"],"reason":"一句话理由"},"B":{...},"C":{...},"D":{...}}"""

@dataclass
class AgenticAnswer:
    qid: str
    answer: str
    path: str
    archetype: str
    sub: list[dict] = field(default_factory=list)
    verdicts: dict[str, dict] = field(default_factory=dict)
    selection_rule: dict[str, str] = field(default_factory=dict)


class AgenticSolver:
    def __init__(self):
        idx = BM25Index.from_file(os.path.join(config.path("index_dir"), "bm25.pkl"))
        outlines = json.load(open(os.path.join(config.path("index_dir"), "outlines.json"),
                                  encoding="utf-8"))
        self.index = idx
        self.llm = QwenClient()
        self.decomposer = Decomposer(self.llm, outlines)
        self.loop = SubQLoop(self.llm, idx)
        self.idrouter = IdRouter()
        a = config.load().get("agentic", {})
        self.synth_max_tokens = a.get("synth_max_tokens", 600)
        self.fact_workers = a.get("fact_workers", 6)
        self.pool_max_chars = a.get("synth_pool_chars", 16000)
        self.sib = a.get("verify_siblings", 2)   # 每 fact 除 src 外多带几块同源候选(治近重复表/口径不一: 文档同指标多版值)
        self.final_thinking = config.get("llm.reasoning_effort") is not None  # 仅"最终答题"(verdict/synth)开思考; judge不开. 配了reasoning_effort才生效
        self._cid2text = {ch["chunk_id"]: ch["text"] for ch in idx.chunks}
        self._cid2chunk = {ch["chunk_id"]: ch for ch in idx.chunks}

    def _run_one_fact(self, f: dict, doc_ids) -> dict:
        cand = [str(d) for d in doc_ids]
        d = f.get("doc")                              # 分解器钉死的篇(精确命中候选id)直接用; 否则按内容路由
        dids = [d] if d in cand else route(self.index, f["ask"], cand, self.idrouter)
        ask = _DOC_REF.sub("", f["ask"]).strip()
        r = self.loop.solve(ask, dids)
        chunk_ids = [cid for t in r.trace for cid in t.get("chunks", [])]
        return {"ask": f["ask"], "doc": "/".join(dids),
                "option_id": f.get("option_id", "shared"),
                "value": r.value if r.found else "未查到",
                "found": r.found, "src": r.source_chunk_id, "evidence": r.source_text,
                "chunks": chunk_ids}

    def _run_facts(self, d: dict, doc_ids) -> list[dict]:
        facts = d["facts"]
        if self.fact_workers > 1 and len(facts) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.fact_workers) as ex:
                return list(ex.map(lambda f: self._run_one_fact(f, doc_ids), facts))
        return [self._run_one_fact(f, doc_ids) for f in facts]

    def _evidence_pool(self, facts: list[dict], *, tagged: bool = False) -> str:
        seen: set[str] = set()
        out, chars = [], 0
        for s in facts:
            for cid in s.get("chunks", []):
                if cid in seen:
                    continue
                seen.add(cid)
                t = self._cid2text.get(cid)
                if not t:
                    continue
                if tagged:
                    ch = self._cid2chunk.get(cid, {})
                    bc = ch.get("breadcrumb") or ch.get("article_no") or ch.get("type") or ""
                    block = f"[{cid} | {ch.get('doc_id')} | {bc}]\n{t}"
                else:
                    block = t
                out.append(block)
                chars += len(block)
                if chars >= self.pool_max_chars:
                    return "\n---\n".join(out)
        return "\n---\n".join(out)

    def _evidence_by_option(self, q: dict, facts: list[dict]) -> tuple[str, set[str]]:
        """按【选项】分组，每选项只放 judge 选中的【答案块 src】(+相邻块补上下文)，不是整条检索轨迹。
        loop 的价值=judge 已从一批块里挑出答案块；塞全部 chunks 等于丢掉这层筛选、退化成 legacy 大池。
        src 为空(没命中)的 fact 退取它检索到的前1块当弱线索。返回(分组文本, 实际展示的 chunk_id 集)。"""
        opts = [k for k in q.get("options", {}) if k in "ABCD"]
        by_opt: dict[str, list[str]] = {}
        for s in facts:
            bucket = by_opt.setdefault(s.get("option_id") or "shared", [])
            src, ranked = s.get("src"), s.get("chunks", [])
            if src:                                  # 答案块 + 前 sib 个同源候选: 近重复表/口径不一时让 verdict 找到对得上选项的那一版
                picks = [src] + [c for c in ranked if c != src][:self.sib]
            else:
                picks = ranked[:1]                   # 没命中 → 前1块当弱线索
            for i, cid in enumerate(picks):          # 仅 src 补相邻块, 其余只放本块, 控规模
                for c in (self._src_with_context(cid) if i == 0 else [cid]):
                    if c not in bucket:
                        bucket.append(c)
        shown: set[str] = set()

        def render(cids: list[str]) -> str:
            blocks = []
            for cid in cids:
                t = self._cid2text.get(cid)
                if not t:
                    continue
                shown.add(cid)
                ch = self._cid2chunk.get(cid, {})
                bc = ch.get("breadcrumb") or ch.get("article_no") or ch.get("type") or ""
                blocks.append(f"[{cid} | {ch.get('doc_id')} | {bc}]\n{t}")
            return "\n---\n".join(blocks) if blocks else "(该选项未检索到定向证据，参考公共证据)"

        sections = [f"《选项{o} 证据》\n{render(by_opt.get(o, []))}" for o in opts]
        shared = by_opt.get("shared", [])
        if shared:
            sections.append(f"《公共证据(跨选项对比/题干共用)》\n{render(shared)}")
        return "\n\n".join(sections), shown

    def _src_with_context(self, cid: str, window: int = 1) -> list[str]:
        """src 块 + 前后 window 个相邻块(治答案跨块/'详见'指针)，按 seq 连贯排序。"""
        ch = self._cid2chunk.get(cid)
        if not ch:
            return [cid]
        out = {cid}
        seq = ch.get("seq")
        if seq is not None:
            try:
                for nb in self.index.neighbors(ch["doc_id"], seq, window):
                    out.add(nb["chunk_id"])
            except Exception:
                pass
        return sorted(out, key=lambda c: self._cid2chunk.get(c, {}).get("seq", 0))

    @staticmethod
    def _json_obj(text: str) -> dict | None:
        t = re.sub(r"```(?:json)?|```", "", text).strip()
        dec = json.JSONDecoder()
        for i, ch in enumerate(t):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(t[i:])
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    @staticmethod
    def _has_options(obj) -> bool:
        """模型是否真给出了含 A-D 键的 verdict JSON（区分'判了'与'压根没吐 JSON'）。"""
        if not isinstance(obj, dict):
            return False
        raw = obj.get("verdicts") if isinstance(obj.get("verdicts"), dict) else obj
        return isinstance(raw, dict) and any(k in raw for k in "ABCD")

    @staticmethod
    def _norm_verdict(v) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        s = str(v or "").strip().lower()
        false_words = ("false", "no", "假", "错误", "不正确", "不成立", "不符合",
                       "不一致", "冲突", "矛盾")
        true_words = ("true", "yes", "真", "正确", "成立", "符合", "一致", "支持")
        unknown_words = ("unknown", "uncertain", "无法判断", "不能判断", "证据不足",
                         "未查到", "不确定", "未知")
        if any(w in s for w in unknown_words):
            return "unknown"
        if any(w in s for w in false_words):
            return "false"
        if any(w in s for w in true_words):
            return "true"
        return "unknown"

    def _normalize_verdicts(self, obj: dict | None, q: dict,
                            allowed_evidence: set[str]) -> dict[str, dict]:
        raw = obj.get("verdicts") if isinstance(obj, dict) and isinstance(obj.get("verdicts"), dict) else obj
        out: dict[str, dict] = {}
        for opt in sorted(k for k in q.get("options", {}) if k in "ABCD"):
            item = raw.get(opt) if isinstance(raw, dict) else None
            if not isinstance(item, dict):
                out[opt] = {"verdict": "unknown", "evidence": [], "invalid_evidence": [],
                            "reason": "未输出该选项判断"}
                continue
            ev = item.get("evidence", [])
            if isinstance(ev, str):
                ev = [ev]
            if not isinstance(ev, list):
                ev = []
            ev = [str(x).strip() for x in ev if str(x).strip()]
            invalid = [x for x in ev if x not in allowed_evidence]
            ev = [x for x in ev if x in allowed_evidence]
            out[opt] = {
                "verdict": self._norm_verdict(item.get("verdict")),
                "evidence": ev,
                "invalid_evidence": invalid,
                "reason": str(item.get("reason", "")).strip()[:500],
            }
        return out

    @staticmethod
    def _answer_from_verdicts(q: dict, verdicts: dict[str, dict]) -> str:
        true_opts = sorted(k for k, v in verdicts.items() if v.get("verdict") == "true")
        if q.get("answer_format") == "multi":
            return "".join(true_opts)
        if len(true_opts) == 1:
            return true_opts[0]
        return ""

    @staticmethod
    def _selection_rule(q: dict, arch: str) -> dict[str, str]:
        question = str(q.get("question", "")).strip()
        if arch == "value_compare":
            return {
                "mode": "value_compare",
                "criterion": "选择与题干数值、条款规则、计算结果或排序关系最一致的选项。",
                "verdict_true": "该选项的数值/排序/计算结论与证据一致，应选入最终答案。",
                "verdict_false": "该选项的数值/排序/计算结论与证据冲突，或关键比较关系不成立，不应选入最终答案。",
            }
        if _STATEMENT_TRUTH_RE.search(question):
            return {
                "mode": "statement_truth",
                "criterion": "选择被原文证据支持、与文档内容一致的选项陈述。",
                "verdict_true": "该选项陈述整体被原文证据支持，应选入最终答案。",
                "verdict_false": "该选项陈述整体与原文证据冲突，或关键主体/数值/条件/口径不一致，不应选入最终答案。",
            }
        for pat, criterion in _ENTITY_CRITERION_PATTERNS:
            if pat.search(question):
                return {
                    "mode": "entity_satisfies_predicate",
                    "criterion": criterion,
                    "verdict_true": "该选项对应的对象/情形满足原题选择条件，应选入最终答案。",
                    "verdict_false": "该选项对应的对象/情形不满足原题选择条件，不应选入最终答案；即使选项括号里的否定性解释本身被证据支持，也要判 false。",
                }
        return {
            "mode": "statement_truth",
            "criterion": "选择被原文证据支持、与题目要求一致的选项。",
            "verdict_true": "该选项按原题要求被证据支持，应选入最终答案。",
            "verdict_false": "该选项按原题要求不被证据支持或与证据冲突，不应选入最终答案。",
        }

    def _verify_options(self, q: dict, arch: str, facts: list[dict],
                        rule: dict[str, str]) -> tuple[str, dict[str, dict], str]:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        subqs = "\n".join(f"- {s['ask']}" for s in facts)
        pool, allowed = self._evidence_by_option(q, facts)
        user = (f"【题目】{q['question']}\n\n【选项】\n{opts}\n\n"
                f"【答案判定规则】\n"
                f"- mode: {rule['mode']}\n"
                f"- selection_criterion: {rule['criterion']}\n"
                f"- verdict=true: {rule['verdict_true']}\n"
                f"- verdict=false: {rule['verdict_false']}\n"
                f"- 注意：verdict 表示该选项字母是否应进入最终答案；"
                f"不要只判断括号里的解释句是否事实为真。\n\n"
                f"【已分解的待核子问题(回到下面原文证据里逐一核到那一格)】\n{subqs}\n\n"
                f"【原文证据(已按选项分组：判某选项时先核它名下《选项X 相关证据》，"
                f"《公共证据》供跨选项对比)】\n{pool}\n\n"
                f"【题型】{_ARCH_HINT.get(arch, '')}\n"
                "请逐项核验选项，并只输出指定 JSON。")
        out = self.llm.complete([{"role": "system", "content": _VERDICT_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.synth_max_tokens, enable_thinking=self.final_thinking)
        obj = self._json_obj(out)
        if not self._has_options(obj):          # 没吐出含 A-D 的 JSON → 硬提醒重试一次，别直接落 fallback 猜
            out2 = self.llm.complete(
                [{"role": "system", "content": _VERDICT_SYS},
                 {"role": "user", "content": user},
                 {"role": "assistant", "content": out[:600]},
                 {"role": "user", "content": "你没有输出规定格式的 JSON。现在【只】输出那个 JSON（必须含 A/B/C/D 四个键），不要任何解释。"}],
                max_tokens=self.synth_max_tokens, enable_thinking=self.final_thinking)
            if self._has_options(self._json_obj(out2)):
                obj, out = self._json_obj(out2), out2
        verdicts = self._normalize_verdicts(obj, q, allowed)
        ans = self._answer_from_verdicts(q, verdicts)
        return ans, verdicts, out

    def _synthesize(self, q: dict, arch: str, facts: list[dict]) -> str:
        opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
        subqs = "\n".join(f"- {s['ask']}" for s in facts)
        pool = self._evidence_pool(facts)
        user = (f"【题目】{q['question']}\n\n【选项】\n{opts}\n\n"
                f"【已分解的待核子问题(回到原文证据里逐一核实)】\n{subqs}\n\n"
                f"【检索到的原文证据】\n{pool}\n\n"
                f"【题型】{_ARCH_HINT.get(arch, '')}\n"
                "请【以原文证据为准】，对每个选项逐一核对判断，给出最终答案。")
        out = self.llm.complete([{"role": "system", "content": _SYN_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.synth_max_tokens, enable_thinking=self.final_thinking)
        fmt = q.get("answer_format", "mcq")
        ans = normalize_answer(out, fmt) or parse_option_verdicts(out, fmt)
        return ans if is_valid(ans, fmt) else "A"

    def answer(self, q: dict, doc_ids) -> AgenticAnswer:
        d = self.decomposer.decompose(q, doc_ids)
        facts = self._run_facts(d, doc_ids)
        rule = self._selection_rule(q, d["archetype"])
        ans, verdicts, _ = self._verify_options(q, d["archetype"], facts, rule)
        path = "verdict"
        if not is_valid(ans, q.get("answer_format", "mcq")):
            ans = self._synthesize(q, d["archetype"], facts)
            path = f"{path}+synth_fallback"
        return AgenticAnswer(q["qid"], ans, path, d["archetype"], facts, verdicts, rule)
