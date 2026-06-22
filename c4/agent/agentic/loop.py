"""单子问题 agentic loop（plan.md §v4 阶段3）。无状态：每轮只看【子问题 + 一小批块】，
不累积上下文。子问题自带数字，块只需提供规则，judge 直接吐值。
- genQuery: LLM 生成 BM25 检索词（治"整句/数字无法分词"），可在 loop 内重出。
- judge: LLM 读一小批块 → 命中则算值+给来源块号；否则可建议换词。
- 硬上限: max_iter 轮、batch 块/轮、requery 次数。每步留 trace 供审计。
"""
from __future__ import annotations
import os
import json
import re
import threading
from dataclasses import dataclass, field
from ..llm.base import LLMClient
from ..index.bm25 import BM25Index
from .. import config

_GEN_SYS = """你为 BM25 检索生成查询词，目标是在【已选定的单篇文档内】定位某个事实所在的段落/表格。
- 只留能在文档内区分内容的词：指标名/条款名/具体事项/关键数字（如 经营活动现金流量净额、身故保险金、主体信用评级、8894.3、30个工作日、56%）。
- 【去掉用于定位是哪篇文档的词】：公司名(如比亚迪)、报告名、年份(如2024年)、"第一份/第二份/前者/后者"——文档已选定，这些每块都有、是噪声、会把排序带偏。
- 忽略"选项/是否/成立/正确/主张/下列/说法"等设问套话。
2-6 个词，空格分隔，只输出一行，不要解释。"""

_JUDGE_SYS = """你在从检索块里查【一个事实的值】。问题问的是某个具体事实（数值/名称/评级/日期/时限/金额/规则等）。
- 若某块含该事实：抽出它的值，输出 JSON：{"found":true,"value":"<事实的值，如 不超过10亿元 / 广东省广晟控股集团 / AAA / 30个工作日>","source":<块号整数>}。若问题自带数字需套条款计算，就算出最终值。
- 若这批块都不含：输出 JSON：{"found":false,"requery":"<更可能命中的检索词，没有则空字符串>"}
可先核对一句，最后一行只输出该 JSON。"""

_VERIFY_SYS = """你在核验一条【主张】是否与文档一致。主张已含所有数字，检索块提供事实/条款。两步判断：
(1) 这批块里有没有与主张【同一主题/事项】的条款或事实？
    - 完全没有相关内容 → 输出 {"found":false,"requery":"<更可能命中的检索词，没有则空字符串>"}
    - 有相关块 → 必须进入(2)给出 verdict，不要因为措辞不同或不完全确定就说没找到。
(2) 对照该相关块判断主张成立与否：数字/主体/时限/条件实质一致即 verdict=true（措辞不同、表述更宽泛但意思一致也算 true）；有实质冲突才 verdict=false。
    若主张是【跨文档比较】（如"第二份低于第一份""两份均为AAA"），必须在块里分别找出每一份对应的值再比较，不要只看一份就下结论。
    输出 {"found":true,"verdict":<true 或 false>,"source":<块号整数>}
可先核对一句，最后一行只输出该 JSON。"""


@dataclass
class LoopResult:
    sq: str
    value: str | None
    found: bool
    source_chunk_id: str | None
    n_calls: int
    verdict: bool | None = None      # verify shape: 主张是否成立
    source_text: str | None = None   # 命中块的文本片段(喂 synthesize + 审计)
    trace: list[dict] = field(default_factory=list)


def _parse_json(text: str) -> dict | None:
    t = re.sub(r"```(?:json)?|```", "", text)
    cands = re.findall(r"\{[^{}]*\}", t, re.S)        # 取最后一个 {...}
    for m in reversed(cands):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


class SubQLoop:
    def __init__(self, llm: LLMClient, index: BM25Index):
        self.llm = llm
        self.index = index
        a = config.load().get("agentic", {})
        self.max_iter = a.get("loop_max_iter", 3)
        self.batch = a.get("loop_batch", 4)
        self.requery_budget = a.get("loop_requery", 1)
        self.search_mult = a.get("loop_search_mult", 3)
        self.window_chars = a.get("loop_window_chars", 500)   # 喂 judge 的窗口大小(以命中为中心)
        self.narrow = a.get("loop_narrow", True)
        self.narrow_k = a.get("loop_narrow_k", 20)
        self.narrow_ratio = a.get("loop_narrow_ratio", 1.5)
        self.genquery_max_tokens = a.get("loop_genquery_max_tokens", 40)
        self.judge_max_tokens = a.get("loop_judge_max_tokens", 400)
        self.evidence_chars = a.get("loop_evidence_chars", 240)
        # 检索词缓存：同一(子问题,已试词)复用上次生成的检索词 → 跨运行可复现 + 省 token
        self.cache_query = a.get("loop_cache_query", True)
        self._qpath = os.path.join(config.path("index_dir"), "query_cache.json")
        self._qlock = threading.Lock()
        self._qcache = (json.load(open(self._qpath, encoding="utf-8"))
                        if self.cache_query and os.path.exists(self._qpath) else {})

    def _narrow_docs(self, entity: str, doc_ids: list[str]) -> list[str]:
        """value_compare 多文档: 按实体名(产品/公司)缩到最相关的文档, 去跨产品污染。
        规则块常不含产品名, 但封面/标题块含 → 文档级 BM25 能定位。命中明确才缩。"""
        if not (self.narrow and entity and len(doc_ids) > 1):
            return doc_ids
        score: dict[str, float] = {}
        for h in self.index.search(entity, k=self.narrow_k, doc_ids=doc_ids):
            score[h.doc_id] = score.get(h.doc_id, 0.0) + h.score
        if not score:
            return doc_ids
        top = max(score, key=lambda d: score[d])
        # 仅当 top 明显领先才缩, 否则保留全部更安全
        rest = [v for d, v in score.items() if d != top]
        if not rest or score[top] >= self.narrow_ratio * max(rest):
            return [top]
        return doc_ids

    def _window(self, text: str, terms: str) -> str:
        """以 BM25 命中为中心截窗口喂 judge：命中可能在块顶/块底，从头截会错过。
        选包含【最多不同检索词】的窗口，避免被某个高频词带偏。"""
        W = self.window_chars
        if len(text) <= W:
            return text
        toks = [t for t in dict.fromkeys(terms.split()) if len(t) >= 2]
        pos = {t: [m.start() for m in re.finditer(re.escape(t), text)] for t in toks}
        anchors = sorted(p for ps in pos.values() for p in ps)
        if not anchors:
            return text[:W]
        best_start, best_cov = 0, -1
        for a in anchors:                       # 以每个命中为中心试窗, 取覆盖词种最多者
            start = max(0, min(a - W // 2, len(text) - W))
            cov = sum(1 for ps in pos.values() if any(start <= p < start + W for p in ps))
            if cov > best_cov:
                best_cov, best_start = cov, start
        pre = "…" if best_start > 0 else ""
        return pre + text[best_start:best_start + W]

    def _retrieve(self, terms: str, doc_ids: list[str]) -> list:
        """按 query 排序的候选块。多文档时每篇轮询取(round-robin)，
        保证每篇都有代表，不被单篇高分块挤占(治"搜两篇却全召回一篇")。"""
        if len(doc_ids) <= 1:
            return self.index.search(terms, k=self.batch * self.search_mult, doc_ids=doc_ids)
        pools = [self.index.search(terms, k=self.batch * self.search_mult, doc_ids=[d])
                 for d in doc_ids]
        merged, i = [], 0
        while any(i < len(p) for p in pools):
            for p in pools:
                if i < len(p):
                    merged.append(p[i])
            i += 1
        return merged

    def _gen_query(self, sq: str, tried: list[str]) -> str:
        key = f"{sq}||{'/'.join(tried)}"
        if self.cache_query and key in self._qcache:
            return self._qcache[key]
        user = sq if not tried else f"{sq}\n\n已试过无效的检索词: {' / '.join(tried)}\n换一组更可能命中的词。"
        out = self.llm.complete([{"role": "system", "content": _GEN_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.genquery_max_tokens, enable_thinking=False)
        terms = out.strip().splitlines()[0][:60] if out.strip() else sq
        if self.cache_query:
            with self._qlock:
                self._qcache[key] = terms
                json.dump(self._qcache, open(self._qpath, "w", encoding="utf-8"), ensure_ascii=False)
        return terms

    def _judge(self, sq: str, hits: list, shape: str, terms: str) -> dict:
        blocks = []
        for i, h in enumerate(hits):
            bc = h.meta.get("breadcrumb") or h.meta.get("article_no") or ""
            blocks.append(f"[块{i}] {bc}\n{self._window(h.text, terms)}")
        sys = _VERIFY_SYS if shape == "verify" else _JUDGE_SYS
        label = "主张" if shape == "verify" else "子问题"
        user = f"【{label}】{sq}\n\n【候选块】\n" + "\n\n".join(blocks)
        out = self.llm.complete([{"role": "system", "content": sys},
                                 {"role": "user", "content": user}],
                                max_tokens=self.judge_max_tokens, enable_thinking=False)
        return _parse_json(out) or {"found": False, "requery": ""}

    def solve(self, sq: str, doc_ids: list[str], entity: str = "",
              shape: str = "compute") -> LoopResult:
        doc_ids = self._narrow_docs(entity, doc_ids)
        seen: set[str] = set()
        tried_terms: list[str] = []
        requery_left = self.requery_budget
        n_calls = 0
        trace: list[dict] = []

        terms = self._gen_query(sq, tried_terms)
        n_calls += 1
        tried_terms.append(terms)

        for _ in range(self.max_iter):
            ranked = self._retrieve(terms, doc_ids)
            hits = [h for h in ranked if h.chunk_id not in seen][:self.batch]
            if not hits:
                if requery_left:
                    terms = self._gen_query(sq, tried_terms)
                    n_calls += 1
                    tried_terms.append(terms)
                    requery_left -= 1
                    continue
                break
            seen |= {h.chunk_id for h in hits}
            v = self._judge(sq, hits, shape, terms)
            n_calls += 1
            trace.append({"terms": terms, "chunks": [h.chunk_id for h in hits],
                          "verdict": v})
            if v.get("found"):
                si = v.get("source")
                valid = isinstance(si, int) and 0 <= si < len(hits)
                src = hits[si].chunk_id if valid else None
                src_text = hits[si].text[:self.evidence_chars] if valid else None
                verdict = bool(v.get("verdict")) if shape == "verify" else None
                return LoopResult(sq, str(v.get("value", "")).strip() or None, True, src,
                                  n_calls, verdict=verdict, source_text=src_text, trace=trace)
            rq = (v.get("requery") or "").strip()
            if rq and requery_left:
                terms = rq
                tried_terms.append(terms)
                requery_left -= 1

        return LoopResult(sq, None, False, None, n_calls, trace=trace)
