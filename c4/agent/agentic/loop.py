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

_GEN_SYS = """你为 BM25 检索做【同义扩展】——只加、不改、不删。原查询会保留子问题原文(判别词全在)，你只负责补上【同一事物的其它正式写法】，治词法不匹配(原文用别的说法导致漏召)。
任务：只输出子问题里【实体/指标/科目/条款名】的【其它正式写法/同义词】，拼到原查询后面补召回。
- 例：「保单贷款」→ 借款 质押贷款；「归母净利润」→ 归属于母公司股东的净利润 归属于上市公司股东的净利润；「现金分红」→ 派息 现金股利。
- 铁律：只扩【同一事物】的不同写法，绝不换成别的科目/实体(净利润≠归母净利润、营业收入≠营业总收入)；产品名/公司名一字不改。
- 【不要】重复子问题里已有的词、【不要】输出整句或设问套话；没有可扩的就输出空行。
空格分隔，一行，不解释。"""

_JUDGE_SYS = """你从检索块里给出子问题要的【值】。分三类：
A. 直接取值：块里有现成的事实（数值/名称/评级/日期/时限/金额/比例），直接抽出。
B. 套规则计算：子问题自带题干数字、块里是【公式/条款/比例】时，把题干数字代入算出【最终数值】，不要只回公式。
   例：子问题给"基本保额90万、账户85万"，块里是"身故金=max(身故给付比例×基本保额, 账户价值), 40岁比例160%" → 算 max(160%×90, 85)=144万 → value填"144万"。
C. 定性答案【同样算命中】：很多事实的答案【本就不是数字】，而是一条规则/条件/范围/约定/是否。块里讲清了就 found:true，把那条规则原文要点填进 value，【别因为"不是固定数值"就判没命中】。
   例："免赔额是多少"→块说"由投保人与保险人协商确定，并在本合同中载明" → value填"由投保人与保险人协商确定，合同中载明"（这就是答案，不是没命中）。
   例："是否适用等待期"→块说"等待期内确诊不承担给付责任" → value填"适用，等待期内不赔"。"保险责任范围"/"免责范围"同理，把范围/条款要点填进 value。
- 命中(取到值/算出值/取到定性规则)：输出 JSON {"found":true,"value":"<最终值或规则要点>","source":<块号整数>}
- 这批块确实【没有】回答子问题（不是"不是数字"，是真没讲）：输出 JSON {"found":false,"requery":"<更可能命中的检索词，没有则空字符串>"}
可先核对/计算一句，最后一行只输出该 JSON。"""

@dataclass
class LoopResult:
    sq: str
    value: str | None
    found: bool
    source_chunk_id: str | None
    n_calls: int
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
        self.genquery_max_tokens = a.get("loop_genquery_max_tokens", 40)
        self.judge_max_tokens = a.get("loop_judge_max_tokens", 400)
        self.evidence_chars = a.get("loop_evidence_chars", 240)
        # 检索词缓存：同一(子问题,已试词)复用上次生成的检索词 → 跨运行可复现 + 省 token
        self.cache_query = a.get("loop_cache_query", True)
        self._qpath = os.path.join(config.path("index_dir"), "query_cache.json")
        self._qlock = threading.Lock()
        self._qcache = (json.load(open(self._qpath, encoding="utf-8"))
                        if self.cache_query and os.path.exists(self._qpath) else {})
        # BM25 检索结果缓存：同一(query,候选篇)复用上次召回的块序 → 跨运行迭代 judge 时不再重检索
        self.cache_retrieve = a.get("loop_cache_retrieve", True)
        self._rpath = os.path.join(config.path("index_dir"), "retrieve_cache.json")
        self._rlock = threading.Lock()
        self._rcache = (json.load(open(self._rpath, encoding="utf-8"))
                        if self.cache_retrieve and os.path.exists(self._rpath) else {})

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
        """按 query 排序的候选块。用 search_local(per-candidate IDF, 每篇单独重算)修全局IDF坑；
        多文档时每篇轮询取(round-robin)，保证每篇都有代表(治"搜两篇却全召回一篇")。"""
        key = terms + "||" + "/".join(sorted(map(str, doc_ids)))
        if self.cache_retrieve and key in self._rcache:          # 缓存命中: 用块id重建 Hit(loop只用顺序+text+meta)
            return self.index.hits_from_ids(self._rcache[key])
        if len(doc_ids) <= 1:
            merged = self.index.search_local(terms, doc_ids, k=self.batch * self.search_mult)
        else:
            pools = [self.index.search_local(terms, [d], k=self.batch * self.search_mult)
                     for d in doc_ids]
            merged, i = [], 0
            while any(i < len(p) for p in pools):
                for p in pools:
                    if i < len(p):
                        merged.append(p[i])
                i += 1
        if self.cache_retrieve:
            with self._rlock:
                self._rcache[key] = [h.chunk_id for h in merged]
                json.dump(self._rcache, open(self._rpath, "w", encoding="utf-8"), ensure_ascii=False)
        return merged

    def _gen_query(self, sq: str, tried: list[str]) -> str:
        key = f"{sq}||{'/'.join(tried)}"
        if self.cache_query and key in self._qcache:
            return self._qcache[key]
        user = sq if not tried else f"{sq}\n\n已试过无效的检索词: {' / '.join(tried)}\n换一组更可能命中的词。"
        out = self.llm.complete([{"role": "system", "content": _GEN_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.genquery_max_tokens, enable_thinking=False)
        terms = out.strip().splitlines()[0][:120] if out.strip() else sq   # 同义扩展后查询更长, 放宽截断
        if self.cache_query:
            with self._qlock:
                self._qcache[key] = terms
                json.dump(self._qcache, open(self._qpath, "w", encoding="utf-8"), ensure_ascii=False)
        return terms

    def _judge(self, sq: str, hits: list, terms: str) -> dict:
        blocks = []
        for i, h in enumerate(hits):
            bc = h.meta.get("breadcrumb") or h.meta.get("article_no") or ""
            blocks.append(f"[块{i}] {bc}\n{self._window(h.text, terms)}")
        user = f"【子问题】{sq}\n\n【候选块】\n" + "\n\n".join(blocks)
        out = self.llm.complete([{"role": "system", "content": _JUDGE_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.judge_max_tokens, enable_thinking=False)
        return _parse_json(out) or {"found": False, "requery": ""}

    def solve(self, sq: str, doc_ids: list[str]) -> LoopResult:
        seen: set[str] = set()
        tried_terms: list[str] = []
        requery_left = self.requery_budget
        n_calls = 0
        trace: list[dict] = []

        terms = self._gen_query(sq, tried_terms)
        n_calls += 1
        tried_terms.append(terms)

        for _ in range(self.max_iter):
            query = (sq + " " + terms).strip()        # 原始ask(判别词全在)+同义扩展, 一条query; genQuery只增不减, 结构上不丢判别词
            ranked = self._retrieve(query, doc_ids)
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
            v = self._judge(sq, hits, query)          # window 用合并query, 锚点覆盖判别词
            n_calls += 1
            trace.append({"terms": terms, "chunks": [h.chunk_id for h in hits],
                          "judge": v})
            if v.get("found"):
                si = v.get("source")
                valid = isinstance(si, int) and 0 <= si < len(hits)
                src = hits[si].chunk_id if valid else None
                src_text = hits[si].text[:self.evidence_chars] if valid else None
                return LoopResult(sq, str(v.get("value", "")).strip() or None, True, src,
                                  n_calls, source_text=src_text, trace=trace)
            rq = (v.get("requery") or "").strip()
            if rq and requery_left:
                terms = rq
                tried_terms.append(terms)
                requery_left -= 1

        return LoopResult(sq, None, False, None, n_calls, trace=trace)
