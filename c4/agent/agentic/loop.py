"""单子问题 agentic loop（plan.md §v4 阶段3）。无状态：每轮只看【子问题 + 一小批块】，
不累积上下文。子问题自带数字，块只需提供规则，judge 直接吐值。
- genQuery: LLM 生成 BM25 检索词（治"整句/数字无法分词"），可在 loop 内重出。
- judge: LLM 读一小批块 → 命中则算值+给来源块号；否则可建议换词。
- 硬上限: max_iter 轮、batch 块/轮、requery 次数。每步留 trace 供审计。
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from ..llm.base import LLMClient
from ..index.bm25 import BM25Index
from .. import config

_GEN_SYS = """你为 BM25 检索生成查询词。给定一个子问题，只输出最能命中相关【条款/规则/指标】的中文关键词（2-6 个，空格分隔）。不要写数字、不要整句、不要解释。只输出关键词一行。"""

_JUDGE_SYS = """你在判断检索块能否回答一个子问题。子问题已自带所有数字，块只需提供【计算规则/条款/事实】。
- 若某块含所需规则/事实：用它 + 子问题里的数字算出答案，输出 JSON：{"found":true,"value":"<带单位的最终值，如 144万>","source":<块号整数>}
- 若这批块都不含：输出 JSON：{"found":false,"requery":"<更可能命中的检索词，没有则空字符串>"}
可先用一句话算一步，最后一行只输出该 JSON。"""

_VERIFY_SYS = """你在核验一条【主张】是否与文档一致。主张已含所有数字，检索块提供事实/条款。
- 若块足以判定：输出 JSON：{"found":true,"verdict":<true 或 false>,"source":<块号整数>}（verdict=该主张是否成立）
- 若块信息不足以判定：输出 JSON：{"found":false,"requery":"<更可能命中的检索词，没有则空字符串>"}
判定要严格：主张里的数字/主体/条件都要与块一致才算 true，有一处对不上即 false。可先核对一句，最后一行只输出该 JSON。"""


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
        self.chunk_chars = a.get("loop_chunk_chars", 420)
        self.backfill_window = a.get("loop_backfill_window", 0)
        self.narrow = a.get("loop_narrow", True)
        self.narrow_k = a.get("loop_narrow_k", 20)
        self.narrow_ratio = a.get("loop_narrow_ratio", 1.5)
        self.genquery_max_tokens = a.get("loop_genquery_max_tokens", 40)
        self.judge_max_tokens = a.get("loop_judge_max_tokens", 400)
        self.evidence_chars = a.get("loop_evidence_chars", 240)

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

    def _expand(self, hit) -> str:
        """small-to-big: 拼接 seq 相邻块, 重连被切散的条文(如跨块的 1.1.6 身故金规则)。"""
        seq = hit.meta.get("seq")
        if not self.backfill_window or seq is None:
            return hit.text[:self.chunk_chars]
        parts = [{"seq": seq, "text": hit.text}]
        parts += self.index.neighbors(hit.doc_id, seq, self.backfill_window)
        parts = sorted({p["seq"]: p for p in parts}.values(), key=lambda x: x["seq"])
        return "\n".join(p["text"] for p in parts)[:self.chunk_chars * 2]

    def _gen_query(self, sq: str, tried: list[str]) -> str:
        user = sq if not tried else f"{sq}\n\n已试过无效的检索词: {' / '.join(tried)}\n换一组更可能命中的词。"
        out = self.llm.complete([{"role": "system", "content": _GEN_SYS},
                                 {"role": "user", "content": user}],
                                max_tokens=self.genquery_max_tokens, enable_thinking=False)
        return out.strip().splitlines()[0][:60] if out.strip() else sq

    def _judge(self, sq: str, hits: list, shape: str) -> dict:
        blocks = []
        for i, h in enumerate(hits):
            bc = h.meta.get("breadcrumb") or h.meta.get("article_no") or ""
            blocks.append(f"[块{i}] {bc}\n{self._expand(h)}")
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
            ranked = self.index.search(terms, k=self.batch * self.search_mult, doc_ids=doc_ids)
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
            v = self._judge(sq, hits, shape)
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
