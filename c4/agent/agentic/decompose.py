"""规则分解（无 LLM）：把题目拆成一组【可检索 fact】，每条带 ≤2 篇路由。
设计（与用户对齐）：
- facts = ①每选项一条(ask=选项原文) + ②每个【被题干/选项提名】的候选篇一条(ask=题干, 路由该 1 篇)。
  ②保证保险排序那种"多产品"题每篇都被召回(每选项一条不够覆盖)。
- 路由 ≤2 篇/fact，宽松：位置词/文档名 > 实体名(doc_identities 字面命中) > 兜底 top-2。
  无 embedding；路错篇无有效信息、无害。2 篇只由【两个实体名】触发，比较词/数字不触发。
- judge/verdict 已弃：证据由 solver 合并后一次答题。
"""
from __future__ import annotations
import re
from ..textutil import normalize
from .. import config
from .idroute import IdRouter

# 位置词 → 候选篇序数
_POS_RE = re.compile(r"第([一二三四五六七八九十\d]+)[份篇]|前者|后者|前一[份篇]|后一[份篇]")
_DOCNAME_RE = re.compile(r"(?:fc_)?text[_ ]?0*\d+", re.I)
_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
# 聚合/全称量词: 选项含此 → 必须跨篇核(两份均/各/分别) → 取 BM25 top-2(不能只看一篇)
_AGG = re.compile(r"均|都|两份|两者|双方|各|分别")


class Decomposer:
    def __init__(self, index=None, *_, **__):   # index 供 >2篇 无信号时按内容排序; 兼容旧签名
        self.index = index
        self.idr = IdRouter()
        self.ids = self.idr.identities
        a = config.load().get("agentic", {})
        self.lit_min = a.get("route_id_lit_min", 5)   # 实体名字面命中阈(≥此长公共子串, 仅用于实体拆分leg2)
        self.route_ratio = a.get("route_doc_ratio", 0.4)  # 文档级BM25相对阈: 篇分≥ratio×最高分才入选

    # ---- 路由 ----
    def _pos_idx(self, s: str) -> int | None:
        if "前" in s:
            return 0
        if "后" in s:
            return 1
        m = re.search(r"第([一二三四五六七八九十\d]+)", s)
        if not m:
            return None
        t = m.group(1)
        n = _CN.get(t) or (int(t) if t.isdigit() else None)
        return n - 1 if n else None

    def _match_doc(self, ref: str, cand: list[str]) -> str | None:
        """题干式文档名 → 候选原样id。先精确, 否则按末尾数字组唯一匹配。"""
        ref = str(ref).strip()
        if ref in cand:
            return ref
        nums = re.findall(r"\d+", ref)
        if not nums:
            return None
        rn = int(nums[-1])
        hits = [c for c in cand if (m := re.findall(r"\d+", c)) and int(m[-1]) == rn]
        return hits[0] if len(hits) == 1 else None

    def _doc_rank(self, text: str, cand: list[str]) -> list[tuple[str, float]]:
        """每篇内最佳块对【选项文本】的 BM25 分, 降序。绝不掺题干(题干跨多篇会拽偏所有选项)。"""
        if self.index is None:
            return [(c, 0.0) for c in cand]
        out = [(c, (h[0].score if (h := self.index.search_local(text, [c], k=1)) else 0.0))
               for c in cand]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def _route(self, text: str, cand: list[str]) -> list[str]:
        refs: list[str] = []
        for m in _POS_RE.finditer(text):                       # 1) 硬信号: 位置词
            idx = self._pos_idx(m.group(0))
            if idx is not None and 0 <= idx < len(cand):
                refs.append(cand[idx])
        for m in _DOCNAME_RE.findall(text):                    # 2) 硬信号: 文档名
            d = self._match_doc(m, cand)
            if d:
                refs.append(d)
        for c in cand:
            if len(c) >= 4 and c in text:
                refs.append(c)
        refs = list(dict.fromkeys(refs))
        if refs:
            return refs[:2]
        # 3) 文档级 BM25(仅选项文本) + 相对阈值
        ranked = self._doc_rank(text, cand)
        top = ranked[0][1]
        if top <= 0:                                           # 全 0 兜底: ≤2 篇全取
            return cand[:2]
        if _AGG.search(text):                                  # 聚合词(均/都/各) → 跨篇核 top-2
            return [d for d, _ in ranked[:2]]
        keep = [d for d, s in ranked if s >= self.route_ratio * top][:2]  # 单实体独高→1篇; 双实体→2篇
        return keep or [ranked[0][0]]

    def _mentioned(self, d: str, q: dict) -> bool:
        """候选篇 d 的身份名是否在题干/选项里被提名(字面命中)。"""
        name = self.ids.get(d, "")
        if not name:
            return False
        blob = q.get("question", "") + " " + " ".join(str(v) for v in q.get("options", {}).values())
        return IdRouter._lcs(blob, name) >= self.lit_min

    # ---- 分解 ----
    def decompose(self, q: dict, doc_ids: list[str]) -> dict:
        cand = [str(x) for x in doc_ids]
        opts = q.get("options", {})
        fmt = q.get("answer_format")
        facts: list[dict] = []
        seen: set = set()

        def add(option_id: str, ask: str, docs: list[str]):
            ask = normalize(ask or "")
            docs = [d for d in docs if d in cand] or cand[:2]
            key = (ask, tuple(docs))
            if ask and key not in seen:
                seen.add(key)
                facts.append({"id": f"f{len(facts) + 1}", "option_id": option_id,
                              "ask": ask, "doc": docs})

        # ① 每选项一条
        if fmt == "tf" or len([k for k in opts if k in "ABCD"]) < 2:
            add("shared", q.get("question", ""), self._route(q.get("question", ""), cand))
        else:
            for k, v in opts.items():
                if k in "ABCD":
                    add(k, str(v), self._route(str(v), cand))
        # ② 每被提名候选篇一条(ask=题干), 治多产品/多篇覆盖
        for d in cand:
            if self._mentioned(d, q):
                add("shared", q.get("question", ""), [d])
        if not facts:                                          # 极端兜底
            add("shared", q.get("question", ""), cand[:2])
        return {"archetype": "option_verdict", "facts": facts}
