"""文档身份路由：把一条 fact 的 ask 按【字面命中】匹配到候选文档的【身份】(doc_identities.json)。
身份是 LLM 读文档开头生成的产品/主体全称(见 script/build_doc_identities.py)。
只在 ask 里【唯一】出现某篇身份的【≥lit_min 长公共子串】(产品/主体名)时定篇——这是高精度信号;
否则返回 None, 让调用方【整组齐检】(不对无命名实体的泛 fact 硬猜, 也不赌嵌入相似度阈值)。
"""
from __future__ import annotations
import os
import json
from .. import config


class IdRouter:
    def __init__(self):
        a = config.load().get("agentic", {})
        self.lit_min = a.get("route_id_lit_min", 5)   # 字面阈: ask 出现某篇身份的≥此长公共子串(产品/主体名)→定篇
        p = os.path.join(config.path("index_dir"), "doc_identities.json")
        self.identities = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        self.ok = bool(self.identities)

    @staticmethod
    def _lcs(a: str, b: str) -> int:
        """最长公共子串长度: ask 与某篇身份名共有的最长连续片段。"""
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        best = 0
        for i in range(1, len(a) + 1):
            cur = [0] * (len(b) + 1)
            for j in range(1, len(b) + 1):
                if a[i - 1] == b[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
            prev = cur
        return best

    def route(self, ask: str, cands: list[str]) -> list[str] | None:
        """ask 唯一字面命中某篇身份→返回 [单篇]; 否则 None(调用方整组齐检)。"""
        if not self.ok:
            return None
        ids = {d: self.identities.get(str(d), "") for d in cands}
        named = [d for d in cands if ids[d]]
        if len(named) < 2:
            return None
        # 字面命中: ask 里出现某篇身份的【≥lit_min 长公共子串】且【唯一】(别篇都短)→ 直接定篇。
        # 治近重复产品 / 案例情景词带偏(产品名子串长、情景词短)。多篇都强匹配→不唯一→None→整组齐检。
        lit = sorted(((d, self._lcs(ask, ids[d])) for d in named), key=lambda x: x[1], reverse=True)
        if lit[0][1] >= self.lit_min and (len(lit) < 2 or lit[1][1] < self.lit_min):
            return [str(lit[0][0])]
        return None
