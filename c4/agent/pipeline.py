"""端到端：题目 -> 检索 -> 推理 -> 答案。输出 answer.csv + evidence.json + token summary。"""
from __future__ import annotations
import os
import json
import csv
from . import config
from .index.bm25 import BM25Index
from .retriever.retriever import EvidenceRetriever
from .router.router import DocRouter
from .llm.qwen import QwenClient
from .llm.base import USAGE
from .reasoner.reasoner import Reasoner


class Pipeline:
    def __init__(self):
        idx = BM25Index.from_file(os.path.join(config.path("index_dir"), "bm25.pkl"))
        self.retriever = EvidenceRetriever(idx)
        self.router = DocRouter(idx)
        self.route_k = config.get("router.top_k", 5)
        self.reasoner = Reasoner(QwenClient())

    def answer_one(self, q: dict, doc_ids: list[str] | None = None):
        # doc_ids 显式给出(A榜)就用；为 None(B榜)则路由找文档。
        # 注意：不能回退到 q['doc_ids']，否则 B 模式会偷用 gold。
        gold = doc_ids
        dom = q.get("domain")
        if gold is None:
            gold = self.router.select(q["question"], q["options"], dom,
                                      k=self.route_k)
        evs = self.retriever.retrieve(q["question"], q["options"],
                                      doc_ids=gold, domain=dom)
        ans = self.reasoner.answer(q, evs)
        # 自适应兜底：答案为空（模型答"无"/证据不足）→ 提高 min_per_doc 重检索重问
        if (config.get("retrieval.adaptive_on_empty", True) and not ans.answer):
            evs2 = self.retriever.retrieve(
                q["question"], q["options"], doc_ids=gold, domain=dom,
                min_per_doc=config.get("retrieval.adaptive_min_per_doc", 5))
            ans = self.reasoner.answer(q, evs2)
            evs = evs2
        return ans, evs

    def run(self, questions: list[dict], out_dir: str, use_gold_docs: bool = True):
        os.makedirs(out_dir, exist_ok=True)
        n = len(questions)
        results: list[tuple] = [None] * n  # 按题号定位，保证输出顺序与完成顺序无关
        evmap: list[dict] = [None] * n
        workers = int(config.get("run.concurrency", 8))

        def work(i_q):
            i, q = i_q
            dids = q.get("doc_ids") if use_gold_docs else None
            ans, _ = self.answer_one(q, dids)
            return i, q, ans

        if workers > 1:
            # 每题独立、瓶颈在等百炼API(I/O)，多线程并发 ~workers 倍提速
            from concurrent.futures import ThreadPoolExecutor, as_completed
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(work, (i, q)) for i, q in enumerate(questions)]
                for fut in as_completed(futs):
                    i, q, ans = fut.result()
                    results[i] = (q["qid"], ans.answer)
                    evmap[i] = {"qid": q["qid"], "answer": ans.answer,
                                "evidence_doc_ids": ans.evidence_doc_ids,
                                "reasoning": ans.reasoning[-800:]}
                    done += 1
                    print(f"[{done}/{n}] {q['qid']} -> {ans.answer}  "
                          f"(tok={USAGE.total_tokens})")
        else:
            for i, q in enumerate(questions):
                _, _, ans = work((i, q))
                results[i] = (q["qid"], ans.answer)
                evmap[i] = {"qid": q["qid"], "answer": ans.answer,
                            "evidence_doc_ids": ans.evidence_doc_ids,
                            "reasoning": ans.reasoning[-800:]}
                print(f"[{i+1}/{n}] {q['qid']} -> {ans.answer}  "
                      f"(tok={USAGE.total_tokens})")
        rows = results
        evidence = evmap

        # answer.csv（含 summary 行）
        cpath = os.path.join(out_dir, "answer.csv")
        with open(cpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
            w.writerow(["summary", "", USAGE.prompt_tokens, USAGE.completion_tokens,
                        USAGE.total_tokens])
            for qid, a in rows:
                w.writerow([qid, a, "", "", ""])
        json.dump(evidence, open(os.path.join(out_dir, "evidence.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(f"\nanswers -> {cpath}\nsummary tokens: {USAGE.as_dict()}")
        return rows
