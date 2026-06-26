"""为每篇文档生成一行【身份】(产品/主体全称+类型), 供 agentic 路由按语义相似度选篇。
名字常埋在标题之外/正文深处(如"险种简称:安佑福"在第31块), 规则抽取不可靠 -> 让 LLM 读开头自己认。
输出 index/doc_identities.json: {doc_id: "平安安佑福重大疾病保险", ...}
用法: python -m script.build_doc_identities
"""
from __future__ import annotations
import os
import sys
import json
import glob
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agent import config              # noqa: E402
from agent.llm.qwen import QwenClient  # noqa: E402
from agent.vl.chunker import read_doc  # noqa: E402

PROC = os.path.join(ROOT, "processed_vl")

SYS = """给定一份文档的标题与开头内容，用一句话给出它的【身份】——哪个主体的什么文件，尽量含专有名称：
- 保险: 产品全称+险种类型(如 平安安佑福重大疾病保险 / 众安食品安全责任保险)
- 债券/募集: 发行人+债券或募集说明书全称
- 财报: 公司全称+报告类型+年度
- 法规: 办法/规定/指引全称
- 研报: 研究主题
只输出这一行名称本身，不要解释、不要引号、不要前缀。"""


def head(pages: list[dict], npages: int = 2, cap: int = 2400) -> str:
    # 喂最前几页 md(产品名/封面所在)。
    body = "\n\n".join(p["markdown"] for p in pages[:npages])
    return f"开头:\n{body}"[:cap]


def main():
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(PROC, "*", "*", "manifest.json")))
    llm = QwenClient()

    def work(d):
        man, pages = read_doc(d)
        out = llm.complete([{"role": "system", "content": SYS},
                            {"role": "user", "content": head(pages)}],
                           max_tokens=50, enable_thinking=False)
        name = out.strip().splitlines()[0].strip().strip("：:") if out.strip() else ""
        return man["doc_id"], name[:60]

    res: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (did, name) in enumerate(ex.map(work, dirs), 1):
            res[did] = name
            if i % 50 == 0:
                print(f"[{i}/{len(dirs)}] {did}: {name}")
    out_path = os.path.join(config.path("index_dir"), "doc_identities.json")
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {len(res)} identities -> {out_path}")


if __name__ == "__main__":
    main()
