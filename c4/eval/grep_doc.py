"""按 doc_id + 关键词检索 processed_vl 段落，供人工/Agent 核查证据。
用法: python -m eval.grep_doc <doc_id> <kw1> [kw2 ...] [--ctx N]
任一关键词命中即打印该段(及前后 --ctx 段)。"""
import sys, glob, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_dir(doc_id):
    for d in glob.glob(os.path.join(ROOT, "processed_vl", "*", doc_id)):
        if os.path.isdir(d):
            return d
    return None


def main():
    args = sys.argv[1:]
    ctx = 0
    if "--ctx" in args:
        i = args.index("--ctx"); ctx = int(args[i + 1]); args = args[:i] + args[i + 2:]
    doc_id, kws = args[0], args[1:]
    d = find_dir(doc_id)
    if not d:
        print(f"NOT FOUND: {doc_id}"); return
    paras = []  # (page_no, text)
    files = sorted(glob.glob(os.path.join(d, "p*.md")),
                   key=lambda x: int(re.search(r"p(\d+)", os.path.basename(x)).group(1)))
    for f in files:
        pno = int(re.search(r"p(\d+)", os.path.basename(f)).group(1))
        for blk in re.split(r"\n\s*\n", open(f, encoding="utf-8").read()):
            if blk.strip():
                paras.append((pno, blk.strip()))
    hits = [i for i, (_, t) in enumerate(paras) if any(k in t for k in kws)]
    shown = set()
    for i in hits:
        for j in range(max(0, i - ctx), min(len(paras), i + ctx + 1)):
            if j in shown:
                continue
            shown.add(j)
            mark = ">>" if j in hits else "  "
            pg, t = paras[j]
            print(f"{mark}[{doc_id} p{pg} #{j}] {t[:2000]}")
    if not hits:
        print(f"(no hit for {kws} in {doc_id}; {len(paras)} 段)")


if __name__ == "__main__":
    main()
