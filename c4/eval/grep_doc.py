"""按 doc_id + 关键词检索处理后文本块，供人工/Agent 核查证据。
用法: python -m eval.grep_doc <doc_id> <kw1> [kw2 ...] [--ctx N]
任一关键词命中即打印该块(及前后 --ctx 块)。"""
import sys, json, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find_path(doc_id):
    for f in glob.glob(os.path.join(ROOT, "processed", "*", f"{doc_id}.json")):
        return f
    return None

def main():
    args = sys.argv[1:]
    ctx = 0
    if "--ctx" in args:
        i = args.index("--ctx"); ctx = int(args[i+1]); args = args[:i] + args[i+2:]
    doc_id, kws = args[0], args[1:]
    p = find_path(doc_id)
    if not p:
        print(f"NOT FOUND: {doc_id}"); return
    blocks = json.load(open(p, encoding="utf-8"))["blocks"]
    texts = [b.get("text", "") for b in blocks]
    hits = [i for i, t in enumerate(texts) if any(k in t for k in kws)]
    shown = set()
    for i in hits:
        for j in range(max(0, i-ctx), min(len(texts), i+ctx+1)):
            if j in shown: continue
            shown.add(j)
            mark = ">>" if j in hits else "  "
            print(f"{mark}[{doc_id} #{j} {blocks[j].get('type','')}] {texts[j][:2000]}")
    if not hits:
        print(f"(no hit for {kws} in {doc_id}; {len(blocks)} blocks)")

if __name__ == "__main__":
    main()
