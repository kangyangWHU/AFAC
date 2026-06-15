"""解析质量验证：条款连续性 + PDF问题分类。
用法：python -m eval.parse_check
"""
from __future__ import annotations
import os
import sys
import json
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
       "八": 8, "九": 9, "十": 10, "百": 100, "零": 0, "〇": 0}


def parse_cn(s: str) -> int:
    s = s.replace("第", "").replace("条", "")
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s.startswith("十"):
        s = "一" + s
    total = tmp = 0
    for ch in s:
        if ch == "十":
            tmp = tmp * 10 if tmp else 10
            total += tmp
            tmp = 0
        elif ch in _CN:
            tmp = _CN[ch]
    return total + tmp


def continuity_check():
    print("=== 条款连续性检验 ===")
    good, issues = 0, []
    files = (glob.glob(os.path.join(ROOT, "processed/regulatory/*.json"))
             + glob.glob(os.path.join(ROOT, "processed/insurance/*.json")))
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        arts = [parse_cn(b["article_no"]) for b in d["blocks"] if b.get("article_no")]
        arts = [a for a in arts if a]
        if len(arts) < 3:
            continue
        uniq = sorted(set(arts))
        dups = len(arts) - len(uniq)
        gaps = [x for x in range(uniq[0], uniq[-1] + 1) if x not in uniq]
        if dups > 0 or len(gaps) > 2:
            issues.append((os.path.basename(f)[:32], len(arts), dups, len(gaps)))
        else:
            good += 1
    print(f"  连续无缺(通过): {good}   异常: {len(issues)}")
    for name, n, dups, ngaps in sorted(issues, key=lambda x: -x[2]):
        print(f"    {name:34} arts={n:>4} dups={dups:>3} gaps={ngaps:>3}")


def probe_check():
    print("\n=== 回指校验（已知证据原文）===")
    probes = {
        ("regulatory", "strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）"):
            ["受益所有人", "差异"],
    }
    for (dom, did), kws in probes.items():
        f = os.path.join(ROOT, "processed", dom, did + ".json")
        if not os.path.exists(f):
            print(f"    {did[:30]}: 文件缺失")
            continue
        txt = "".join(b["text"] for b in json.load(open(f, encoding="utf-8"))["blocks"])
        res = {k: (k in txt) for k in kws}
        print(f"    {did[:30]}: {res}")


if __name__ == "__main__":
    continuity_check()
    probe_check()
