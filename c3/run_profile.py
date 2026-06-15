#!/usr/bin/env python
"""Agent 感知阶段入口：解压数据 → 计算画像 → 生成第一版方案。
用法: python run_profile.py [task_root]   默认当前目录
产出: artifacts/profile_{cls,rec}.json, plan_{cls,rec}.json
"""
import os, sys, json, time
from agent import data_io, profile as P, planner


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=2)
    print("  ->", path)


def main(root="."):
    root = os.path.abspath(root)
    out = os.path.join(root, "artifacts")
    cls_npz, rec_dir = data_io.resolve_paths(root)
    print("classification npz:", cls_npz)
    print("recommendation dir:", rec_dir)

    if cls_npz:
        t = time.time()
        prof = P.profile_classification(data_io.load_classification(cls_npz))
        plan = planner.make_initial_plan(prof)
        print(f"\n[分类] 画像 ({time.time()-t:.1f}s):")
        print(json.dumps(prof, ensure_ascii=False, indent=2))
        print(f"[分类] 第一版方案 (source={plan['source']}, first_trial={plan['first_trial']})")
        dump(prof, os.path.join(out, "profile_cls.json"))
        dump(plan, os.path.join(out, "plan_cls.json"))

    if rec_dir:
        t = time.time()
        prof = P.profile_recommendation(data_io.load_recommendation(rec_dir))
        plan = planner.make_initial_plan(prof)
        print(f"\n[推荐] 画像 ({time.time()-t:.1f}s):")
        print(json.dumps(prof, ensure_ascii=False, indent=2))
        print(f"[推荐] 第一版方案 (source={plan['source']}, first_trial={plan['first_trial']})")
        dump(prof, os.path.join(out, "profile_rec.json"))
        dump(plan, os.path.join(out, "plan_rec.json"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
