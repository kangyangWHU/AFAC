#!/usr/bin/env python
"""第一版方案落地：留出评估 → 全量重训 → 生成 A1.csv/A2.csv/prediction.zip。
分类: LightGBM + Correct&Smooth（同时报 backbone vs +C&S 增量）
推荐: repeat+covis+pop 加权 blend
"""
import os, sys, glob, json, time, zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from agent import data_io
from agent.predictors.classification import ClsLGBM_CS, accuracy
from agent.predictors.recommendation import RecBlend


def _find(root, pat):
    h = glob.glob(os.path.join(root, "**", pat), recursive=True)
    return sorted(h)[0] if h else None


def run_classification(cls_npz, out_dir, sub_root):
    d = data_io.load_classification(cls_npz)
    feat, labels, tr, te, adj = d["feat"], d["labels"], d["train_idx"], d["test_idx"], d["adj_sym"]
    nc = int(labels[tr].max() + 1)

    fit_idx, val_idx = train_test_split(tr, test_size=0.2, stratify=labels[tr], random_state=0)
    m = ClsLGBM_CS()
    t = time.time()
    base, _ = m.fit_predict(feat, labels, fit_idx, val_idx, adj, use_cs=False, n_classes=nc)
    cs, _ = m.fit_predict(feat, labels, fit_idx, val_idx, adj, use_cs=True, n_classes=nc)
    acc_base = accuracy(base, labels[val_idx]); acc_cs = accuracy(cs, labels[val_idx])
    print(f"[分类] 留出 acc  backbone={acc_base:.4f}  +C&S={acc_cs:.4f}  增量={acc_cs-acc_base:+.4f}  ({time.time()-t:.1f}s)")

    use_cs = acc_cs >= acc_base
    pred, info = m.fit_predict(feat, labels, tr, te, adj, use_cs=use_cs, n_classes=nc)
    # 按 sample_submission 顺序输出
    sub = pd.read_csv(_find(sub_root, "sample_submission.csv")) if _find(sub_root, "sample_submission.csv") else None
    order = sub["test_idx"].tolist() if sub is not None and "test_idx" in sub.columns else te.tolist()
    pmap = {int(i): int(p) for i, p in zip(te, pred)}
    df = pd.DataFrame({"test_idx": order, "label": [pmap[int(i)] for i in order]})
    path = os.path.join(out_dir, "A1.csv"); df.to_csv(path, index=False)
    print(f"[分类] 最终用 {info['stage']} → {path} ({len(df)}行)")
    return dict(val_backbone=acc_base, val_cs=acc_cs, final_stage=info["stage"])


def run_recommendation(rec_dir, out_dir):
    d = data_io.load_recommendation(rec_dir)
    train, test, items = d["train"], d["test"], d["items"]

    va = train.sample(min(5000, len(train)), random_state=0)
    fit = train.drop(va.index)
    t = time.time()
    blend = RecBlend().fit(fit, items)
    ndcg = blend.score(va)
    print(f"[推荐] 留出 NDCG@10  blend={ndcg:.4f}  ({time.time()-t:.1f}s)")

    blend_full = RecBlend().fit(train, items)
    preds = blend_full.predict(test)
    rows = [(uid, ",".join(preds[uid])) for uid in test["uid"]]
    df = pd.DataFrame(rows, columns=["uid", "prediction"])
    path = os.path.join(out_dir, "A2.csv")
    df.to_csv(path, index=False)  # 默认 QUOTE_MINIMAL：仅含逗号的 prediction 字段加引号
    print(f"[推荐] → {path} ({len(df)}行)")
    return dict(val_blend_ndcg=ndcg)


def main(root="."):
    root = os.path.abspath(root)
    out = os.path.join(root, "artifacts"); os.makedirs(out, exist_ok=True)
    cls_npz, rec_dir = data_io.resolve_paths(root)
    summary = {}
    if cls_npz:
        summary["classification"] = run_classification(cls_npz, out, os.path.dirname(cls_npz))
    if rec_dir:
        summary["recommendation"] = run_recommendation(rec_dir, out)

    zpath = os.path.join(out, "prediction.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("A1.csv", "A2.csv"):
            fp = os.path.join(out, f)
            if os.path.exists(fp): z.write(fp, f)
    json.dump(summary, open(os.path.join(out, "v1_scores.json"), "w"), ensure_ascii=False, indent=2)
    est = 0.5 * summary.get("classification", {}).get("val_cs", 0) + 0.5 * summary.get("recommendation", {}).get("val_blend_ndcg", 0)
    print(f"\n[汇总] 估计 Score_final ≈ {est:.4f}  → {zpath}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
