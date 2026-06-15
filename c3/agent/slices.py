"""切片误差分析：把一个标量指标拆成多个子群指标，让 LLM 看见【哪里烂】。
这是 Agent 区别于普通 HPO 的关键——决策基于"在冷启动用户/少数类上差"，而非单一分数。
每个切片任务自己实现，但都返回统一结构 {slice_name: {metric, n}}，供 memory/LLM 通用消费。"""
import numpy as np


def classification_slices(pred, true, *, degree, has_labeled_nb, n_classes):
    """pred/true 对齐 eval 节点；degree/has_labeled_nb 为对应 eval 节点的辅助信息。"""
    pred, true = np.asarray(pred), np.asarray(true)
    out = {}

    # 每类准确率（揪出少数类崩溃）
    for c in range(n_classes):
        m = true == c
        if m.sum():
            out[f"class={c}"] = dict(metric=round(float((pred[m] == true[m]).mean()), 4), n=int(m.sum()))

    # 孤立 vs 有边、有无带标签邻居（揪出图信号缺失的子群）
    for name, mask in [("isolated", degree == 0), ("connected", degree > 0),
                       ("has_labeled_nb", has_labeled_nb), ("no_labeled_nb", ~has_labeled_nb)]:
        mask = np.asarray(mask)
        if mask.sum():
            out[name] = dict(metric=round(float((pred[mask] == true[mask]).mean()), 4), n=int(mask.sum()))

    # 最常见的混淆对（LLM 可据此判断该合并/加权哪些类）
    conf = {}
    for p, t in zip(pred, true):
        if p != t:
            conf[(int(t), int(p))] = conf.get((int(t), int(p)), 0) + 1
    out["_top_confusions"] = [dict(true=t, pred=p, n=n)
                              for (t, p), n in sorted(conf.items(), key=lambda x: -x[1])[:5]]
    return out


def _ndcg(pred_list, target, k=10):
    import math
    return 1.0 / math.log2(pred_list[:k].index(target) + 2) if target in pred_list[:k] else 0.0


def recommendation_slices(preds, val_df):
    """preds: {uid:[iid]}；val_df 行含 uid/seq/target_iid。"""
    rows = []
    for _, r in val_df.iterrows():
        seq = r["seq"]; t = str(r["target_iid"]); L = len(seq)
        bucket = "empty" if L == 0 else "cold(1-2)" if L <= 2 else "warm(3-10)" if L <= 10 else "long(>10)"
        is_repeat = t in set(seq)
        rows.append((bucket, is_repeat, _ndcg(preds.get(r["uid"], []), t)))
    rows = np.array(rows, dtype=object)
    out = {}
    for name, mask in [("seq=empty", rows[:, 0] == "empty"),
                       ("seq=cold(1-2)", rows[:, 0] == "cold(1-2)"),
                       ("seq=warm(3-10)", rows[:, 0] == "warm(3-10)"),
                       ("seq=long(>10)", rows[:, 0] == "long(>10)"),
                       ("target=repeat", rows[:, 1] == True),
                       ("target=new", rows[:, 1] == False)]:
        if mask.sum():
            out[name] = dict(metric=round(float(np.mean(rows[mask, 2])), 4), n=int(mask.sum()))
    return out
