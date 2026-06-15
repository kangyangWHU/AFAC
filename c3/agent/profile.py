"""数据画像：程序化计算决定建模方向的关键信号，产出结构化 profile（喂给 planner）。
所有数字均运行时从数据算出，不含任何写死的任务答案。"""
import math
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _pct(x):
    return round(float(x), 4)


def profile_classification(data, quick_val=True):
    sym, feat = data["adj_sym"], data["feat"]
    labels, tr, te = data["labels"], data["train_idx"], data["test_idx"]
    n = sym.shape[0]
    deg = np.asarray((sym > 0).sum(1)).ravel()
    tr_set = set(tr.tolist())

    # 同类边比例（两端都有标签）—— 决定 GNN / C&S 是否有用
    coo = sym.tocoo()
    same = tot = 0
    for i, j in zip(coo.row, coo.col):
        if i < j and i in tr_set and j in tr_set:
            tot += 1
            same += int(labels[i] == labels[j])
    homophily = _pct(same / tot) if tot else None

    # 测试点是否能吃到图信号
    csr = sym.tocsr()
    has_lab_nb = sum(
        any(nb in tr_set for nb in csr.indices[csr.indptr[t]:csr.indptr[t + 1]])
        for t in te
    )

    cls_cnt = dict(sorted(Counter(labels[tr].tolist()).items()))
    imbalance = _pct(max(cls_cnt.values()) / max(min(cls_cnt.values()), 1))

    prof = dict(
        task="classification", metric="accuracy",
        n_nodes=int(n), n_edges_undirected=int((sym > 0).sum() // 2),
        avg_degree=_pct(deg.mean()), median_degree=int(np.median(deg)),
        isolated_ratio=_pct((deg == 0).mean()),
        n_features=int(feat.shape[1]),
        feat_nnz_per_node=_pct(feat.getnnz(axis=1).mean()),
        n_classes=len(cls_cnt), class_counts=cls_cnt, imbalance_ratio=imbalance,
        edge_homophily=homophily,
        test_with_labeled_neighbor_ratio=_pct(has_lab_nb / len(te)),
        n_train=int(len(tr)), n_test=int(len(te)),
    )

    if quick_val:  # 给 Agent 一个真实的 round-0 反馈：多数类基线 + 标准化特征 LR
        from sklearn.preprocessing import MaxAbsScaler
        X = feat.tocsr()
        Xtr, Xva, ytr, yva = train_test_split(
            X[tr], labels[tr], test_size=0.2, stratify=labels[tr], random_state=0)
        sc = MaxAbsScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(Xtr), ytr)
        maj = Counter(ytr.tolist()).most_common(1)[0][0]
        prof["val_majority_baseline_acc"] = _pct((yva == maj).mean())
        prof["val_feature_only_LR_acc"] = _pct(clf.score(sc.transform(Xva), yva))
    return prof


def _ndcg_at_k(pred, target, k=10):
    if target in pred[:k]:
        return 1.0 / math.log2(pred[:k].index(target) + 2)
    return 0.0


def profile_recommendation(data, quick_val=True):
    train, test = data["train"], data["test"]
    items = data["items"]
    pop = Counter(train["target_iid"].astype(str))
    poplist = [i for i, _ in pop.most_common(50)]
    top10 = set([i for i, _ in pop.most_common(10)])

    tr_L = train["seq"].apply(len)
    te_L = test["seq"].apply(len)
    n = len(train)
    in_seq = last_eq = mf_eq = in_top = 0
    for _, r in train.iterrows():
        seq = r["seq"]; t = str(r["target_iid"])
        sset = set(seq)
        in_seq += t in sset
        if seq and seq[-1] == t: last_eq += 1
        if t in top10: in_top += 1
        cnts = [c for c in str(r["item_seq_counts"]).split(",") if ":" in c]
        if cnts:
            mf = max(cnts, key=lambda c: int(c.split(":")[1])).split(":")[0]
            mf_eq += int(mf == t)

    prof = dict(
        task="recommendation", metric="ndcg@10",
        n_users=int(len(data["user"])), n_train=int(n), n_test=int(len(test)),
        n_items=int(len(items)),
        train_seq_len=dict(mean=_pct(tr_L.mean()), median=int(tr_L.median()),
                          p90=int(tr_L.quantile(.9)), max=int(tr_L.max())),
        test_seq_len=dict(mean=_pct(te_L.mean()), median=int(te_L.median()),
                         p90=int(te_L.quantile(.9)), max=int(te_L.max())),
        cold_start_share=_pct((te_L <= 2).mean()),
        empty_test_share=_pct((te_L == 0).mean()),
        repeat_rate=_pct(in_seq / n),
        target_eq_last=_pct(last_eq / n),
        target_eq_mostfreq=_pct(mf_eq / n),
        target_in_global_top10=_pct(in_top / n),
        distinct_targets=int(len(pop)), target_concentration_top10=_pct(
            sum(c for _, c in pop.most_common(10)) / n),
    )

    if quick_val:  # round-0 反馈：复购频次 + 热门回填 的留出 NDCG@10
        va = train.sample(min(5000, n), random_state=0)
        s_pop = s_freq = 0.0
        for _, r in va.iterrows():
            t = str(r["target_iid"])
            cnts = [c for c in str(r["item_seq_counts"]).split(",") if ":" in c]
            rank = [c.split(":")[0] for c in sorted(cnts, key=lambda c: -int(c.split(":")[1]))]
            predf = rank + [i for i in poplist if i not in set(rank)]
            s_freq += _ndcg_at_k(predf, t)
            s_pop += _ndcg_at_k(poplist, t)
        prof["val_popularity_ndcg"] = _pct(s_pop / len(va))
        prof["val_repeat_plus_pop_ndcg"] = _pct(s_freq / len(va))
    return prof
