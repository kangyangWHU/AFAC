"""分类预测器：LightGBM 特征 backbone + Correct & Smooth 图后处理。
fit_predict 返回指定节点的预测类别；可单独评估 backbone vs +C&S 的增量。"""
import numpy as np
import lightgbm as lgb
from scipy.sparse import diags


def normalize_adj(adj_sym):
    A = adj_sym.copy().tolil()
    A.setdiag(1.0)               # 加自环
    A = A.tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = diags(dinv)
    return (D @ A @ D).tocsr()


def correct_and_smooth(P, y_onehot, train_mask, A_norm,
                       a1=0.8, a2=0.8, n1=50, n2=50, scale=1.0):
    R = np.zeros_like(P)
    R[train_mask] = y_onehot[train_mask] - P[train_mask]
    R0 = R.copy()
    for _ in range(n1):                       # Correct: 传播残差
        R = (1 - a1) * R0 + a1 * (A_norm @ R)
    Pc = np.clip(P + scale * R, 0, None)
    H = Pc.copy()
    H[train_mask] = y_onehot[train_mask]
    H0 = H.copy()
    for _ in range(n2):                       # Smooth: 传播标签
        H = (1 - a2) * H0 + a2 * (A_norm @ H)
    return H


class ClsLGBM_CS:
    def __init__(self, num_leaves=63, lr=0.1, n_estimators=300,
                 class_weight="balanced", a1=0.8, a2=0.8, n1=50, n2=50):
        self.p = dict(num_leaves=num_leaves, lr=lr, n_estimators=n_estimators,
                     class_weight=class_weight, a1=a1, a2=a2, n1=n1, n2=n2)

    def fit_predict(self, feat, labels, fit_idx, eval_idx, adj_sym,
                    use_cs=True, n_classes=None):
        X = feat.tocsr()
        n_classes = n_classes or int(labels.max() + 1)
        p = self.p
        # sample weight = balanced
        sw = None
        if p["class_weight"] == "balanced":
            cnt = np.bincount(labels[fit_idx], minlength=n_classes)
            w = len(fit_idx) / (n_classes * np.maximum(cnt, 1))
            sw = w[labels[fit_idx]]
        clf = lgb.LGBMClassifier(num_leaves=p["num_leaves"], learning_rate=p["lr"],
                                 n_estimators=p["n_estimators"], n_jobs=-1, verbose=-1)
        clf.fit(X[fit_idx], labels[fit_idx], sample_weight=sw)
        proba = clf.predict_proba(X)                       # (N, C)
        # 补齐缺失类别列
        full = np.zeros((X.shape[0], n_classes))
        full[:, clf.classes_] = proba
        backbone_pred = full[eval_idx].argmax(1)

        if not use_cs:
            return backbone_pred, dict(stage="backbone")
        A_norm = normalize_adj(adj_sym)
        Yoh = np.zeros((X.shape[0], n_classes))
        Yoh[fit_idx, labels[fit_idx]] = 1.0
        mask = np.zeros(X.shape[0], bool); mask[fit_idx] = True
        H = correct_and_smooth(full, Yoh, mask, A_norm,
                               a1=p["a1"], a2=p["a2"], n1=p["n1"], n2=p["n2"])
        return H[eval_idx].argmax(1), dict(stage="backbone+cs")


def accuracy(pred, true):
    return float((np.asarray(pred) == np.asarray(true)).mean())
