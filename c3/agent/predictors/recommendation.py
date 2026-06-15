"""推荐预测器：popularity / repeat_freq / covisitation 的加权融合 blend。
predict 返回 {uid: [top-k iid]}；score 用留出 target 算 NDCG@10。"""
import math
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


def ndcg_at_k(pred, target, k=10):
    if target in pred[:k]:
        return 1.0 / math.log2(pred[:k].index(target) + 2)
    return 0.0


def _norm(v):
    m = v.max()
    return v / m if m > 0 else v


class RecBlend:
    def __init__(self, w_repeat=2.0, w_covis=1.0, w_pop=0.2,
                 covis_window=10, weight_last=1.0, topk=10):
        self.w_repeat, self.w_covis, self.w_pop = w_repeat, w_covis, w_pop
        self.covis_window, self.weight_last, self.topk = covis_window, weight_last, topk

    def fit(self, train_df, items):
        self.items = [str(i) for i in items]
        self.idx = {iid: i for i, iid in enumerate(self.items)}
        n = len(self.items)

        pop = Counter(train_df["target_iid"].astype(str))
        self.pop_score = np.zeros(n)
        for iid, c in pop.items():
            if iid in self.idx:
                self.pop_score[self.idx[iid]] = c
        self.poplist = [iid for iid, _ in pop.most_common(50) if iid in self.idx]

        # 共现矩阵：同一用户去重序列内 item 两两共现（窗口截断）
        cov = defaultdict(float)
        for seq in train_df["seq"]:
            uniq = list(dict.fromkeys(seq))[-self.covis_window:]
            ids = [self.idx[a] for a in uniq if a in self.idx]
            for a in ids:
                for b in ids:
                    if a != b:
                        cov[(a, b)] += 1.0
        if cov:
            ks = np.array(list(cov.keys()))
            self.covis = csr_matrix((list(cov.values()), (ks[:, 0], ks[:, 1])), shape=(n, n))
        else:
            self.covis = csr_matrix((n, n))
        return self

    def _rank_user(self, seq, counts):
        n = len(self.items)
        h = np.zeros(n)  # 用户历史向量（按频次，末位可加权）
        for iid, c in counts:
            if iid in self.idx:
                w = self.weight_last if (seq and iid == seq[-1]) else 1.0
                h[self.idx[iid]] += c * w
        repeat = _norm(h)
        covis = _norm(np.asarray(h @ self.covis).ravel()) if h.any() else np.zeros(n)
        score = self.w_repeat * repeat + self.w_covis * covis + self.w_pop * _norm(self.pop_score)
        order = np.argsort(-score)
        out, seen = [], set()
        for i in order:
            iid = self.items[i]
            if iid not in seen:
                out.append(iid); seen.add(iid)
            if len(out) >= self.topk:
                break
        return out

    @staticmethod
    def _counts(row):
        s = str(row.get("item_seq_counts", "")) if hasattr(row, "get") else str(row)
        return [(c.split(":")[0], int(c.split(":")[1])) for c in s.split(",") if ":" in c]

    def predict(self, df):
        res = {}
        for _, r in df.iterrows():
            res[r["uid"]] = self._rank_user(r["seq"], self._counts(r))
        return res

    def score(self, df):
        s = 0.0
        for _, r in df.iterrows():
            pred = self._rank_user(r["seq"], self._counts(r))
            s += ndcg_at_k(pred, str(r["target_iid"]), self.topk)
        return s / len(df)
