"""数据加载：自动解压、按文件特征定位（绕开乱码目录名）、还原稀疏矩阵与序列。"""
import os, glob, zipfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _find(root, pattern):
    hits = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return sorted(hits)[0]


def ensure_unzipped(zip_path, out_dir):
    """把 zip 解到 out_dir（已存在则跳过），返回解压根目录。"""
    os.makedirs(out_dir, exist_ok=True)
    marker = os.path.join(out_dir, ".done")
    if not os.path.exists(marker):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        open(marker, "w").close()
    return out_dir


def load_classification(npz_path):
    """返回 dict: adj(csr,对称0/1), feat(csr), labels, train_idx, test_idx。"""
    d = np.load(npz_path)
    adj = csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]), shape=tuple(d["adj_shape"]))
    feat = csr_matrix((d["attr_data"], d["attr_indices"], d["attr_indptr"]), shape=tuple(d["attr_shape"]))
    sym = adj + adj.T
    sym.data[:] = 1.0
    return dict(adj=adj, adj_sym=sym, feat=feat,
               labels=d["labels"].astype(np.int64),
               train_idx=d["train_idx"].astype(np.int64),
               test_idx=d["test_idx"].astype(np.int64))


def _toks(s):
    if pd.isna(s) or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def load_recommendation(rec_dir):
    """返回 dict: train/test(df, 含解析后的 seq 列表), user, item, items(候选全集)。"""
    train = pd.read_csv(_find(rec_dir, "train.csv"))
    test = pd.read_csv(_find(rec_dir, "test.csv"))
    user = pd.read_csv(_find(rec_dir, "user.csv"))
    item = pd.read_csv(_find(rec_dir, "item.csv"))
    for df in (train, test):
        df["seq"] = df["item_seq_dedup"].apply(_toks)
    return dict(train=train, test=test, user=user, item=item,
               items=item["iid"].astype(str).tolist())


def resolve_paths(task_root):
    """task_root 下找 zip 并解压；返回分类 npz 路径与推荐目录。"""
    cls_zip = glob.glob(os.path.join(task_root, "data", "*分类*.zip"))
    rec_zip = glob.glob(os.path.join(task_root, "data", "*推荐*.zip"))
    work = os.path.join(task_root, "data", "_unzipped")
    cls_npz = rec_dir = None
    if cls_zip:
        cls_npz = _find(ensure_unzipped(cls_zip[0], os.path.join(work, "cls")), "*.npz")
    if rec_zip:
        rec_dir = os.path.dirname(_find(ensure_unzipped(rec_zip[0], os.path.join(work, "rec")), "train.csv"))
    return cls_npz, rec_dir
