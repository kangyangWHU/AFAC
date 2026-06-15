"""配置加载：单例读取 config.yaml，支持点路径访问与默认值。"""
from __future__ import annotations
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG: dict | None = None


def load(path: str | None = None) -> dict:
    global _CFG
    if _CFG is None or path is not None:
        p = path or os.path.join(ROOT, "config.yaml")
        with open(p, encoding="utf-8") as f:
            _CFG = yaml.safe_load(f)
    return _CFG


def get(dotted: str, default=None):
    """get('chunker.target_chars', 600)"""
    cur = load()
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def resolve_key(val: str | None) -> str:
    """支持 'env:VAR' / 'file:path' / 字面量 三种 api_key 写法。
    也兜底读 BAILIAN 环境变量或 .bailian_key 文件。"""
    if val and val.startswith("env:"):
        return os.environ.get(val[4:], "")
    if val and val.startswith("file:"):
        p = val[5:]
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        return open(p).read().strip() if os.path.exists(p) else ""
    if val and val not in ("EMPTY", ""):
        return val
    # 兜底
    if os.environ.get("BAILIAN"):
        return os.environ["BAILIAN"]
    kf = os.path.join(ROOT, ".bailian_key")
    return open(kf).read().strip() if os.path.exists(kf) else "EMPTY"


def path(name: str) -> str:
    """返回 paths.<name> 的绝对路径。"""
    rel = get(f"paths.{name}")
    if rel is None:
        raise KeyError(f"paths.{name} not in config")
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
