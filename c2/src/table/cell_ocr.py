# -*- coding: utf-8 -*-
"""格级局部 OCR(残差修复专用):PP-OCRv6 small rec(≈7.7M 参数,CPU,onnxruntime),
只做识别不做检测——行列位置由骨架给定,代码逐格迭代,模型只回答"这个格子里是什么"。

为什么它能治 API 治不了的病:残差家族(单调区爆行/漏行、首列错位、行跳读)的共同根因
是 VLM 自回归解码靠自己的输出史追踪表内位置,重复内容摧毁追踪(跳行/复读双向失稳)。
逐格读数把位置控制权收回代码,计数病**结构性不可能发生**;单调区('3000'×N)对逐格
识别反而是最简单的输入(B榜实测 80/80,健康区与 API 一致率 95%,分歧处全是 API 的
行漂移被本地读数纠正)。

只用于**残差修复**:诊断(墨迹期望 vs 实读)仍不一致的 tile 才逐格重读;正常 tile
一律走 API(大模型对复杂表头/生僻字仍更强)。本地模型确定性输出,重放天然稳定。"""
import hashlib
import os
import sqlite3

import numpy as np

from common.config import CACHE_DIR

_ENG = None
_DB = None                                   # 本地 rec 结果缓存(确定性模型,重放稳定)
_DB_PID = None                               # fork 安全:子进程各开各的连接


def _db():
    global _DB, _DB_PID
    if _DB is None or _DB_PID != os.getpid():
        path = os.path.join(os.path.dirname(CACHE_DIR), "rec_cache.sqlite")
        _DB = sqlite3.connect(path, timeout=30)
        _DB.execute("PRAGMA journal_mode=WAL")
        _DB.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        _DB_PID = os.getpid()
    return _DB


def _rec_cached(crop, scale):
    """裁剪图 → 文本,带磁盘缓存。键=图内容 md5+尺寸+scale;PP-OCR 确定性输出,
    缓存命中即等价——全量重放 ~8 万格从 ~13 分钟降到分钟内。"""
    key = (hashlib.md5(crop.tobytes()).hexdigest()
           + f"_{crop.width}x{crop.height}_{scale}")
    db = _db()
    row = db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
    if row is not None:
        return row[0]
    c = crop.resize((crop.width * scale, crop.height * scale))
    r = _engine()(np.asarray(c.convert("RGB")),
                  use_det=False, use_cls=False, use_rec=True)
    txt = r.txts[0].strip() if r and getattr(r, "txts", None) else ""
    db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, txt))
    db.commit()
    return txt


def _engine():
    global _ENG
    if _ENG is None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import OCRVersion, ModelType
        _ENG = RapidOCR(params={
            "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL,
            "Global.use_det": False, "Global.use_cls": False})
    return _ENG


def read_strip(sub, rb, cb, i, j0, j1, scale=3):
    """整簇条带识别(压线行专用):骨架行 i 的连续列 [j0, j1] 裁成一条,一次 rec。
    跨列文字(说明行/斜线表头/合并格)整条进模型不切碎——单行长文本是 rec 的本行。"""
    return _rec_cached(sub.crop((cb[j0], rb[i], cb[j1 + 1], rb[i + 1])), scale)


def read_cells(sub, rb, cb, cells, scale=3):
    """骨架格逐格识别。sub=seg图, rb/cb=骨架边界, cells=[(i,j)骨架行列]。
    3x 放大(小字提精度,与 ocr_text 同原则)。返回 {(i,j): text}。"""
    return {(i, j): _rec_cached(sub.crop((cb[j], rb[i], cb[j + 1], rb[i + 1])),
                                scale)
            for (i, j) in cells}
