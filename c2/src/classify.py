# -*- coding: utf-8 -*-
"""图像分类路由：判定 LONG（面条图）还是 TABLE（大表）。

依据实测：LONG 宽锁死 ~1500、长宽比 20~128；TABLE 近正方形（~1.4）。
用长宽比阈值即可稳健区分。
"""
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from config import LONG_ASPECT_MIN


def classify(im):
    """返回 'long' 或 'table'。"""
    w, h = im.size
    aspect = max(w, h) / max(1, min(w, h))
    return "long" if aspect >= LONG_ASPECT_MIN else "table"
