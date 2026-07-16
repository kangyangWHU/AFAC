# -*- coding: utf-8 -*-
"""公共轻量预处理。

原则：默认从轻——原图裸切已实测 0.996 精度，任何增强都要先验证涨分再启用。
这里只保留低风险操作：确保 RGB。二值化/重增强不在默认链路里
（对密集小数字有害）。
"""
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def ensure_rgb(im):
    """统一成 RGB（API 接受彩色，保留抗锯齿有利于小字识别）。"""
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def prep(im):
    """默认链路：RGB。"""
    return ensure_rgb(im)
