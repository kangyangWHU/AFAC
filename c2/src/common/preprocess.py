# -*- coding: utf-8 -*-
"""公共轻量预处理。

原则：默认从轻——原图裸切已实测 0.996 精度，任何增强都要先验证涨分再启用。
这里只保留低风险操作：确保 RGB、可选去倾斜。二值化/重增强不在默认链路里
（对密集小数字有害）。
"""
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def ensure_rgb(im):
    """统一成 RGB（API 接受彩色，保留抗锯齿有利于小字识别）。"""
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def estimate_skew(im, max_check_px=4000):
    """粗估倾斜角（度）。截图类一般 ~0，用于决定是否值得纠偏。
    用文字像素的最小外接矩形角度估计；失败返回 0。
    """
    try:
        import cv2
    except Exception:
        return 0.0
    g = np.array(im.convert("L"))
    if g.shape[0] > max_check_px:                 # 超长图只看顶部一段
        g = g[:max_check_px]
    th = cv2.threshold(g, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(th > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def deskew(im, min_angle=0.5):
    """仅当倾斜超过阈值时纠偏，避免无谓重采样。"""
    a = estimate_skew(im)
    if abs(a) < min_angle:
        return im
    return im.rotate(a, expand=True, fillcolor=(255, 255, 255),
                     resample=Image.BICUBIC)


def prep(im, do_deskew=False):
    """默认链路：RGB（去倾斜默认关，需实证有益再开）。"""
    im = ensure_rgb(im)
    if do_deskew:
        im = deskew(im)
    return im
