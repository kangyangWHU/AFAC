# -*- coding: utf-8 -*-
"""tile 公共层：两条 OCR 流水线（主路 grid_ocr / 回退路 run_table.ocr_table）共用的
tile 尺寸策略、上采样策略、图像准备（白pad/放大）与 API 并发调用。策略常数只在这里定义。
"""
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

import common.api_client as api
from common.config import MAX_CONCURRENCY, API_USER_IDS

MAX_TILE_COLS = 15   # 单 tile 最大列数。OCR 在"宽 tile × 长数字"上会漏列(b326 26列/tile
# 读不全),限 15 列后 015bd47c +16、b326dfb6 +15、密集表零退化。
MAX_TILE_ROWS = 25   # 单 tile 最大行数。行25 优于行20:行20 切得太碎、表头行易被 OCR 漏读
# →子表切分错(1674392a +52、583ac07b +41),而行20 不救任何表。隔离实验定论:行25。

EDGE_PAD = 3         # tile 四周留白px(边界在缝中心,±3 不吃邻格墨)
ASPECT_SAFE = 180    # API 拒收 >200:1(实测),安全边 180

UP_EDGE = 40         # cell 边长(√像素/cell) < 此值 = 密集小字 → 上采样重读
UP_TARGET = 55       # 目标格边长(实证:edge27 在 2× 已 99.5~99.8%,55/27≈2.0 正中甜点)
UP_CAP = 3.0         # 上采样上限:3× 只留给最小格(19px→2.9×);再高的放大把 tile 撑得
#                      过宽反而让 OCR 漏列(015bd47c 实测)


def upsample_for(edge):
    """密集小字的上采样倍数（两条流水线同一套常数）。"""
    return round(min(UP_CAP, UP_TARGET / edge), 2) if edge < UP_EDGE else 1.0


def upscale(t, factor):
    """LANCZOS 放大；factor≤1 或 t 为 None 时原样返回。"""
    if t is not None and factor and factor > 1:
        return t.resize((round(t.width * factor), round(t.height * factor)),
                        Image.Resampling.LANCZOS)
    return t


def pad_white(core, pad=EDGE_PAD):
    """贴到四周留白 pad px 的白画布上。**白pad而非实pad**:实pad会把相邻块的半截字/
    底边残影带进来,VLM 对残影输出幻觉空行/垃圾格(9c7857f3/34e53b1c 病根)。"""
    core = core.convert("RGB")
    t = Image.new("RGB", (core.width + 2 * pad, core.height + 2 * pad), (255, 255, 255))
    t.paste(core, (pad, pad))
    return t


def call_tiles(imgs, timeout=240, upsample=1, cache_dir=None, retry_rounds=0):
    """并发调用一组 tile（轮询 userId，可选上采样/独立缓存）。
    retry_rounds>0:对"非空图却空返回"(限流被 call_safe 置空→随机丢行、跑分不可复现)
    降并发重试,直到拿到内容或轮次用尽。CACHE_ONLY 离线评测下跳过(空=未缓存)。"""
    def _call_set(idxs, workers):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(
                lambda x: api.call_safe(upscale(imgs[x[1]], upsample), timeout=timeout,
                                        user_id=API_USER_IDS[x[0] % len(API_USER_IDS)],
                                        cache_dir=cache_dir),
                list(enumerate(idxs))))
    idxs = list(range(len(imgs)))
    outs = list(_call_set(idxs, min(MAX_CONCURRENCY, max(1, len(idxs))))) if idxs else []
    if retry_rounds and not api.CACHE_ONLY:
        for _ in range(retry_rounds):
            empt = [i for i in idxs if not (outs[i] or "").strip()]
            if not empt:
                break
            for i, o in zip(empt, _call_set(empt, max(1, min(6, len(empt))))):
                if (o or "").strip():
                    outs[i] = o
    return outs
