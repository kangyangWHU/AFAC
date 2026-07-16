# -*- coding: utf-8 -*-
"""纯几何计算的磁盘缓存(当前仅 crop.subtables 用)。

几何分表/连通域/线检测是纯 CPU 重算,不依赖外部、对同一图确定。重跑评测时 API 结果
由 api_client 缓存命中,但几何仍每图从头算 → CPU-bound 慢。这里按
【输入图内容哈希 + 依赖源文件 mtime】落盘:
  - 内容哈希:同一图命中;
  - 依赖源文件 mtime 进键:改了任一依赖文件,键变 → 自动失效重算(依赖列表由
    @cached 调用点声明,见 crop.py——漏声明 = 陈旧缓存,教训见下)。
"""
import os
import pickle
import hashlib
import functools
import threading
from PIL import Image
from common.config import CACHE_DIR

_DIR = os.path.join(CACHE_DIR, "imcache")
os.makedirs(_DIR, exist_ok=True)


def _img_fp(im):
    """廉价图像指纹:尺寸 + NEAREST 缩到 48×48 后哈希。PIL C 级快速采样,不物化全图
    (避免对巨图 np.asarray/tobytes 的 ~1s 开销)。尺寸+缩略已唯一区分真实文档图。"""
    thumb = im.resize((48, 48), Image.NEAREST).tobytes()
    return "%dx%dx%s_%s" % (im.width, im.height, getattr(im, "mode", "?"),
                            hashlib.sha1(thumb).hexdigest())


def cached(name, *srcfiles):
    """srcfiles: 结果依赖的**全部**源文件(含被调模块,如 subtables 依赖 crop.py+geom.py)。
    漏传依赖文件 → 改了依赖、键不变 → 陈旧缓存(998ada4c off-by-one 修复曾被旧缓存吞掉)。"""
    ver = "-".join(str(int(os.path.getmtime(f))) for f in srcfiles)  # 任一源码变 → 失效
    def deco(fn):
        @functools.wraps(fn)
        def wrap(im, *args, **kw):
            raw = "%s|%s|%s|%s|%s" % (
                name, ver, _img_fp(im), repr(args), repr(sorted(kw.items())))
            path = os.path.join(_DIR, hashlib.sha1(raw.encode()).hexdigest() + ".pkl")
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except Exception:
                    pass
            r = fn(im, *args, **kw)
            try:                                        # 临时文件 + 原子 rename:并发写不互相撕裂
                tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
                with open(tmp, "wb") as f:
                    pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
            except Exception:
                pass
            return r
        return wrap
    return deco
