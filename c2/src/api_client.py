# -*- coding: utf-8 -*-
"""FinixDoc-VL API 客户端。

职责：
  - 统一封装 multipart 图片上传调用（仅图片输入，返回 Markdown）
  - 5 个白名单 userId 轮询，做简单负载均衡
  - 失败重试 + 指数退避 + 超时控制
  - 按图像内容哈希做本地缓存，避免重复调用（省时省额度，且保证可复现）

⚠️ 官方未公开返回体 schema，本模块对返回做**鲁棒解析**：
   优先按 JSON 解析并在常见字段里找 markdown；否则回退为纯文本。
   首次实跑后用 `probe()` 打印原始返回，确认字段名再按需收紧。
"""
import os
import io
import time
import json
import random
import hashlib

import requests
from PIL import Image

from config import API_URL, API_KEY, API_USER_IDS, CACHE_DIR

Image.MAX_IMAGE_PIXELS = None        # 解除 PIL 超大图保护（赛题图可达 3.8 亿像素）

# markdown 可能出现的返回字段名（按经验罗列，命中即用）
_MD_KEYS = ["markdown", "md", "result", "data", "content", "text", "output"]

_user_idx = 0
CACHE_ONLY = False          # 置 True 时只读缓存、未命中返回 ""（API 限流期离线评测用）


def _next_user():
    """轮询取下一个 userId。"""
    global _user_idx
    uid = API_USER_IDS[_user_idx % len(API_USER_IDS)]
    _user_idx += 1
    return uid


def _img_bytes(image, fmt="PNG"):
    """把 路径/PIL.Image/bytes 统一转成 (bytes, filename)。"""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image), "input.png"
    if isinstance(image, str):
        with open(image, "rb") as f:
            return f.read(), os.path.basename(image)
    if isinstance(image, Image.Image):
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return buf.getvalue(), "input." + fmt.lower()
    raise TypeError("image 必须是 路径/PIL.Image/bytes")


def _cache_path(raw):
    h = hashlib.sha1(raw).hexdigest()
    return os.path.join(CACHE_DIR, h + ".md")


def _strip_fences(s):
    """去掉 ```markdown ... ``` / ``` ... ``` 代码围栏。"""
    s = s.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]                       # 去掉首行 ```lang
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _extract_content(obj):
    """从已解析的对象中取出 OpenAI 风格 choices[0].message.content。
    本 API 实测结构：{message,request_id,result:{result:"<json字符串>"}}，
    内层字符串再解析得 {choices:[{message:{content:...}}]}。做多层鲁棒兜底。
    """
    # 1) 标准 chat-completion
    if isinstance(obj, dict):
        ch = obj.get("choices")
        if isinstance(ch, list) and ch:
            msg = ch[0].get("message", {}) if isinstance(ch[0], dict) else {}
            c = msg.get("content")
            if isinstance(c, str):
                return c
        # 2) result.result 里嵌了一层 JSON 字符串
        res = obj.get("result")
        if isinstance(res, dict):
            inner = res.get("result")
            if isinstance(inner, str):
                try:
                    return _extract_content(json.loads(inner))
                except Exception:
                    return inner
            return _extract_content(res)
        if isinstance(res, str):
            try:
                return _extract_content(json.loads(res))
            except Exception:
                return res
    return None


_ERR_MARKERS = ('"success": false', '"success":false', "错误", "异常",
                "Exception", "Request Timeout", "无法查询")


def _looks_like_error(md):
    """判断返回是否是服务端错误信封（而非真正解析内容）。"""
    if not md:
        return False
    head = md.strip()[:300]
    if head.startswith("{") and any(m in head for m in _ERR_MARKERS):
        return True
    return False


def _parse_response(resp):
    """从 HTTP 响应中抽取 markdown 字符串。返回 (markdown, raw_text)。"""
    raw = resp.text
    ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype or raw.strip()[:1] in "{[":
        try:
            obj = resp.json()
        except Exception:
            obj = None
        if obj is not None:
            content = _extract_content(obj)
            if content is not None:
                return _strip_fences(content), raw
            # 兜底：递归找像 markdown 的字段
            found = _find_md_in_obj(obj)
            if found is not None:
                return _strip_fences(found), raw
            return json.dumps(obj, ensure_ascii=False), raw
    return raw, raw


def _find_md_in_obj(obj):
    """在嵌套 dict/list 中按 _MD_KEYS 找最可能的 markdown 文本。"""
    if isinstance(obj, dict):
        for k in _MD_KEYS:
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                return obj[k]
        for v in obj.values():
            r = _find_md_in_obj(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_md_in_obj(v)
            if r is not None:
                return r
    return None


def call(image, *, fmt="PNG", timeout=120, retries=6, use_cache=True,
         user_id=None):
    """调用 FinixDoc-VL，返回 markdown 字符串。

    image     : 图片路径 / PIL.Image / bytes
    fmt       : PIL.Image 时的编码格式（PNG 无损，推荐）
    timeout   : 单次请求超时（秒）
    retries   : 失败重试次数（指数退避）
    use_cache : 命中内容哈希缓存则直接返回
    user_id   : 指定 userId，None 则轮询
    """
    raw_bytes, fname = _img_bytes(image, fmt)

    cpath = _cache_path(raw_bytes)
    if use_cache and os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            cached = f.read()
        if not _looks_like_error(cached):        # 旧缓存里若是错误信封则忽略，重调
            return cached

    if CACHE_ONLY:                               # 离线评测：未命中缓存直接置空，不调 API
        return ""

    last_err = None
    for attempt in range(retries):
        uid = user_id or _next_user()
        try:
            resp = requests.post(
                API_URL,
                data={"userId": uid, "apiKey": API_KEY, "fileName": fname},
                files={"file": (fname, raw_bytes)},
                timeout=timeout,
            )
            if resp.status_code != 200:
                last_err = "HTTP %d: %s" % (resp.status_code, resp.text[:200])
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 2))
                continue
            md, _ = _parse_response(resp)
            if _looks_like_error(md):            # 服务端错误信封 → 当失败，重试
                last_err = md.strip()[:160]
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 2))
                continue
            if use_cache:
                with open(cpath, "w", encoding="utf-8") as f:
                    f.write(md)
            return md
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 2))
    raise RuntimeError("FinixDoc-VL 调用失败（重试 %d 次）：%s" % (retries, last_err))


def call_safe(image, **kw):
    """容错调用：任何异常都返回空串，保证单块失败不拖垮整图/整批。"""
    try:
        return call(image, **kw)
    except Exception as e:
        print("  [warn] tile 调用失败，置空：%s" % str(e)[:120])
        return ""


def probe(image, timeout=120, user_id=None):
    """探测用：返回原始 HTTP 信息，用于第一次确认返回体格式。不走缓存。"""
    raw_bytes, fname = _img_bytes(image)
    uid = user_id or _next_user()
    t0 = time.time()
    resp = requests.post(
        API_URL,
        data={"userId": uid, "apiKey": API_KEY, "fileName": fname},
        files={"file": (fname, raw_bytes)},
        timeout=timeout,
    )
    dt = time.time() - t0
    return {
        "user_id": uid,
        "status": resp.status_code,
        "latency_s": round(dt, 2),
        "content_type": resp.headers.get("Content-Type", ""),
        "len": len(resp.text),
        "head": resp.text[:800],
    }


if __name__ == "__main__":
    # 用一张训练图探测返回格式
    import argparse
    import glob
    from config import TRAIN_TABLE_DIR, TRAIN_LONG_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="指定图片路径；缺省用一张训练图")
    ap.add_argument("--kind", choices=["long", "table"], default="table")
    args = ap.parse_args()

    img = args.image
    if not img:
        d = TRAIN_TABLE_DIR if args.kind == "table" else TRAIN_LONG_DIR
        img = sorted(glob.glob(os.path.join(d, "images", "*.jpg")))[0]
    print("探测图片：", img)
    print("尺寸：", Image.open(img).size)
    info = probe(img)
    print(json.dumps(info, ensure_ascii=False, indent=2))
