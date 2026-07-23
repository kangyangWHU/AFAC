# -*- coding: utf-8 -*-
"""FinixDoc-VL API 客户端。

职责：
  - 统一封装 multipart 图片上传调用（仅图片输入，返回 Markdown）
  - 5 个白名单 userId 轮询，做简单负载均衡
  - 失败重试 + 指数退避 + 超时控制
  - 按图像内容哈希做本地缓存，避免重复调用（省时省额度，且保证可复现）

返回体 schema(实测已确认): {message,request_id,result:{result:"<json字符串>"}},内层再解析得
   {choices:[{message:{content:...}}]}——见 _extract_content;未知形状由 _find_md_in_obj 兜底。
"""
import os
import io
import time
import json
import random
import hashlib
import re

import requests
from PIL import Image

from common.config import (API_URL, API_KEY, API_USER_IDS, CACHE_DIR,
                    API_TIMEOUT, API_RETRIES, API_BACKOFF_CAP)

Image.MAX_IMAGE_PIXELS = None        # 解除 PIL 超大图保护（赛题图可达 3.8 亿像素）
CACHE_UP_DIR = os.path.join(os.path.dirname(CACHE_DIR), "cache_up")  # 上采样 tile 缓存(分离)

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


def _cache_path(raw, cache_dir=None):
    h = hashlib.sha1(raw).hexdigest()
    d = cache_dir or CACHE_DIR
    if d is not CACHE_DIR:
        os.makedirs(d, exist_ok=True)
    return os.path.join(d, h + ".md")


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
_ERR_PAGES = ("<!doctype", "<html")   # 服务端偶发吐 HTML 错误页——不是解析内容,当失败
#               (6c205e72 全量高压下 '<!DOCTYPE' 被当单元格内容入表并污染缓存)


def _looks_like_error(md):
    """判断返回是否是服务端错误信封/错误页（而非真正解析内容）。"""
    if not md:
        return False
    head = md.strip()[:300]
    if head.startswith("{") and any(m in head for m in _ERR_MARKERS):
        return True
    low = head.lower()
    if any(low.startswith(p) or p in low[:80] for p in _ERR_PAGES):
        return True
    return False


def is_truncated(md):
    """tile 输出被 ~12k 字符上限截断：有 <table 却无闭合 </table>。
    截断是 200 成功内容、错误信封识别不出，必须单独拦下：**不写缓存**，
    由上层 _split_call_merge 拆小重读后用 write_cache 回填完整结果。"""
    if not md:
        return False
    low = md.lower()
    return "<table" in low and "</table>" not in low


def is_degenerate(md, min_lines=40, dup_ratio=0.5):
    """VLM 退化成复读:同几行被反复吐出直到长度上限。

    截断检测抓不到它(标签是闭合的、也不是错误信封),但它比截断更毒 —— 内容看着
    正常,却把一行灌进去上百次,TextEdit 直接崩。判据是版面事实:一个正常的文档
    横条不会有一半以上的**有字**行是重复行。

    只数含文字的行:稀疏表里 `<tr><td></td><td></td></tr>` 连出几百行是真实版面,
    不是退化。行数少的短条带不判(目录/清单本就多重复行)。
    """
    lines = []
    for line in (md or "").splitlines():
        text = re.sub(r"<[^>]+>", "", line).strip()
        if text:
            lines.append(text)
    if len(lines) < min_lines:
        return False
    return (len(lines) - len(set(lines))) / len(lines) > dup_ratio


def collapse_repeats(md):
    """把复读退化的返回压回近似正确的内容:同一行只保留首次出现,顺序不变。

    退化响应里真实内容是在的,只是被反复灌了几十上百遍。整行完全相同在正文里
    本就罕见,而这里的替代方案是整条带作废(内容全丢),压重复严格更优。
    """
    seen, out = set(), []
    for line in (md or "").splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def write_cache(image, md, *, fmt="PNG", cache_dir=None):
    """把上层修复后的**完整** tile 结果回写缓存，覆盖此前因截断而未落盘的 key。
    使下次（含 CACHE_ONLY 离线评测）直接命中完整结果、无需再拆分重读。"""
    raw_bytes, _ = _img_bytes(image, fmt)
    with open(_cache_path(raw_bytes, cache_dir), "w", encoding="utf-8") as f:
        f.write(md)


def _parse_response(resp):
    """从 HTTP 响应中抽取 markdown 字符串。"""
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
                return _strip_fences(content)
            # 兜底：递归找像 markdown 的字段
            found = _find_md_in_obj(obj)
            if found is not None:
                return _strip_fences(found)
            return json.dumps(obj, ensure_ascii=False)
    return raw


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


def call(image, *, fmt="PNG", timeout=API_TIMEOUT, retries=API_RETRIES, use_cache=True,
         user_id=None, cache_dir=None):
    """调用 FinixDoc-VL，返回 markdown 字符串。

    image     : 图片路径 / PIL.Image / bytes
    fmt       : PIL.Image 时的编码格式（PNG 无损，推荐）
    timeout   : 单次请求超时（秒）
    retries   : 失败重试次数（指数退避）
    use_cache : 命中内容哈希缓存则直接返回
    user_id   : 指定 userId，None 则轮询
    cache_dir : 缓存目录（默认 CACHE_DIR；上采样 tile 用 cache_up/ 分离，便于对比/回退）
    """
    raw_bytes, fname = _img_bytes(image, fmt)

    cpath = _cache_path(raw_bytes, cache_dir)
    if use_cache and os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            cached = f.read()
        # 错误信封 / 旧的截断半截结果 → 视为未命中，重调（截断会触发拆分修复）
        if (not _looks_like_error(cached) and not is_truncated(cached)
                and not is_degenerate(cached)):
            return cached

    if CACHE_ONLY:                               # 离线评测：未命中缓存直接置空，不调 API
        return ""

    last_err = None
    last_degen = None                            # 重试全退化时的兜底素材
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
                time.sleep(min(2 ** attempt, API_BACKOFF_CAP) + random.uniform(0, 2))
                continue
            md = _parse_response(resp)
            if is_degenerate(md):                # 复读退化 → 当失败，重试(**不写缓存**)
                last_degen = md
                last_err = "degenerate(重复行占比过高) len=%d" % len(md)
                time.sleep(min(2 ** attempt, API_BACKOFF_CAP) + random.uniform(0, 2))
                continue
            if _looks_like_error(md):            # 服务端错误信封 → 当失败，重试
                last_err = md.strip()[:160]
                if "Error code: 400" in md or "aspect ratio" in md:
                    break                        # 永久性错误(参数/图形不合法):重试无意义,
                #                                  fail-fast(此前 8 次重试×4 轮白烧几分钟)
                time.sleep(min(2 ** attempt, API_BACKOFF_CAP) + random.uniform(0, 2))
                continue
            # 截断（输出超 ~12k 上限）**不写缓存**：避免半截结果污染缓存，交由
            # 上层 _split_call_merge 拆小重读、再 write_cache 回填完整结果。
            if use_cache and not is_truncated(md):
                with open(cpath, "w", encoding="utf-8") as f:
                    f.write(md)
            return md
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, API_BACKOFF_CAP) + random.uniform(0, 2))
    if last_degen is not None:
        # 重试次次退化 = 该条带就是会让模型复读,再打也一样。压掉重复取回内容,
        # 并落缓存:否则每次运行都要在这里空烧一轮退避。
        fixed = collapse_repeats(last_degen)
        if use_cache:
            with open(cpath, "w", encoding="utf-8") as f:
                f.write(fixed)
        return fixed
    raise RuntimeError("FinixDoc-VL 调用失败（重试 %d 次）：%s" % (retries, last_err))


def call_safe(image, **kw):
    """容错调用：任何异常都返回空串，保证单块失败不拖垮整图/整批。"""
    try:
        return call(image, **kw)
    except Exception as e:
        print("  [warn] tile 调用失败，置空：%s" % str(e)[:120])
        return ""


# ---------------------------------------------------------------------------
# 统一发送层(唯一的并发出口):所有需要发 tile 的地方一律经此,调用方不自办线程池。
# API 是纯 IO 等待 → 线程池;CPU 并行仍由上层图级进程池负责,两层各司其职。
# ---------------------------------------------------------------------------
_EXEC = None


def _pool():
    """每进程惰性建一个共享线程池(fork 后子进程各自新建,互不串台)。"""
    global _EXEC
    if _EXEC is None:
        from concurrent.futures import ThreadPoolExecutor
        from common.config import MAX_CONCURRENCY
        _EXEC = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
    return _EXEC


def submit(image, **kw):
    """单发:返回 Future(call_safe 语义,异常置空)。"""
    return _pool().submit(call_safe, image, **kw)


def call_many(images, *, rotate_user=True, retry_rounds=0, **kw):
    """批量发送,次序保持。userId 轮询;retry_rounds>0 对"非空图空返回"(限流被
    call_safe 静默置空 → 随机丢行、跑分不可复现)按 ≤6 并发分组重试,直到拿到内容
    或轮次用尽;CACHE_ONLY 离线评测下跳过重试(空=未缓存,重试无意义)。"""
    if not images:
        return []
    futs = [_pool().submit(call_safe, im,
                           user_id=(API_USER_IDS[i % len(API_USER_IDS)]
                                    if rotate_user else None), **kw)
            for i, im in enumerate(images)]
    outs = [f.result() for f in futs]
    if retry_rounds and not CACHE_ONLY:
        for _ in range(retry_rounds):
            empt = [i for i, o in enumerate(outs) if not (o or "").strip()]
            if not empt:
                break
            for k in range(0, len(empt), 6):          # 降并发重试(≤6 在飞)
                grp = empt[k:k + 6]
                fs = [_pool().submit(call_safe, images[i], **kw) for i in grp]
                for i, f in zip(grp, fs):
                    o = f.result()
                    if (o or "").strip():
                        outs[i] = o
    return outs


def map_io(fn, items):
    """把一批**含内部串行 API 调用**的任务函数并发跑(如坏 tile 修复:拆半→判→再拆)。
    fn 内部只允许用 call_safe(同步直连),不得再 submit/call_many —— 单层池无嵌套等待,
    天然无死锁。返回与 items 同序的结果列表。"""
    if not items:
        return []
    futs = [_pool().submit(fn, it) for it in items]
    return [f.result() for f in futs]


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
    from common.config import TRAIN_TABLE_DIR, TRAIN_LONG_DIR

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
