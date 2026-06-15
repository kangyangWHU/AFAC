"""Qwen（阿里云百炼 / dashscope）调用封装。
- 有 DASHSCOPE_API_KEY 时调用 Qwen 做实验决策；
- 无 key 时返回 None，由 planner 走确定性后备策略（保证可复现 & 可离线开发）。
温度固定 0，所有调用记录到 trajectory，便于复审核对。"""
import os, json, re

DEFAULT_MODEL = os.environ.get("AFAC_QWEN_MODEL", "qwen-plus")


def available():
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def ask_json(system, user, model=None, temperature=0.0):
    """调用 Qwen，强制解析出 JSON 对象。失败或无 key 返回 None。"""
    if not available():
        return None
    try:
        import dashscope
        rsp = dashscope.Generation.call(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            model=model or DEFAULT_MODEL,
            messages=[{"role": "system", "content": system},
                     {"role": "user", "content": user}],
            result_format="message", temperature=temperature,
        )
        text = rsp["output"]["choices"][0]["message"]["content"]
        return _extract_json(text), {"model": model or DEFAULT_MODEL, "raw": text}
    except Exception as e:
        return None, {"error": str(e)}


def _extract_json(text):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(blob)
