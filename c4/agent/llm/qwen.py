"""Qwen 客户端：OpenAI 兼容（本地 vLLM / 百炼）。唯一合规模型出口。"""
from __future__ import annotations
import time
from openai import OpenAI
from .base import LLMClient, USAGE
from .. import config


class QwenClient(LLMClient):
    def __init__(self):
        super().__init__(usage=USAGE)
        c = config.load()["llm"]
        self.model = c["model"]
        self.temperature = c.get("temperature", 0.0)
        self.max_tokens = c.get("max_tokens", 1024)
        self.enable_thinking = c.get("enable_thinking", False)
        self.seed = c.get("seed", 42)            # 固定 seed → 百炼输出确定可复现(temp0 仍有微随机)
        self.retries = c.get("retries", 2)
        self.base_url = c["base_url"]
        self._is_bailian = "dashscope" in self.base_url
        self._client = OpenAI(base_url=self.base_url,
                              api_key=config.resolve_key(c.get("api_key")),
                              timeout=c.get("timeout", 120))

    def complete(self, messages: list[dict], max_tokens: int | None = None,
                 enable_thinking: bool | None = None, **kw) -> str:
        think = self.enable_thinking if enable_thinking is None else enable_thinking
        # 百炼用扁平 enable_thinking；本地 vLLM 用 chat_template_kwargs
        extra = ({"enable_thinking": think} if self._is_bailian
                 else {"chat_template_kwargs": {"enable_thinking": think}})
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.chat.completions.create(
                    model=self.model, messages=messages,  # type: ignore
                    temperature=kw.get("temperature", self.temperature),
                    max_tokens=max_tokens or self.max_tokens,
                    seed=self.seed, extra_body=extra)
                u = r.usage
                if u:
                    self.usage.add(u.prompt_tokens, u.completion_tokens)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Qwen call failed after {self.retries+1} tries: {last_err}")
