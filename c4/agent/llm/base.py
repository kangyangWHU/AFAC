"""LLM 抽象 + 全局 Token 计量（评测要交 summary，必须统计每次调用）。"""
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, p: int, c: int):
        self.prompt_tokens += p
        self.completion_tokens += c
        self.calls += 1

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens, "calls": self.calls}


# 进程级累加器（所有调用经此，保证 summary 完整）
USAGE = TokenUsage()


@dataclass
class LLMClient(ABC):
    usage: TokenUsage = field(default_factory=lambda: USAGE)

    @abstractmethod
    def complete(self, messages: list[dict], **kw) -> str:
        ...

    def ask(self, prompt: str, system: str | None = None, **kw) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.complete(msgs, **kw)
