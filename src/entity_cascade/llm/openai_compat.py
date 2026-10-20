"""Client for any OpenAI-compatible /chat/completions endpoint (Ollama, vLLM, cloud APIs)."""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable

import httpx

from entity_cascade.config import LLMConfig
from entity_cascade.llm.base import LLMError, LLMResponse

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAICompatibleLLM:
    def __init__(
        self,
        cfg: LLMConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._cfg = cfg
        self._sleep = sleep
        self._rng = rng or random.Random()
        key = os.environ.get(cfg.api_key_env, "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s),
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._cfg.model

    async def aclose(self) -> None:
        await self._http.aclose()

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, 60.0)
        base = self._cfg.backoff_base_s * (2**attempt)
        return float(base * (0.5 + self._rng.random()))  # jitter in [0.5, 1.5) x base

    async def complete(self, prompt: str) -> LLMResponse:
        body = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        last = "no attempt made"
        for attempt in range(self._cfg.max_retries + 1):
            retry_after: float | None = None
            try:
                resp = await self._http.post("/chat/completions", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp)
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRYABLE_STATUS:
                    raise LLMError(last)  # 4xx other than 429: retrying cannot help
                header = resp.headers.get("retry-after")
                if header is not None and header.replace(".", "", 1).isdigit():
                    retry_after = float(header)
            if attempt < self._cfg.max_retries:
                await self._sleep(self._backoff(attempt, retry_after))
        raise LLMError(f"gave up after {self._cfg.max_retries + 1} attempts: {last}")

    @staticmethod
    def _parse(resp: httpx.Response) -> LLMResponse:
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            return LLMResponse(
                text=text,
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"malformed response: {exc}") from exc
