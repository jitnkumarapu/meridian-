"""Small provider-neutral LLM interface and concrete HTTP clients."""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx


class LLMError(RuntimeError):
    """Raised for actionable provider configuration or response errors."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model_id: str
    token_usage: int | None = None
    retries: int = 0


_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


def _safe_post(
    url: str,
    *,
    provider: str,
    headers: Mapping[str, str] | None = None,
    json: Any = None,
    timeout: float = 60,
    attempts: int = 3,
) -> tuple[httpx.Response, int]:
    """POST with bounded transient retries and secret-free errors."""
    for attempt in range(attempts):
        try:
            response = httpx.post(url, headers=headers, json=json, timeout=timeout)
            response.raise_for_status()
            return response, attempt
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status not in _TRANSIENT_STATUS or attempt == attempts - 1:
                raise LLMError(f"{provider} request failed with HTTP {status}") from exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == attempts - 1:
                raise LLMError(f"{provider} request failed: {type(exc).__name__}") from exc
        time.sleep(0.1 * (2**attempt))
    raise AssertionError("unreachable")


class LLMClient(Protocol):
    @property
    def model_id(self) -> str: ...

    def complete(self, prompt: str, *, image_paths: list[str] | None = None) -> LLMResponse: ...


class MockLLM:
    """Deterministic evidence-only generator for tests and explicit demo mode."""

    def __init__(self, responses: list[str] | None = None, model_id: str = "mock-evidence-v1") -> None:
        self._responses = list(responses or [])
        self._model_id = model_id
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, prompt: str, *, image_paths: list[str] | None = None) -> LLMResponse:
        del image_paths
        self.calls += 1
        if self._responses:
            text = self._responses.pop(0)
        else:
            import re

            match = re.search(
                r"EVIDENCE \d+ \[(?P<label>[^\]\n]+, p\.\d+, (?:text|table|figure))\]",
                prompt,
            )
            if match is None:
                text = "Insufficient evidence"
            else:
                evidence = ""
                content_match = re.search(r"CONTENT:\s*(.+)", prompt)
                if content_match:
                    evidence = content_match.group(1).strip()[:240]
                text = f"Based on the retrieved evidence: {evidence} [{match.group('label')}]".strip()
        return LLMResponse(text=text, model_id=self.model_id, token_usage=len(text.split()))


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        timeout: float = 60,
        retry_count: int = 2,
    ) -> None:
        self._model = model
        resolved_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.base_url = resolved_url.rstrip("/")
        resolved_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not resolved_key:
            raise LLMError("OPENAI_API_KEY is required for the OpenAI-compatible provider")
        self.api_key = resolved_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_count = retry_count

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str, *, image_paths: list[str] | None = None) -> LLMResponse:
        if image_paths:
            raise LLMError("This HTTP client accepts text evidence only; disable generation.use_vision")
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response, retries = _safe_post(
            f"{self.base_url}/chat/completions",
            provider="OpenAI-compatible",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
            attempts=self.retry_count + 1,
        )
        body = response.json()
        try:
            text = str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI-compatible endpoint returned an unexpected response") from exc
        usage = body.get("usage", {}).get("total_tokens")
        return LLMResponse(text=text, model_id=self._model, token_usage=usage, retries=retries)


class AnthropicClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        timeout: float = 60,
        retry_count: int = 2,
    ) -> None:
        self._model = model
        resolved_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if not resolved_key:
            raise LLMError("ANTHROPIC_API_KEY is required for the Anthropic provider")
        self.api_key = resolved_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_count = retry_count

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str, *, image_paths: list[str] | None = None) -> LLMResponse:
        if image_paths:
            raise LLMError("Vision input is not enabled by this client; disable generation.use_vision")
        response, retries = _safe_post(
            "https://api.anthropic.com/v1/messages",
            provider="Anthropic",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
            attempts=self.retry_count + 1,
        )
        body = response.json()
        try:
            text = "".join(str(block["text"]) for block in body["content"] if block["type"] == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError("Anthropic returned an unexpected response") from exc
        usage_data = body.get("usage", {})
        usage = usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
        return LLMResponse(text=text, model_id=self._model, token_usage=usage, retries=retries)


class GeminiClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0,
        max_tokens: int = 512,
        timeout: float = 60,
        retry_count: int = 2,
    ) -> None:
        self._model = model
        resolved_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        if not resolved_key:
            raise LLMError("GEMINI_API_KEY is required for the Gemini provider")
        self.api_key = resolved_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_count = retry_count

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str, *, image_paths: list[str] | None = None) -> LLMResponse:
        parts: list[dict[str, object]] = [{"text": prompt}]
        for image_path in image_paths or []:
            path = Path(image_path)
            if not path.is_file():
                raise LLMError(f"Gemini image evidence file is missing: {path}")
            mime_type, _ = mimetypes.guess_type(path.name)
            if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise LLMError(f"Unsupported Gemini image evidence type: {path.suffix}")
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                }
            )
        response, retries = _safe_post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent",
            provider="Gemini",
            headers={"x-goog-api-key": self.api_key},
            json={
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            },
            timeout=self.timeout,
            attempts=self.retry_count + 1,
        )
        body = response.json()
        try:
            text = "".join(str(part["text"]) for part in body["candidates"][0]["content"]["parts"] if "text" in part)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Gemini returned an unexpected response") from exc
        usage = body.get("usageMetadata", {}).get("totalTokenCount")
        return LLMResponse(text=text, model_id=self._model, token_usage=usage, retries=retries)


class OllamaClient:
    def __init__(self, model: str, *, base_url: str | None = None, timeout: float = 120, retry_count: int = 2) -> None:
        self._model = model
        resolved_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        self.base_url = resolved_url.rstrip("/")
        self.timeout = timeout
        self.retry_count = retry_count

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, prompt: str, *, image_paths: list[str] | None = None) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        if image_paths:
            import base64
            from pathlib import Path

            payload["images"] = [base64.b64encode(Path(path).read_bytes()).decode() for path in image_paths]
        try:
            response, retries = _safe_post(
                f"{self.base_url}/api/generate",
                provider="Ollama",
                json=payload,
                timeout=self.timeout,
                attempts=self.retry_count + 1,
            )
        except LLMError as exc:
            raise LLMError(f"Ollama is unavailable at {self.base_url}") from exc
        body = response.json()
        if "response" not in body:
            raise LLMError("Ollama returned an unexpected response")
        usage = body.get("prompt_eval_count", 0) + body.get("eval_count", 0)
        return LLMResponse(text=str(body["response"]), model_id=self._model, token_usage=usage, retries=retries)


def provider_health(provider: str, *, use_vision: bool = False, key_env: str | None = None) -> dict[str, object]:
    """Return readiness without making a billable provider request."""
    normalized = provider.casefold()
    required = key_env or {
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openai-compatible": "OPENAI_API_KEY",
    }.get(normalized)
    configured = required is None or bool(os.getenv(required, "").strip())
    vision_supported = normalized in {"gemini", "ollama"} or not use_vision
    return {
        "provider": normalized,
        "configured": configured,
        "vision_supported": vision_supported,
        "status": "ready" if configured and vision_supported else "configuration_required",
        "checks": {
            "credential": "configured" if configured else "missing",
            "vision": "supported" if vision_supported else "unsupported",
            "endpoint": "not_probed",
            "model": "not_probed",
        },
    }


def create_llm_client(
    provider: str,
    model: str,
    *,
    temperature: float = 0,
    max_tokens: int = 512,
    base_url: str | None = None,
    key_env: str | None = None,
    timeout: float = 60,
    retry_count: int = 2,
) -> LLMClient:
    provider = provider.lower()
    explicit_key = os.getenv(key_env, "") if key_env else None
    if provider == "mock":
        return MockLLM(model_id=model)
    if provider in {"openai", "openai-compatible"}:
        return OpenAICompatibleClient(
            model,
            base_url=base_url,
            api_key=explicit_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_count=retry_count,
        )
    if provider == "anthropic":
        return AnthropicClient(
            model,
            api_key=explicit_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_count=retry_count,
        )
    if provider == "gemini":
        return GeminiClient(
            model,
            api_key=explicit_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_count=retry_count,
        )
    if provider == "ollama":
        return OllamaClient(model, base_url=base_url, timeout=timeout, retry_count=retry_count)
    raise LLMError(f"Unsupported LLM provider: {provider}")
