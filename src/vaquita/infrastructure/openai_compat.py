from collections.abc import Callable
import json
import os
from typing import Any

import requests

from vaquita.core.providers import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ProviderAuthError,
    ProviderError,
    ToolCall,
    ToolDefinition,
)


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for i, raw_call in enumerate(raw_calls):
        function_data = raw_call.get("function", {})
        call_name = str(function_data.get("name", ""))
        raw_arguments = function_data.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
                arguments = parsed if isinstance(parsed, dict) else {}
            except ValueError:
                arguments = {}
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}

        tool_calls.append(
            ToolCall(
                name=call_name,
                arguments=arguments,
                call_id=raw_call.get("id") or f"call_{i}",
            )
        )
    return tool_calls


def _resolve_float_env(env_name: str, default: float | None) -> float | None:
    raw = os.getenv(env_name)
    return float(raw) if raw else default


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "qwen/qwen3.8-27b",
        timeout: float = 30.0,
        max_tokens: int = 800,
        temperature: float = 0.3,
        top_p: float | None = None,
        min_p: float | None = None,
        repeat_penalty: float | None = None,
        require_api_key: bool = True,
    ) -> None:
        resolved_key = api_key if api_key is not None else (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
        if require_api_key and not resolved_key:
            raise ProviderAuthError("API key for OpenAI-compatible provider is not set.")

        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        env_tokens = os.getenv("VAQUITA_MAX_TOKENS")
        self.max_tokens = int(env_tokens) if env_tokens else max_tokens
        env_temp = os.getenv("VAQUITA_TEMPERATURE")
        self.temperature = float(env_temp) if env_temp else temperature
        self.top_p = _resolve_float_env("VAQUITA_TOP_P", top_p)
        self.min_p = _resolve_float_env("VAQUITA_MIN_P", min_p)
        self.repeat_penalty = _resolve_float_env("VAQUITA_REPEAT_PENALTY", repeat_penalty)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role in ("model", "assistant"):
                entry: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    entry["content"] = msg.content
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.call_id or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ]
                payload_messages.append(entry)
            elif msg.role == "tool":
                payload_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "call_0",
                        "content": msg.content or "",
                    }
                )
            else:
                payload_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content or "",
                    }
                )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.repeat_penalty is not None:
            payload["repeat_penalty"] = self.repeat_penalty
        if stream:
            payload["stream"] = True

        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        return payload

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"Authentication failed ({response.status_code}): {response.text}"
            )
        if response.status_code >= 400:
            error_msg = response.text
            try:
                err_json = response.json()
                if isinstance(err_json, dict) and "error" in err_json:
                    err_obj = err_json["error"]
                    if isinstance(err_obj, dict) and "message" in err_obj:
                        error_msg = str(err_obj["message"])
            except ValueError:
                pass
            raise ProviderError(f"API error ({response.status_code}): {error_msg}")

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        endpoint = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                endpoint,
                headers=self._build_headers(),
                json=self._build_payload(messages, tools=tools, stream=False),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"HTTP request failed: {exc}") from exc

        self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"Invalid JSON response: {response.text}") from exc

        choices = data.get("choices")
        if not choices:
            raise ProviderError(f"API returned no choices: {response.text}")

        choice = choices[0].get("message", {})
        text_content = choice.get("content")
        tool_calls = _parse_tool_calls(choice.get("tool_calls", []))
        return LLMResponse(text=text_content, tool_calls=tool_calls)

    def generate_stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        endpoint = f"{self.base_url}/chat/completions"
        try:
            response = requests.post(
                endpoint,
                headers=self._build_headers(),
                json=self._build_payload(messages, tools=tools, stream=True),
                timeout=self.timeout,
                stream=True,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"HTTP request failed: {exc}") from exc

        self._raise_for_status(response)

        text_parts: list[str] = []
        accumulated_calls: dict[int, dict[str, Any]] = {}

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            decoded = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
            line = decoded.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except ValueError:
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            content_piece = delta.get("content")
            if isinstance(content_piece, str) and content_piece:
                text_parts.append(content_piece)
                if on_token is not None:
                    on_token(content_piece)

            for tc_delta in delta.get("tool_calls") or []:
                idx = int(tc_delta.get("index", 0))
                slot = accumulated_calls.setdefault(
                    idx,
                    {"id": None, "function": {"name": "", "arguments": ""}},
                )
                if tc_delta.get("id"):
                    slot["id"] = tc_delta["id"]
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += str(fn["name"])
                if fn.get("arguments"):
                    slot["function"]["arguments"] += str(fn["arguments"])

        full_text = "".join(text_parts) if text_parts else None
        ordered_raw_calls = [accumulated_calls[k] for k in sorted(accumulated_calls)]
        tool_calls = _parse_tool_calls(ordered_raw_calls)
        return LLMResponse(text=full_text, tool_calls=tool_calls)
