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


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 30.0,
    ) -> None:
        resolved_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ProviderAuthError("API key for OpenAI-compatible provider is not set.")

        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
        }

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

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"HTTP request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                f"Authentication failed ({response.status_code}): {response.text}"
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"Invalid JSON response: {response.text}") from exc

        choice = data.get("choices", [{}])[0].get("message", {})
        text_content = choice.get("content")

        tool_calls: list[ToolCall] = []
        for raw_call in choice.get("tool_calls", []):
            function_data = raw_call.get("function", {})
            call_name = function_data.get("name", "")
            raw_arguments = function_data.get("arguments", "{}")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except ValueError:
                    arguments = {}
            else:
                arguments = raw_arguments

            tool_calls.append(
                ToolCall(
                    name=call_name,
                    arguments=arguments,
                    call_id=raw_call.get("id"),
                )
            )

        return LLMResponse(text=text_content, tool_calls=tool_calls)
