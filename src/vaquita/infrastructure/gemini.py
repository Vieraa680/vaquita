import os
from typing import Any

from google import genai
from google.genai import errors, types

from vaquita.core.providers import (
    LLMResponse,
    Message,
    ProviderAuthError,
    ProviderError,
    ToolCall,
    ToolDefinition,
)


class GeminiProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: genai.Client | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key and client is None:
            raise ProviderAuthError("GEMINI_API_KEY environment variable is not set.")

        self.model: str = model or os.getenv("GEMINI_MODEL") or "gemini-3.1-flash-lite"
        self._client = client or genai.Client(
            api_key=resolved_key,
            http_options={"timeout": 30000},
        )

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        system_instruction: str | None = None
        contents: list[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue

            if msg.role in ("model", "assistant") and msg.raw_content is not None:
                contents.append(msg.raw_content)
                continue

            parts: list[types.Part] = []
            if msg.role in ("user", "human"):
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                contents.append(types.Content(role="user", parts=parts))

            elif msg.role in ("model", "assistant"):
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                for tc in msg.tool_calls:
                    parts.append(
                        types.Part.from_function_call(
                            name=tc.name,
                            args=tc.arguments,
                        )
                    )
                contents.append(types.Content(role="model", parts=parts))

            elif msg.role == "tool":
                parts.append(
                    types.Part.from_function_response(
                        name=msg.name or "tool",
                        response={"output": msg.content or ""},
                    )
                )
                contents.append(types.Content(role="user", parts=parts))

        config_params: dict[str, Any] = {}
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if tools:
            declarations = [
                types.FunctionDeclaration.model_validate(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                )
                for t in tools
            ]
            config_params["tools"] = [types.Tool(function_declarations=declarations)]

        config = types.GenerateContentConfig(**config_params) if config_params else None

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        except errors.APIError as exc:
            raise ProviderError(f"Gemini API error: {exc.message}") from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected error communicating with Gemini: {exc}") from exc

        tool_calls: list[ToolCall] = []
        if response.function_calls:
            for call in response.function_calls:
                tool_calls.append(
                    ToolCall(
                        name=call.name or "",
                        arguments=dict(call.args or {}),
                    )
                )

        text = response.text if not tool_calls else None

        raw_content = None
        if response.candidates and response.candidates[0].content:
            raw_content = response.candidates[0].content

        return LLMResponse(text=text, tool_calls=tool_calls, raw_content=raw_content)
