from collections.abc import Callable, Sequence

from vaquita.core.providers import (
    BaseLLMProvider,
    Message,
    ToolCall,
    ToolDefinition,
)
from vaquita.tools.base import BaseTool

DEFAULT_SYSTEM_PROMPT = (
    "You are Vaquita, a focused AI assistant for Linux terminal automation. "
    "Execute terminal commands safely, analyze the system accurately, "
    "and provide concise, operational answers. Never execute destructive commands."
)


def build_system_prompt(host_context: str | None = None) -> str:
    if host_context and host_context.strip():
        return f"{DEFAULT_SYSTEM_PROMPT}\n\n{host_context.strip()}"
    return DEFAULT_SYSTEM_PROMPT


class VaquitaAgent:
    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 5,
        max_history_messages: int = 20,
        on_tool_execute: Callable[[ToolCall], None] | None = None,
    ) -> None:
        self._provider = provider
        self._tools_map = {tool.definition.name: tool for tool in (tools or [])}
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        self._on_tool_execute = on_tool_execute
        self._history: list[Message] = []
        self.reset()

    def reset(self) -> None:
        self._history = []
        if self._system_prompt.strip():
            self._history.append(Message(role="system", content=self._system_prompt))

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools_map.values()]

    def _trim_history(self) -> None:
        has_system = bool(self._history and self._history[0].role == "system")
        system_offset = 1 if has_system else 0
        non_system = self._history[system_offset:]

        if len(non_system) <= self._max_history_messages:
            return

        cut_idx = len(non_system) - self._max_history_messages
        while cut_idx < len(non_system) and non_system[cut_idx].role != "user":
            cut_idx += 1

        kept = non_system[cut_idx:]
        if not kept:
            user_indices = [i for i, m in enumerate(non_system) if m.role == "user"]
            if user_indices:
                kept = non_system[user_indices[-1]:]
            else:
                kept = non_system[-1:]

        if has_system:
            self._history = [self._history[0]] + kept
        else:
            self._history = kept

    def run(self, user_input: str) -> str:
        self._history.append(Message(role="user", content=user_input))
        self._trim_history()
        tool_defs = self.tool_definitions or None

        for iteration in range(self._max_iterations):
            response = self._provider.generate(list(self._history), tools=tool_defs)

            if not response.tool_calls:
                self._history.append(
                    Message(
                        role="model",
                        content=response.text,
                        raw_content=response.raw_content,
                    )
                )
                self._trim_history()
                return response.text or ""

            normalized_calls = [
                ToolCall(
                    name=tc.name,
                    arguments=tc.arguments,
                    call_id=tc.call_id or f"call_{iteration}_{idx}",
                )
                for idx, tc in enumerate(response.tool_calls)
            ]

            self._history.append(
                Message(
                    role="model",
                    content=response.text,
                    tool_calls=normalized_calls,
                    raw_content=response.raw_content,
                )
            )

            for call in normalized_calls:
                if self._on_tool_execute:
                    self._on_tool_execute(call)

                tool = self._tools_map.get(call.name)
                if tool is None:
                    result_content = f"Error: Tool '{call.name}' is not registered."
                else:
                    result_content = tool.execute(**call.arguments)

                self._history.append(
                    Message(
                        role="tool",
                        name=call.name,
                        content=result_content,
                        tool_call_id=call.call_id,
                    )
                )

        self._trim_history()
        return "Execution stopped: Maximum iteration limit reached."
