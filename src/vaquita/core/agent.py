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


class VaquitaAgent:
    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 5,
        on_tool_execute: Callable[[ToolCall], None] | None = None,
    ) -> None:
        self._provider = provider
        self._tools_map = {tool.definition.name: tool for tool in (tools or [])}
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._on_tool_execute = on_tool_execute

    @property
    def tool_definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools_map.values()]

    def run(self, user_input: str) -> str:
        messages: list[Message] = []
        if self._system_prompt.strip():
            messages.append(Message(role="system", content=self._system_prompt))

        messages.append(Message(role="user", content=user_input))
        tool_defs = self.tool_definitions or None

        for _ in range(self._max_iterations):
            response = self._provider.generate(messages, tools=tool_defs)

            if not response.tool_calls:
                return response.text or ""

            messages.append(
                Message(
                    role="model",
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
            )

            for call in response.tool_calls:
                if self._on_tool_execute:
                    self._on_tool_execute(call)

                tool = self._tools_map.get(call.name)
                if tool is None:
                    result_content = f"Error: Tool '{call.name}' is not registered."
                else:
                    result_content = tool.execute(**call.arguments)

                messages.append(
                    Message(
                        role="tool",
                        name=call.name,
                        content=result_content,
                        tool_call_id=call.call_id,
                    )
                )

        return "Execution stopped: Maximum iteration limit reached."
