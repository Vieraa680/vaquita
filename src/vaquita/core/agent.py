from collections.abc import Callable, Sequence

from vaquita.core.fast_path import BaseFastPath
from vaquita.core.providers import (
    BaseLLMProvider,
    Message,
    StreamingLLMProvider,
    ToolCall,
    ToolDefinition,
)
from vaquita.core.router import BaseIntentRouter, IntentRoute
from vaquita.tools.base import BaseTool

DEFAULT_SYSTEM_PROMPT = (
    "Tu nombre es Vaquita. Eres un asistente personal conciso y operativo. "
    "Responde en el idioma del usuario, de forma breve y directa. "
    "Usa herramientas solo cuando sea necesario y nunca ejecutes comandos destructivos."
)


def build_system_prompt(
    host_context: str | None = None,
    memory_context: str | None = None,
    current_date: str | None = None,
) -> str:
    parts = [DEFAULT_SYSTEM_PROMPT]
    if current_date and current_date.strip():
        parts.append(f"Current Date: {current_date.strip()}")
    if host_context and host_context.strip():
        parts.append(host_context.strip())
    if memory_context and memory_context.strip():
        parts.append(memory_context.strip())
    return "\n\n".join(parts)


_MEMORY_CUES = (
    "recuerda",
    "recorda",
    "recordar",
    "acordate",
    "acordar",
    "olvides",
    "memoria",
    "guarda",
    "me llamo",
    "prefiero",
    "soy ",
)
_MEMORY_TOOL_NAMES = frozenset({"remember_fact", "save_episode", "recall_memory"})


class VaquitaAgent:
    def __init__(
        self,
        provider: BaseLLMProvider,
        tools: Sequence[BaseTool] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 5,
        max_history_messages: int = 20,
        on_tool_execute: Callable[[ToolCall], None] | None = None,
        router: BaseIntentRouter | None = None,
        on_intent: Callable[[IntentRoute], None] | None = None,
        fast_path: BaseFastPath | None = None,
    ) -> None:
        self._provider = provider
        self._tools_map = {tool.definition.name: tool for tool in (tools or [])}
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        self._on_tool_execute = on_tool_execute
        self._router = router
        self._on_intent = on_intent
        self._fast_path = fast_path
        self._last_route: IntentRoute | None = None
        self._history: list[Message] = []
        self.reset()

    def reset(self) -> None:
        self._history = []
        self._last_route = None
        if self._system_prompt.strip():
            self._history.append(Message(role="system", content=self._system_prompt))

    @property
    def last_route(self) -> IntentRoute | None:
        return self._last_route

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

    def _select_tools(
        self,
        user_input: str,
        route: IntentRoute | None,
    ) -> list[ToolDefinition] | None:
        if not self._tools_map:
            return None

        if route is not None and route.is_confident:
            lowered = user_input.lower()
            has_memory_cue = any(cue in lowered for cue in _MEMORY_CUES)

            if route.intent == "charla":
                if not has_memory_cue:
                    return None
                memory_tools = [
                    t.definition
                    for t in self._tools_map.values()
                    if t.definition.name in _MEMORY_TOOL_NAMES
                ]
                return memory_tools if memory_tools else self.tool_definitions

            if route.intent == "terminal" and not has_memory_cue:
                terminal_tools = [
                    t.definition
                    for t in self._tools_map.values()
                    if t.definition.name not in _MEMORY_TOOL_NAMES
                ]
                return terminal_tools or None

        return self.tool_definitions

    def run(self, user_input: str) -> str:
        return self.run_stream(user_input, on_token=None)

    def run_stream(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        if self._router is not None:
            self._last_route = self._router.route(user_input)
            if self._on_intent:
                self._on_intent(self._last_route)
        else:
            self._last_route = None

        self._history.append(Message(role="user", content=user_input))

        if self._fast_path is not None:
            fast_reply = self._fast_path.try_handle(user_input, self._last_route)
            if fast_reply is not None:
                self._history.append(Message(role="model", content=fast_reply))
                self._trim_history()
                if on_token is not None:
                    on_token(fast_reply)
                return fast_reply

        self._trim_history()
        tool_defs = self._select_tools(user_input, self._last_route)

        for iteration in range(self._max_iterations):
            if on_token is not None and isinstance(self._provider, StreamingLLMProvider):
                response = self._provider.generate_stream(
                    list(self._history),
                    tools=tool_defs,
                    on_token=on_token,
                )
            else:
                response = self._provider.generate(list(self._history), tools=tool_defs)
                if on_token is not None and response.text and not response.tool_calls:
                    on_token(response.text)

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
