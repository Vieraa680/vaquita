from unittest.mock import MagicMock

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import (
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)
from vaquita.core.router import IntentRoute
from vaquita.tools.base import BaseTool


class DummyTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="test_tool",
            description="Dummy tool for testing",
            parameters={"type": "object"},
        )

    def execute(self, **kwargs: object) -> str:
        return "dummy result"


def test_agent_direct_response() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(text="Direct answer")

    agent = VaquitaAgent(provider=mock_provider)
    result = agent.run("hello")

    assert result == "Direct answer"
    mock_provider.generate.assert_called_once()


def test_agent_executes_tool_flow() -> None:
    mock_provider = MagicMock()
    tool_call = ToolCall(name="test_tool", arguments={})
    first_response = LLMResponse(text=None, tool_calls=[tool_call])
    second_response = LLMResponse(text="Final summarized answer", tool_calls=[])

    mock_provider.generate.side_effect = [first_response, second_response]

    recorded_calls: list[ToolCall] = []
    agent = VaquitaAgent(
        provider=mock_provider,
        tools=[DummyTool()],
        on_tool_execute=recorded_calls.append,
    )
    result = agent.run("do action")

    assert result == "Final summarized answer"
    assert len(recorded_calls) == 1
    assert recorded_calls[0].name == "test_tool"
    assert mock_provider.generate.call_count == 2


def test_agent_handles_unregistered_tool() -> None:
    mock_provider = MagicMock()
    unregistered_call = ToolCall(name="missing_tool", arguments={})
    first_response = LLMResponse(text=None, tool_calls=[unregistered_call])
    second_response = LLMResponse(text="Recovered from missing tool", tool_calls=[])

    mock_provider.generate.side_effect = [first_response, second_response]

    agent = VaquitaAgent(provider=mock_provider, tools=[])
    result = agent.run("call unknown")

    assert result == "Recovered from missing tool"
    assert mock_provider.generate.call_count == 2


def test_agent_iteration_limit() -> None:
    mock_provider = MagicMock()
    loop_call = ToolCall(name="test_tool", arguments={})
    mock_provider.generate.return_value = LLMResponse(text=None, tool_calls=[loop_call])

    agent = VaquitaAgent(
        provider=mock_provider,
        tools=[DummyTool()],
        max_iterations=3,
    )
    result = agent.run("infinite loop")

    assert "Maximum iteration limit reached" in result
    assert mock_provider.generate.call_count == 3


def test_agent_preserves_history_across_turns() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        LLMResponse(text="response one"),
        LLMResponse(text="response two"),
    ]

    agent = VaquitaAgent(provider=mock_provider)
    agent.run("query one")
    agent.run("query two")

    assert mock_provider.generate.call_count == 2
    second_call_messages = mock_provider.generate.call_args_list[1].args[0]
    roles = [m.role for m in second_call_messages]
    assert roles == ["system", "user", "model", "user"]
    assert second_call_messages[1].content == "query one"
    assert second_call_messages[2].content == "response one"
    assert second_call_messages[3].content == "query two"


def test_agent_reset() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(text="answer")

    agent = VaquitaAgent(provider=mock_provider)
    agent.run("initial query")
    assert len(agent.history) == 3

    agent.reset()
    assert len(agent.history) == 1
    assert agent.history[0].role == "system"


def test_agent_history_trimming() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(text="ok")

    agent = VaquitaAgent(provider=mock_provider, max_history_messages=4)
    for i in range(5):
        agent.run(f"turn {i}")

    # 1 system message + at most 4 recent non-system messages
    assert len(agent.history) == 5
    assert agent.history[0].role == "system"
    assert agent.history[-1].role == "model"


def test_agent_trimming_preserves_atomic_user_turn() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(text="answer")

    agent = VaquitaAgent(provider=mock_provider, max_history_messages=4)
    agent._history.extend(
        [
            Message(role="user", content="first query"),
            Message(role="model", tool_calls=[ToolCall(name="bash", arguments={"command": "ls"})]),
            Message(role="tool", name="bash", content="files"),
            Message(role="model", content="files listed"),
            Message(role="user", content="second query"),
            Message(role="model", content="answer 2"),
        ]
    )

    agent._trim_history()

    assert agent.history[0].role == "system"
    assert agent.history[1].role == "user"
    assert agent.history[1].content == "second query"


def test_agent_with_confident_router_enriches_prompt_and_notifies() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(text="Terminal handled")

    mock_router = MagicMock()
    mock_router.route.return_value = IntentRoute(
        intent="terminal",
        confidence=0.98,
        is_confident=True,
    )

    recorded_routes: list[IntentRoute] = []
    agent = VaquitaAgent(
        provider=mock_provider,
        router=mock_router,
        on_intent=recorded_routes.append,
    )

    result = agent.run("revisa el disco")
    assert result == "Terminal handled"
    assert len(recorded_routes) == 1
    assert recorded_routes[0].intent == "terminal"
    assert agent.last_route == recorded_routes[0]

    # Verify injected intent guidance into user turn
    user_turn = agent.history[1]
    assert user_turn.role == "user"
    assert user_turn.content == "[Intent: terminal]\nrevisa el disco"

    # Reset clears route state
    agent.reset()
    assert agent.last_route is None


def test_agent_with_unconfident_router_leaves_prompt_raw() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.return_value = LLMResponse(text="General answer")

    mock_router = MagicMock()
    mock_router.route.return_value = IntentRoute(
        intent="unknown",
        confidence=0.25,
        is_confident=False,
    )

    agent = VaquitaAgent(provider=mock_provider, router=mock_router)
    result = agent.run("algo no visto")

    assert result == "General answer"
    user_turn = agent.history[1]
    assert user_turn.content == "algo no visto"

