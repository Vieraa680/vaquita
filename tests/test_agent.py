from unittest.mock import MagicMock

from vaquita.core.agent import VaquitaAgent
from vaquita.core.providers import LLMResponse, ToolCall, ToolDefinition
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
