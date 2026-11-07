from unittest.mock import MagicMock

from vaquita.core.providers import LLMResponse, ToolCall
from vaquita.factory import create_agent


def test_smoke_agent_creation_and_run() -> None:
    mock_provider = MagicMock()
    mock_provider.generate.side_effect = [
        LLMResponse(tool_calls=[ToolCall(name="bash", arguments={"command": "echo 'vaquita online'"})]),
        LLMResponse(text="System ready"),
    ]

    agent = create_agent(provider=mock_provider, auto_approve=True)
    result = agent.run("check readiness")

    assert result == "System ready"
    assert mock_provider.generate.call_count == 2
    assert len(agent.history) >= 4
