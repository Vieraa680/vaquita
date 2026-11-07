from unittest.mock import MagicMock, patch

import pytest

from vaquita.core.providers import (
    BaseLLMProvider,
    Message,
    ProviderAuthError,
    ProviderError,
    ToolCall,
    ToolDefinition,
)
from vaquita.infrastructure.gemini import GeminiProvider


def test_gemini_provider_satisfies_protocol() -> None:
    mock_client = MagicMock()
    provider = GeminiProvider(client=mock_client)
    assert isinstance(provider, BaseLLMProvider)


def test_gemini_provider_missing_key_raises_auth_error() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderAuthError, match="GEMINI_API_KEY"):
            GeminiProvider()


def test_gemini_provider_generate_text() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "System is operational"
    mock_response.function_calls = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)
    messages = [Message(role="user", content="status")]
    response = provider.generate(messages)

    assert response.text == "System is operational"
    assert response.tool_calls == []
    mock_client.models.generate_content.assert_called_once()


def test_gemini_provider_generate_tool_call() -> None:
    mock_client = MagicMock()
    mock_call = MagicMock()
    mock_call.name = "bash"
    mock_call.args = {"command": "df -h"}

    mock_response = MagicMock()
    mock_response.text = None
    mock_response.function_calls = [mock_call]
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)
    tools = [
        ToolDefinition(
            name="bash",
            description="Execute command",
            parameters={"type": "object"},
        )
    ]
    messages = [Message(role="user", content="check disk")]
    response = provider.generate(messages, tools=tools)

    assert response.text is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0] == ToolCall(name="bash", arguments={"command": "df -h"})


def test_gemini_provider_handles_system_and_tool_messages() -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Result processed"
    mock_response.function_calls = None
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)
    messages = [
        Message(role="system", content="Act as terminal assistant"),
        Message(role="user", content="run check"),
        Message(role="model", tool_calls=[ToolCall(name="bash", arguments={"command": "whoami"})]),
        Message(role="tool", name="bash", content="albers"),
    ]
    response = provider.generate(messages)

    assert response.text == "Result processed"
    called_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert called_kwargs["config"].system_instruction == "Act as terminal assistant"
    assert len(called_kwargs["contents"]) == 3


def test_gemini_provider_wraps_api_error() -> None:
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Network timeout")

    provider = GeminiProvider(client=mock_client)
    messages = [Message(role="user", content="hello")]

    with pytest.raises(ProviderError, match="Unexpected error communicating with Gemini"):
        provider.generate(messages)
