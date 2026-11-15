from unittest.mock import MagicMock, patch

import pytest
import requests

from vaquita.core.providers import (
    BaseLLMProvider,
    Message,
    ProviderAuthError,
    ProviderError,
    ToolCall,
    ToolDefinition,
)
from vaquita.infrastructure.openai_compat import OpenAICompatibleProvider


def test_provider_satisfies_protocol() -> None:
    provider = OpenAICompatibleProvider(api_key="dummy_key")
    assert isinstance(provider, BaseLLMProvider)


def test_provider_missing_key_raises_auth_error() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderAuthError, match="API key"):
            OpenAICompatibleProvider()


def test_provider_generate_text() -> None:
    provider = OpenAICompatibleProvider(api_key="test_key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "All systems operational",
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        resp = provider.generate([Message(role="user", content="status")])
        assert resp.text == "All systems operational"
        assert resp.tool_calls == []
        mock_post.assert_called_once()


def test_provider_generate_tool_call() -> None:
    provider = OpenAICompatibleProvider(api_key="test_key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "free -h"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    tools = [
        ToolDefinition(
            name="bash",
            description="Run command",
            parameters={"type": "object"},
        )
    ]

    with patch("requests.post", return_value=mock_response):
        resp = provider.generate([Message(role="user", content="check ram")], tools=tools)
        assert resp.text is None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0] == ToolCall(
            name="bash",
            arguments={"command": "free -h"},
            call_id="call_123",
        )


def test_provider_multi_turn_payload() -> None:
    provider = OpenAICompatibleProvider(api_key="test_key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Done"}}]
    }

    messages = [
        Message(role="system", content="Be concise"),
        Message(role="user", content="Run uptime"),
        Message(
            role="model",
            tool_calls=[ToolCall(name="bash", arguments={"command": "uptime"}, call_id="call_99")],
        ),
        Message(role="tool", name="bash", content="up 2 hours", tool_call_id="call_99"),
    ]

    with patch("requests.post", return_value=mock_response) as mock_post:
        resp = provider.generate(messages)
        assert resp.text == "Done"
        sent_json = mock_post.call_args.kwargs["json"]
        assert len(sent_json["messages"]) == 4
        assert sent_json["messages"][2]["tool_calls"][0]["id"] == "call_99"
        assert sent_json["messages"][3]["role"] == "tool"
        assert sent_json["messages"][3]["tool_call_id"] == "call_99"


def test_provider_auth_error_status_code() -> None:
    provider = OpenAICompatibleProvider(api_key="invalid_key")
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(ProviderAuthError, match="401"):
            provider.generate([Message(role="user", content="hello")])


def test_provider_server_error_status_code() -> None:
    provider = OpenAICompatibleProvider(api_key="test_key")
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(ProviderError, match="500"):
            provider.generate([Message(role="user", content="hello")])


def test_provider_empty_choices_raises_provider_error() -> None:
    provider = OpenAICompatibleProvider(api_key="test_key")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}
    mock_response.text = '{"choices": []}'

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(ProviderError, match="no choices"):
            provider.generate([Message(role="user", content="hello")])


def test_provider_network_exception() -> None:
    provider = OpenAICompatibleProvider(api_key="test_key")

    with patch("requests.post", side_effect=requests.ConnectionError("Connection refused")):
        with pytest.raises(ProviderError, match="HTTP request failed"):
            provider.generate([Message(role="user", content="hello")])


def test_provider_without_api_key_allowed_for_local() -> None:
    provider = OpenAICompatibleProvider(
        api_key="",
        base_url="http://127.0.0.1:8080/v1",
        model="local-model",
        require_api_key=False,
    )
    assert "Authorization" not in provider._build_headers()


def test_provider_generate_stream_tokens_and_tool_calls() -> None:
    provider = OpenAICompatibleProvider(api_key="", require_api_key=False)
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"¿Cómo "}}]}'.encode("utf-8"),
        'data: {"choices":[{"delta":{"content":"estás?"}}]}'.encode("utf-8"),
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"ba","arguments":"{\\"com"}}]}}]}',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"sh","arguments":"mand\\":\\"ls\\"}"}}]}}]}',
        b"data: [DONE]",
    ]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = sse_lines

    tokens: list[str] = []
    with patch("requests.post", return_value=mock_response):
        resp = provider.generate_stream(
            [Message(role="user", content="hola")],
            on_token=tokens.append,
        )

    assert tokens == ["¿Cómo ", "estás?"]
    assert resp.text == "¿Cómo estás?"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "bash"
    assert resp.tool_calls[0].arguments == {"command": "ls"}


def test_provider_optional_sampling_parameters_in_payload() -> None:
    default_provider = OpenAICompatibleProvider(api_key="", require_api_key=False)
    default_payload = default_provider._build_payload([Message(role="user", content="hola")])
    assert "top_p" not in default_payload
    assert "min_p" not in default_payload
    assert "repeat_penalty" not in default_payload

    tuned_provider = OpenAICompatibleProvider(
        api_key="",
        top_p=0.9,
        min_p=0.05,
        repeat_penalty=1.1,
        require_api_key=False,
    )
    tuned_payload = tuned_provider._build_payload([Message(role="user", content="hola")])
    assert tuned_payload["top_p"] == 0.9
    assert tuned_payload["min_p"] == 0.05
    assert tuned_payload["repeat_penalty"] == 1.1
