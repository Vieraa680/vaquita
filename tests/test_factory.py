from unittest.mock import MagicMock, patch
import pytest

from vaquita.core.providers import BaseLLMProvider, ProviderAuthError
from vaquita.factory import create_agent, create_provider
from vaquita.infrastructure.gemini import GeminiProvider
from vaquita.infrastructure.openai_compat import OpenAICompatibleProvider


def test_create_provider_gemini() -> None:
    with patch.dict("os.environ", {"GEMINI_API_KEY": "dummy_gemini_key"}):
        provider = create_provider("gemini")
        assert isinstance(provider, GeminiProvider)
        assert isinstance(provider, BaseLLMProvider)


def test_create_provider_groq() -> None:
    with patch.dict("os.environ", {"GROQ_API_KEY": "dummy_groq_key"}):
        provider = create_provider("groq")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.model == "qwen/qwen3.8-27b"


def test_create_provider_nvidia() -> None:
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "dummy_nvidia_key"}):
        provider = create_provider("nvidia")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.model == "meta/llama-3.3-70b-instruct"


def test_create_provider_openai_compat() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "dummy_openai_key"}):
        provider = create_provider("openai_compat")
        assert isinstance(provider, OpenAICompatibleProvider)


def test_create_provider_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown provider 'invalid'"):
        create_provider("invalid")


def test_create_agent_with_custom_provider() -> None:
    mock_provider = MagicMock()
    agent = create_agent(provider=mock_provider, auto_approve=True)
    assert agent._provider is mock_provider
    assert len(agent.tool_definitions) == 1
    assert agent.tool_definitions[0].name == "bash"
    assert agent.history[0].content is not None
    assert "System Environment:" in agent.history[0].content


def test_create_agent_with_explicit_system_prompt() -> None:
    mock_provider = MagicMock()
    custom_prompt = "Custom system instructions."
    agent = create_agent(
        provider=mock_provider,
        system_prompt=custom_prompt,
    )
    assert agent.history[0].content == custom_prompt
