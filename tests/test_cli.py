from unittest.mock import MagicMock, patch
import pytest

from vaquita.core.providers import ProviderAuthError, ProviderError, ToolCall
from vaquita.interface.cli import _log_tool_call, main


def test_log_tool_call(capsys: pytest.CaptureFixture[str]) -> None:
    call = ToolCall(name="bash", arguments={"command": "uptime"})
    _log_tool_call(call)
    captured = capsys.readouterr()
    assert "[bash] uptime" in captured.out


def test_main_missing_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("vaquita.interface.cli.build_agent", side_effect=ProviderAuthError("no key")):
        code = main(["test prompt"])
        assert code == 1
        captured = capsys.readouterr()
        assert "GEMINI_API_KEY is not configured" in captured.err


def test_main_direct_prompt_success(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "System is operational"

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent):
        code = main(["check", "system"])
        assert code == 0
        mock_agent.run.assert_called_once_with("check system")
        captured = capsys.readouterr()
        assert "System is operational" in captured.out


def test_main_direct_prompt_provider_error(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()
    mock_agent.run.side_effect = ProviderError("API failure")

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent):
        code = main(["fail"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error: API failure" in captured.err


def test_main_interactive_exit(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()

    with (
        patch("vaquita.interface.cli.build_agent", return_value=mock_agent),
        patch("builtins.input", side_effect=["exit"]),
    ):
        code = main([])
        assert code == 0
        captured = capsys.readouterr()
        assert "Vaquita - Terminal Assistant" in captured.out
        mock_agent.run.assert_not_called()


def test_main_interactive_query_flow(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "All good"

    with (
        patch("vaquita.interface.cli.build_agent", return_value=mock_agent),
        patch("builtins.input", side_effect=["status", "exit"]),
    ):
        code = main([])
        assert code == 0
        mock_agent.run.assert_called_once_with("status")
        captured = capsys.readouterr()
        assert "All good" in captured.out
