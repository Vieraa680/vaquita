from unittest.mock import MagicMock, patch
import pytest

from vaquita.core.providers import ProviderAuthError, ProviderError, ToolCall
from vaquita.core.router import IntentRoute
from vaquita.interface.cli import (
    _confirm_command,
    _log_intent,
    _log_tool_call,
    build_agent,
    load_configuration,
    main,
)


def test_log_tool_call(capsys: pytest.CaptureFixture[str]) -> None:
    call = ToolCall(name="bash", arguments={"command": "uptime"})
    _log_tool_call(call)
    captured = capsys.readouterr()
    assert "[bash] uptime" in captured.out


def test_log_intent(capsys: pytest.CaptureFixture[str]) -> None:
    confident = IntentRoute(intent="terminal", confidence=0.95, is_confident=True)
    _log_intent(confident)
    assert "[router] terminal (95%)" in capsys.readouterr().out

    unknown = IntentRoute(intent="unknown", confidence=0.4, is_confident=False)
    _log_intent(unknown)
    assert capsys.readouterr().out == ""


def test_main_missing_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("vaquita.interface.cli.build_agent", side_effect=ProviderAuthError("Key is missing")):
        code = main(["test prompt"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Key is missing" in captured.err


def test_main_provider_flag(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "Fast response"

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent) as mock_builder:
        code = main(["--provider", "groq", "test prompt"])
        assert code == 0
        mock_builder.assert_called_once_with(
            provider_name="groq",
            auto_approve=False,
            enable_router=True,
        )
        assert "Fast response" in capsys.readouterr().out


def test_main_yes_flag() -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "Done"

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent) as mock_builder:
        code = main(["-y", "do action"])
        assert code == 0
        mock_builder.assert_called_once_with(
            provider_name=None,
            auto_approve=True,
            enable_router=True,
        )


def test_main_no_router_flag() -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "Done"

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent) as mock_builder:
        code = main(["--no-router", "do action"])
        assert code == 0
        mock_builder.assert_called_once_with(
            provider_name=None,
            auto_approve=False,
            enable_router=False,
        )


def test_confirm_command_input() -> None:
    with patch("builtins.input", side_effect=["y", "si", "sí", "n", ""]):
        assert _confirm_command("rm a")
        assert _confirm_command("rm b")
        assert _confirm_command("rm c")
        assert not _confirm_command("rm d")
        assert not _confirm_command("rm e")


def test_confirm_command_interrupt() -> None:
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert not _confirm_command("rm f")


def test_main_direct_prompt_success(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "System is operational"

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent):
        code = main(["check", "system"])
        assert code == 0
        mock_agent.run.assert_called_once_with("check system")
        captured = capsys.readouterr()
        assert "System is operational" in captured.out


def test_main_direct_prompt_with_flags(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = "Files listed"

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent):
        code = main(["ls", "-la", "/tmp"])
        assert code == 0
        mock_agent.run.assert_called_once_with("ls -la /tmp")
        captured = capsys.readouterr()
        assert "Files listed" in captured.out


def test_main_direct_prompt_provider_error(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()
    mock_agent.run.side_effect = ProviderError("API failure")

    with patch("vaquita.interface.cli.build_agent", return_value=mock_agent):
        code = main(["fail"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error: API failure" in captured.err


def test_build_agent_groq_missing_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderAuthError, match="GROQ_API_KEY"):
            build_agent("groq")


def test_build_agent_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider 'unsupported'"):
        build_agent("unsupported")


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


def test_main_interactive_clear_command(capsys: pytest.CaptureFixture[str]) -> None:
    mock_agent = MagicMock()

    with (
        patch("vaquita.interface.cli.build_agent", return_value=mock_agent),
        patch("builtins.input", side_effect=["clear", "exit"]),
    ):
        code = main([])
        assert code == 0
        mock_agent.reset.assert_called_once()
        captured = capsys.readouterr()
        assert "Session history cleared." in captured.out


def test_load_configuration_calls() -> None:
    with patch("vaquita.interface.cli.load_dotenv") as mock_load:
        load_configuration()
        assert mock_load.call_count >= 1
