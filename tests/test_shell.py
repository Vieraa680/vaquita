from vaquita.infrastructure.shell import execute_command


def test_execute_command_success() -> None:
    res = execute_command("echo 'vaquita test'")
    assert res.exit_code == 0
    assert res.stdout.strip() == "vaquita test"
    assert res.stderr == ""
    assert not res.timed_out


def test_execute_command_failure() -> None:
    res = execute_command("exit 42")
    assert res.exit_code == 42
    assert not res.timed_out


def test_execute_command_stderr() -> None:
    res = execute_command("echo 'error message' >&2; exit 1")
    assert res.exit_code == 1
    assert "error message" in res.stderr
    assert not res.timed_out


def test_execute_command_timeout() -> None:
    res = execute_command("sleep 1", timeout=0.05)
    assert res.exit_code == -1
    assert res.timed_out
    assert "timed out" in res.stderr
