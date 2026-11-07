from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def execute_command(command: str, timeout: float = 30.0) -> CommandResult:
    try:
        process = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            exit_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_str = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
        stderr_str = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
        return CommandResult(
            exit_code=-1,
            stdout=stdout_str,
            stderr=f"Command timed out after {timeout} seconds: {stderr_str}".strip(),
            timed_out=True,
        )
    except OSError as exc:
        return CommandResult(
            exit_code=-1,
            stdout="",
            stderr=f"Execution failed due to OS error: {exc}",
            timed_out=False,
        )
