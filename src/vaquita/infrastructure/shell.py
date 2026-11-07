from dataclasses import dataclass
import os
import signal
import subprocess


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def execute_command(command: str, timeout: float = 30.0) -> CommandResult:
    try:
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return CommandResult(
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            stderr_msg = f"Command timed out after {timeout} seconds: {stderr or ''}".strip()
            return CommandResult(
                exit_code=-1,
                stdout=stdout or "",
                stderr=stderr_msg,
                timed_out=True,
            )
    except OSError as exc:
        return CommandResult(
            exit_code=-1,
            stdout="",
            stderr=f"Execution failed due to OS error: {exc}",
            timed_out=False,
        )
