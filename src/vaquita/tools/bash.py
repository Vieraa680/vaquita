from collections.abc import Callable
import re
from typing import Any

from vaquita.core.providers import ToolDefinition
from vaquita.infrastructure.shell import execute_command
from vaquita.tools.base import BaseTool

_BLOCKED_PATTERNS = [
    r"\brm\s+.*-(?:[a-zA-Z]*r[a-zA-Z]*f|[a-zA-Z]*f[a-zA-Z]*r)\s+.*(/|~|\$HOME)(\s|$|\*)",
    r"\brm\s+-[a-zA-Z]*\s+-[a-zA-Z]*\s+.*(/|~|\$HOME)(\s|$|\*)",
    r"\bmkfs\b",
    r"\bdd\s+if=.*of=/dev/",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\b(shutdown|reboot|poweroff|init\s+0)\b",
]

_MUTATING_PATTERNS = [
    r"\b(rm|rmdir|unlink)\b",
    r"\b(kill|pkill|killall)\b",
    r"\bsystemctl\s+(start|stop|restart|reload|enable|disable|mask|unmask)\b",
    r"\b(pacman|yay|paru|apt|apt-get|dnf|yum|zypper)\b",
    r"\b(chmod|chown|chgrp)\b",
    r"\b(mv|cp)\b",
    r"\b(useradd|userdel|usermod|groupadd|groupdel)\b",
    r"\b(iptables|ufw|firewall-cmd)\b",
    r"(?:^|[^<])>(?:[^>]|$)",
    r">>",
    r"\bsed\s+.*-i\b",
    r"\btruncate\b",
]


def is_mutating_command(command: str) -> bool:
    normalized = command.strip()
    return any(re.search(p, normalized) for p in _MUTATING_PATTERNS)


class BashTool(BaseTool):
    def __init__(
        self,
        timeout: float = 30.0,
        max_lines: int = 60,
        max_chars: int = 3000,
        confirm_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.confirm_fn = confirm_fn

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Execute a bash command on the Linux host system and retrieve its output.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact bash command line to run.",
                    }
                },
                "required": ["command"],
            },
        )

    def _is_destructive(self, command: str) -> bool:
        normalized = command.strip()
        for pattern in _BLOCKED_PATTERNS:
            if re.search(pattern, normalized):
                return True
        return False

    def _truncate(self, text: str) -> str:
        clean_text = text.strip()
        if not clean_text:
            return clean_text

        lines = clean_text.splitlines()
        if len(lines) > self.max_lines:
            head_count = max(1, int(self.max_lines * 0.6))
            tail_count = max(1, self.max_lines - head_count)
            omitted = len(lines) - (head_count + tail_count)
            if omitted > 0:
                head = lines[:head_count]
                tail = lines[-tail_count:]
                notice = f"[... {omitted} lines omitted to conserve tokens ...]"
                clean_text = "\n".join(head + [notice] + tail)

        if len(clean_text) > self.max_chars:
            half = max(10, (self.max_chars - 60) // 2)
            notice = "\n[... content truncated to conserve tokens ...]\n"
            clean_text = clean_text[:half] + notice + clean_text[-half:]

        return clean_text

    def execute(self, command: str = "", **kwargs: Any) -> str:
        if not command.strip():
            return "Error: Empty command provided."

        if self._is_destructive(command):
            return "Execution blocked: Potentially destructive host command detected."

        if self.confirm_fn and is_mutating_command(command):
            approved = self.confirm_fn(command)
            if not approved:
                return f"Execution cancelled by user for command: '{command}'"

        result = execute_command(command, timeout=self.timeout)

        if result.timed_out:
            stderr_msg = self._truncate(result.stderr)
            return f"Command timed out after {self.timeout}s: {stderr_msg}"

        stdout_clean = self._truncate(result.stdout)
        stderr_clean = self._truncate(result.stderr)

        if result.exit_code != 0:
            error_details = []
            if stderr_clean:
                error_details.append(f"stderr: {stderr_clean}")
            if stdout_clean:
                error_details.append(f"stdout: {stdout_clean}")
            msg = "\n".join(error_details) or "no error output provided"
            return f"Command failed with exit code {result.exit_code}:\n{msg}"

        return stdout_clean if stdout_clean else "Command executed successfully with no output."
