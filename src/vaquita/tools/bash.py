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


class BashTool(BaseTool):
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

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

    def execute(self, command: str = "", **kwargs: Any) -> str:
        if not command.strip():
            return "Error: Empty command provided."

        if self._is_destructive(command):
            return "Execution blocked: Potentially destructive host command detected."

        result = execute_command(command, timeout=self.timeout)

        if result.timed_out:
            return f"Command timed out after {self.timeout}s: {result.stderr}"

        if result.exit_code != 0:
            error_details = []
            if result.stderr.strip():
                error_details.append(f"stderr: {result.stderr.strip()}")
            if result.stdout.strip():
                error_details.append(f"stdout: {result.stdout.strip()}")
            msg = "\n".join(error_details) or "no error output provided"
            return f"Command failed with exit code {result.exit_code}:\n{msg}"

        output = result.stdout.strip()
        return output if output else "Command executed successfully with no output."
