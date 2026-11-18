from pathlib import Path
import re
from typing import Any

from vaquita.core.providers import ToolDefinition
from vaquita.tools.base import BaseTool

_SKIP_DIRS = frozenset({".git", ".venv", "__pycache__", "node_modules", "target"})
_MAX_OUTPUT_CHARS = 8000
_DEFAULT_LINE_WINDOW = 200


class ReadFileTool(BaseTool):
    """Read-only tool to inspect text files by line range without shell overhead."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="Read a text file from disk with optional start_line and end_line (1-indexed).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-indexed start line (default: 1).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-indexed end line inclusive (optional).",
                    },
                },
                "required": ["path"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        raw_path = str(kwargs.get("path", "")).strip()
        if not raw_path:
            return "Error: path must not be empty."

        target = Path(raw_path).expanduser()
        if not target.exists():
            return f"Error: file '{raw_path}' does not exist."
        if not target.is_file():
            return f"Error: '{raw_path}' is not a regular file."

        start_line = max(1, int(kwargs.get("start_line") or 1))
        raw_end = kwargs.get("end_line")
        end_line = int(raw_end) if raw_end is not None else (start_line + _DEFAULT_LINE_WINDOW - 1)
        if end_line < start_line:
            return "Error: end_line must be >= start_line."

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start_line - 1 : end_line]
        if not selected:
            return f"(empty range: file has {len(lines)} lines)"

        formatted = [f"{idx}: {line}" for idx, line in enumerate(selected, start=start_line)]
        output = "\n".join(formatted)
        if len(output) > _MAX_OUTPUT_CHARS:
            return output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
        return output


class SearchFilesTool(BaseTool):
    """Read-only tool to search for text patterns across files in a directory."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_files",
            description="Search for a text or regex pattern across files in a directory.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text or regex pattern to search for.",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Root directory to search (default: current directory).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines (default: 30).",
                    },
                },
                "required": ["pattern"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        pattern = str(kwargs.get("pattern", "")).strip()
        if not pattern:
            return "Error: pattern must not be empty."

        raw_dir = str(kwargs.get("directory") or ".").strip()
        root = Path(raw_dir).expanduser()
        if not root.is_dir():
            return f"Error: '{raw_dir}' is not a directory."

        max_results = max(1, min(100, int(kwargs.get("max_results") or 30)))
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        matches: list[str] = []
        for path in sorted(root.rglob("*")):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue

            for lineno, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    rel = path.relative_to(root)
                    matches.append(f"{rel}:{lineno}: {line.strip()}")
                    if len(matches) >= max_results:
                        return "\n".join(matches)

        return "\n".join(matches) if matches else "No matches found."
