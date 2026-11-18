import shutil
import subprocess
from typing import Any

from vaquita.core.providers import ToolDefinition
from vaquita.tools.base import BaseTool

_PLAYERCTL_ACTIONS: dict[str, list[str]] = {
    "play": ["playerctl", "play"],
    "pause": ["playerctl", "pause"],
    "play_pause": ["playerctl", "play-pause"],
    "next": ["playerctl", "next"],
    "previous": ["playerctl", "previous"],
    "status": ["playerctl", "status"],
}


def _run_cmd(args: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


class MediaControlTool(BaseTool):
    """Control desktop media playback (playerctl) and audio volume (wpctl/pactl)."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="media_control",
            description=(
                "Control desktop media playback or audio volume on Linux. "
                "Actions: play, pause, play_pause, next, previous, status, "
                "volume_up, volume_down, mute_toggle, volume_set."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "One of: play, pause, play_pause, next, previous, status, "
                            "volume_up, volume_down, mute_toggle, volume_set."
                        ),
                    },
                    "value": {
                        "type": "integer",
                        "description": "Target volume percentage (0-100) for volume_set, or step for volume_up/down (default: 5).",
                    },
                },
                "required": ["action"],
            },
        )

    def _run_playback(self, action: str) -> str:
        if not shutil.which("playerctl"):
            return "Error: 'playerctl' is not installed on this system."
        try:
            code, out, err = _run_cmd(_PLAYERCTL_ACTIONS[action])
        except subprocess.TimeoutExpired:
            return f"Error: playerctl timed out on '{action}'."
        if code != 0:
            return f"Error: {err or f'playerctl exited with code {code}'}"
        return out or f"Media action '{action}' executed."

    def _run_volume(self, action: str, value: int | None) -> str:
        step = max(1, min(50, value if value is not None else 5))

        if shutil.which("wpctl"):
            if action == "volume_up":
                cmd = ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{step}%+"]
            elif action == "volume_down":
                cmd = ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{step}%-"]
            elif action == "mute_toggle":
                cmd = ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"]
            else:
                pct = max(0, min(100, value if value is not None else 50))
                cmd = ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct}%"]
        elif shutil.which("pactl"):
            if action == "volume_up":
                cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{step}%"]
            elif action == "volume_down":
                cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{step}%"]
            elif action == "mute_toggle":
                cmd = ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]
            else:
                pct = max(0, min(100, value if value is not None else 50))
                cmd = ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"]
        else:
            return "Error: neither 'wpctl' nor 'pactl' is installed on this system."

        try:
            code, out, err = _run_cmd(cmd)
        except subprocess.TimeoutExpired:
            return f"Error: volume command timed out on '{action}'."
        if code != 0:
            return f"Error: {err or f'volume command exited with code {code}'}"
        return out or f"Audio action '{action}' executed."

    def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action", "")).strip().lower()
        if not action:
            return "Error: action must not be empty."

        if action in _PLAYERCTL_ACTIONS:
            return self._run_playback(action)

        if action in ("volume_up", "volume_down", "mute_toggle", "volume_set"):
            raw_val = kwargs.get("value")
            value = int(raw_val) if raw_val is not None else None
            return self._run_volume(action, value)

        return f"Error: unsupported media action '{action}'."
