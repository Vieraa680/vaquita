from subprocess import CompletedProcess
from unittest.mock import patch

from vaquita.tools.media import MediaControlTool


def test_media_control_playback_and_missing_binary() -> None:
    tool = MediaControlTool()

    with patch("vaquita.tools.media.shutil.which", return_value=None):
        assert "playerctl" in tool.execute(action="play_pause")

    with (
        patch("vaquita.tools.media.shutil.which", return_value="/usr/bin/playerctl"),
        patch(
            "vaquita.tools.media.subprocess.run",
            return_value=CompletedProcess(["playerctl", "status"], 0, stdout="Playing\n", stderr=""),
        ) as mock_run,
    ):
        assert tool.execute(action="status") == "Playing"
        mock_run.assert_called_once()


def test_media_control_volume_wpctl_and_invalid_action() -> None:
    tool = MediaControlTool()

    assert tool.execute(action="unknown_cmd").startswith("Error:")

    with (
        patch("vaquita.tools.media.shutil.which", side_effect=lambda name: "/usr/bin/wpctl" if name == "wpctl" else None),
        patch(
            "vaquita.tools.media.subprocess.run",
            return_value=CompletedProcess([], 0, stdout="", stderr=""),
        ) as mock_run,
    ):
        res = tool.execute(action="volume_set", value=40)
        assert "executed" in res
        assert mock_run.call_args.args[0] == ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "40%"]
