from collections.abc import Generator
from http import HTTPStatus
import json
from pathlib import Path
import subprocess
from typing import Any
import urllib.request
import pytest
from unittest.mock import MagicMock, patch

from vaquita.core.agent import VaquitaAgent
from vaquita.core.voice import (
    BaseAudioRecorder,
    BaseAudioTranscriber,
    RecordingError,
    TranscriptionError,
    VoiceError,
)
from vaquita.infrastructure.voice import (
    GroqAudioTranscriber,
    LocalAudioRecorder,
    VoskAudioTranscriber,
)
from vaquita.interface.server import VaquitaServer


def test_voice_protocols_and_hierarchy() -> None:
    assert issubclass(RecordingError, VoiceError)
    assert issubclass(TranscriptionError, VoiceError)


def test_local_audio_recorder_ffmpeg_success(tmp_path: Path) -> None:
    recorder = LocalAudioRecorder(sample_rate=16000)

    fake_wav_bytes = b"RIFF____WAVEfmt "

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        out_file = Path(cmd[-1])
        out_file.write_bytes(fake_wav_bytes)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout=b"", stderr=b"")

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch("subprocess.run", side_effect=fake_run):
        audio = recorder.record(duration_seconds=1.0)
        assert audio == fake_wav_bytes


def test_local_audio_recorder_failure_raises_recording_error() -> None:
    recorder = LocalAudioRecorder()

    def fake_run_err(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout=b"", stderr=b"device busy")

    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch("subprocess.run", side_effect=fake_run_err):
        with pytest.raises(RecordingError, match="Audio recording failed with exit code 1"):
            recorder.record(duration_seconds=1.0)


def test_local_audio_recorder_no_tool_raises() -> None:
    recorder = LocalAudioRecorder()
    with patch("shutil.which", return_value=None):
        with pytest.raises(RecordingError, match="No audio capture utility available"):
            recorder.record(duration_seconds=1.0)


def test_groq_audio_transcriber_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "  hola vaquita qué hora es  "}

    transcriber = GroqAudioTranscriber(api_key="test-key")
    with patch("vaquita.infrastructure.voice.requests.post", return_value=mock_resp) as mock_post:
        result = transcriber.transcribe(b"fake_audio_bytes")
        assert result == "hola vaquita qué hora es"
        assert mock_post.called


def test_groq_audio_transcriber_api_error() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "rate limit exceeded"

    transcriber = GroqAudioTranscriber(api_key="test-key")
    with patch("vaquita.infrastructure.voice.requests.post", return_value=mock_resp):
        with pytest.raises(TranscriptionError, match="429"):
            transcriber.transcribe(b"fake_audio_bytes")


def test_groq_audio_transcriber_empty_bytes() -> None:
    transcriber = GroqAudioTranscriber(api_key="test-key")
    assert transcriber.transcribe(b"") == ""


def test_vosk_audio_transcriber_empty_bytes() -> None:
    transcriber = VoskAudioTranscriber()
    assert transcriber.transcribe(b"") == ""


class DummyRecorder(BaseAudioRecorder):
    def record(self, duration_seconds: float = 4.0) -> bytes:
        return b"recorded_dummy_wav"


class DummyTranscriber(BaseAudioTranscriber):
    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        return "reproduce mi música"


@pytest.fixture
def running_voice_server() -> Generator[VaquitaServer, None, None]:
    mock_agent = MagicMock(spec=VaquitaAgent)
    mock_agent.run.return_value = "Reproduciendo tu playlist favorita."
    mock_agent.last_route = None

    server = VaquitaServer(
        agent_factory=lambda: mock_agent,
        host="127.0.0.1",
        port=0,
        recorder=DummyRecorder(),
        transcriber=DummyTranscriber(),
    )
    server.start_background()
    yield server
    server.stop()


def test_server_voice_endpoint_success(running_voice_server: VaquitaServer) -> None:
    url = f"http://127.0.0.1:{running_voice_server.port}/api/voice"
    body = json.dumps({"record_seconds": 1.0}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        assert resp.status == HTTPStatus.OK
        data = json.loads(resp.read().decode("utf-8"))
        assert data["transcribed_prompt"] == "reproduce mi música"
        assert "Reproduciendo" in data["response"]
