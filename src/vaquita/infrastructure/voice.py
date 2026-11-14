import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

import requests

from vaquita.core.voice import (
    BaseAudioRecorder,
    BaseAudioTranscriber,
    RecordingError,
    TranscriptionError,
)


class LocalAudioRecorder(BaseAudioRecorder):
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def record(self, duration_seconds: float = 4.0) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            if shutil.which("ffmpeg"):
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "pulse",
                    "-i",
                    "default",
                    "-t",
                    str(duration_seconds),
                    "-ar",
                    str(self.sample_rate),
                    "-ac",
                    "1",
                    "-loglevel",
                    "error",
                    str(tmp_path),
                ]
            elif shutil.which("arecord"):
                cmd = [
                    "arecord",
                    "-q",
                    "-f",
                    "cd",
                    "-t",
                    "wav",
                    "-d",
                    str(max(1, int(round(duration_seconds)))),
                    "-r",
                    str(self.sample_rate),
                    str(tmp_path),
                ]
            else:
                raise RecordingError("No audio capture utility available on system (ffmpeg or arecord required).")

            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=duration_seconds + 5.0,
                check=False,
            )
            if res.returncode != 0:
                err_msg = res.stderr.decode("utf-8", errors="replace").strip()
                raise RecordingError(f"Audio recording failed with exit code {res.returncode}: {err_msg}")

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RecordingError("Captured audio file is empty.")

            return tmp_path.read_bytes()
        except subprocess.TimeoutExpired as exc:
            raise RecordingError("Audio recording timed out.") from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


class GroqAudioTranscriber(BaseAudioTranscriber):
    """Fast speech-to-text transcriber backed by Groq Whisper API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        language: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        resolved_key = api_key or os.getenv("GROQ_API_KEY")
        if not resolved_key:
            raise TranscriptionError("GROQ_API_KEY is not configured for voice transcription.")

        self.api_key: str = resolved_key
        self.model: str = model or os.getenv("GROQ_WHISPER_MODEL") or "whisper-large-v3-turbo"
        resolved_base = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.endpoint: str = f"{resolved_base}/audio/transcriptions"
        self.language: str | None = language or os.getenv("VAQUITA_VOICE_LANG")
        self.timeout_seconds = timeout_seconds

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        if not audio_bytes:
            return ""

        ext = "wav" if "wav" in mime_type else "ogg"
        data: dict[str, str] = {
            "model": self.model,
            "response_format": "json",
            "temperature": "0",
        }
        if self.language:
            data["language"] = self.language

        try:
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (f"audio.{ext}", audio_bytes, mime_type)},
                data=data,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TranscriptionError(f"Groq transcription request failed: {exc}") from exc

        if response.status_code >= 400:
            raise TranscriptionError(
                f"Groq transcription API error ({response.status_code}): {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise TranscriptionError(f"Invalid JSON from Groq transcription API: {exc}") from exc

        return str(payload.get("text", "")).strip()


class VoskAudioTranscriber(BaseAudioTranscriber):
    """Offline speech-to-text transcriber powered by local Vosk model."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        from vaquita.core.voice import WakeWordModelError
        from vaquita.infrastructure.vosk_wake import ensure_vosk_model, load_vosk_module

        try:
            vosk = load_vosk_module()
            resolved = ensure_vosk_model(model_path=model_path)
        except WakeWordModelError as exc:
            raise TranscriptionError(str(exc)) from exc

        self._vosk = vosk
        self._model = vosk.Model(str(resolved))

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        if not audio_bytes:
            return ""

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                framerate = float(wf.getframerate())
                pcm_data = wf.readframes(wf.getnframes())
        except Exception as exc:
            raise TranscriptionError(f"Invalid WAV audio for Vosk transcription: {exc}") from exc

        if not pcm_data:
            return ""

        rec = self._vosk.KaldiRecognizer(self._model, framerate)
        rec.AcceptWaveform(pcm_data)
        try:
            payload = json.loads(rec.FinalResult())
        except json.JSONDecodeError as exc:
            raise TranscriptionError(f"Invalid Vosk transcription output: {exc}") from exc
        return str(payload.get("text", "")).strip()
