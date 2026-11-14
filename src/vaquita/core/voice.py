from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class VoiceError(Exception):
    """Base exception for voice and audio processing operations."""


class RecordingError(VoiceError):
    """Raised when audio capture fails."""


class TranscriptionError(VoiceError):
    """Raised when speech transcription fails."""


class WakeWordModelError(VoiceError):
    """Raised when a wake word model cannot be loaded or initialized."""


@dataclass
class WakeWordEvent:
    wake_word: str
    command: str
    raw_transcript: str


@runtime_checkable
class BaseAudioTranscriber(Protocol):
    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        """Transcribe raw audio bytes into text."""
        ...


@runtime_checkable
class BaseAudioRecorder(Protocol):
    def record(self, duration_seconds: float = 4.0) -> bytes:
        """Record audio from default input device and return WAV bytes."""
        ...


@runtime_checkable
class BaseWakeWordDetector(Protocol):
    def accept_audio(self, pcm_bytes: bytes) -> WakeWordEvent | None:
        """Process a 16-bit mono PCM audio chunk and return an event if wake word is detected."""
        ...

    def flush(self) -> WakeWordEvent | None:
        """Finalize any buffered utterance when silence is detected."""
        ...

    def reset(self) -> None:
        """Reset internal recognition state after a detection or silence window."""
        ...


@runtime_checkable
class BaseWakeWordListener(Protocol):
    def start(self, on_wake: Callable[[WakeWordEvent], None]) -> None:
        """Start listening for wake word continuously."""
        ...

    def stop(self) -> None:
        """Stop listening."""
        ...

