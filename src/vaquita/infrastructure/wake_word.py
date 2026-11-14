from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import io
import math
import re
import shutil
import struct
import subprocess
import sys
import threading
import wave

from vaquita.core.voice import (
    BaseAudioTranscriber,
    BaseWakeWordDetector,
    BaseWakeWordListener,
    RecordingError,
    WakeWordEvent,
)

_WAKE_REGEX = re.compile(
    r"^(?:(?:oye|hey|hola|ok)\s+)?(?:la\s+)?(?:vaquita|paquita|bakita)[,\s]*(.*)$",
    re.IGNORECASE,
)


def parse_wake_word(transcript: str) -> tuple[bool, str, str]:
    cleaned = transcript.strip().strip(".?!,;").lower()
    match = _WAKE_REGEX.match(cleaned)
    if match:
        command = match.group(1).strip()
        return True, "vaquita", command
    return False, "", ""


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


class TranscriberWakeWordDetector(BaseWakeWordDetector):
    """Buffered wake word detector backed by a generic audio transcriber."""

    def __init__(
        self,
        transcriber: BaseAudioTranscriber,
        sample_rate: int = 16000,
        min_chunks: int = 5,
        max_chunks: int = 60,
    ) -> None:
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self._buffer: list[bytes] = []

    def accept_audio(self, pcm_bytes: bytes) -> WakeWordEvent | None:
        if not pcm_bytes:
            return None
        self._buffer.append(pcm_bytes)
        if len(self._buffer) >= self.max_chunks:
            return self.flush()
        return None

    def flush(self) -> WakeWordEvent | None:
        if len(self._buffer) < self.min_chunks:
            self._buffer.clear()
            return None

        raw_pcm = b"".join(self._buffer)
        self._buffer.clear()
        wav_data = pcm_to_wav(raw_pcm, sample_rate=self.sample_rate)
        try:
            transcript = self.transcriber.transcribe(wav_data)
        except Exception:
            return None

        is_wake, wake_word, cmd = parse_wake_word(transcript)
        if not is_wake:
            return None
        return WakeWordEvent(
            wake_word=wake_word,
            command=cmd,
            raw_transcript=transcript,
        )

    def reset(self) -> None:
        self._buffer.clear()


class LocalWakeWordListener(BaseWakeWordListener):
    """Continuous local microphone listener with RMS energy gate and pluggable detector."""

    def __init__(
        self,
        detector: BaseWakeWordDetector | None = None,
        transcriber: BaseAudioTranscriber | None = None,
        sample_rate: int = 16000,
        energy_threshold: float = 300.0,
        silence_timeout_chunks: int = 3,
        max_duration_seconds: float = 6.0,
    ) -> None:
        if detector is not None:
            self.detector: BaseWakeWordDetector = detector
        elif transcriber is not None:
            max_chunks = max(1, int(max_duration_seconds / 0.1))
            self.detector = TranscriberWakeWordDetector(
                transcriber=transcriber,
                sample_rate=sample_rate,
                max_chunks=max_chunks,
            )
        else:
            raise ValueError("Either detector or transcriber must be provided.")

        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_timeout_chunks = silence_timeout_chunks
        self.chunk_size = int(self.sample_rate * 0.1 * 2)
        self.max_chunks = max(1, int(max_duration_seconds / 0.1))

        self._running = False
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._executor: ThreadPoolExecutor | None = None

    def is_running(self) -> bool:
        return self._running

    def parse_wake_word(self, transcript: str) -> tuple[bool, str, str]:
        return parse_wake_word(transcript)

    def _calculate_rms(self, raw_bytes: bytes) -> float:
        if len(raw_bytes) < 2:
            return 0.0
        num_samples = len(raw_bytes) // 2
        try:
            samples = struct.unpack(f"<{num_samples}h", raw_bytes)
            sum_squares = sum(s * s for s in samples)
            return math.sqrt(sum_squares / num_samples)
        except struct.error:
            return 0.0

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        return pcm_to_wav(pcm_bytes, sample_rate=self.sample_rate)

    def _spawn_capture_process(self) -> subprocess.Popen[bytes]:
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "pulse",
                "-i",
                "default",
                "-f",
                "s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                "1",
                "-loglevel",
                "error",
                "pipe:1",
            ]
        elif shutil.which("arecord"):
            cmd = [
                "arecord",
                "-q",
                "-r",
                str(self.sample_rate),
                "-c",
                "1",
                "-t",
                "raw",
                "-f",
                "S16_LE",
            ]
        else:
            raise RecordingError("No audio capture program (ffmpeg or arecord) available for wake word detection.")

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def _dispatch_event(
        self,
        on_wake: Callable[[WakeWordEvent], None],
        event: WakeWordEvent,
    ) -> None:
        if self._executor is not None:
            self._executor.submit(self._safe_callback, on_wake, event)
        else:
            self._safe_callback(on_wake, event)

    def _safe_callback(
        self,
        on_wake: Callable[[WakeWordEvent], None],
        event: WakeWordEvent,
    ) -> None:
        try:
            on_wake(event)
        except Exception as exc:
            print(f"[wake] Callback error: {exc}", file=sys.stderr)

    def _terminate_process(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=1.0)
        except OSError:
            pass
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        finally:
            self._process = None

    def _loop(self, on_wake: Callable[[WakeWordEvent], None]) -> None:
        try:
            self._process = self._spawn_capture_process()
        except Exception:
            self._running = False
            return

        stdout = self._process.stdout
        if stdout is None:
            self._running = False
            return

        silent_count = 0
        speech_chunks = 0
        in_speech = False

        while self._running:
            chunk = stdout.read(self.chunk_size)
            if not chunk:
                break

            rms = self._calculate_rms(chunk)

            if rms >= self.energy_threshold:
                in_speech = True
                silent_count = 0
                speech_chunks += 1
                event = self.detector.accept_audio(chunk)
                if event is not None:
                    self._dispatch_event(on_wake, event)
                    in_speech = False
                    silent_count = 0
                    speech_chunks = 0
            elif in_speech:
                silent_count += 1
                speech_chunks += 1
                event = self.detector.accept_audio(chunk)
                if event is not None:
                    self._dispatch_event(on_wake, event)
                    in_speech = False
                    silent_count = 0
                    speech_chunks = 0
                    continue

                if silent_count >= self.silence_timeout_chunks or speech_chunks >= self.max_chunks:
                    flushed = self.detector.flush()
                    if flushed is not None:
                        self._dispatch_event(on_wake, flushed)
                    self.detector.reset()
                    in_speech = False
                    silent_count = 0
                    speech_chunks = 0

        self._terminate_process()

    def start(self, on_wake: Callable[[WakeWordEvent], None]) -> None:
        if self._running:
            return
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vaquita-wake-cb")
        self._thread = threading.Thread(
            target=self._loop,
            args=(on_wake,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._terminate_process()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
