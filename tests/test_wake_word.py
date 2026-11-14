import io
import math
import struct
import wave
from unittest.mock import MagicMock, patch

from vaquita.core.voice import (
    BaseAudioTranscriber,
    BaseWakeWordListener,
    WakeWordEvent,
)
from vaquita.infrastructure.wake_word import LocalWakeWordListener


def test_wake_word_protocol_implementation() -> None:
    mock_transcriber = MagicMock(spec=BaseAudioTranscriber)
    listener = LocalWakeWordListener(transcriber=mock_transcriber)
    assert isinstance(listener, BaseWakeWordListener)


def test_parse_wake_word_matches() -> None:
    mock_transcriber = MagicMock(spec=BaseAudioTranscriber)
    listener = LocalWakeWordListener(transcriber=mock_transcriber)

    cases = [
        ("Vaquita", True, "vaquita", ""),
        ("vaquita.", True, "vaquita", ""),
        ("oye vaquita", True, "vaquita", ""),
        ("hey vaquita!", True, "vaquita", ""),
        ("hola vaquita, qué hora es?", True, "vaquita", "qué hora es"),
        ("ok vaquita abre el navegador", True, "vaquita", "abre el navegador"),
        ("paquita", True, "vaquita", ""),
        ("la vaquita", True, "vaquita", ""),
        ("esto no es una llamada", False, "", ""),
        ("hola juan", False, "", ""),
    ]

    for transcript, expected_match, expected_wake, expected_cmd in cases:
        matched, wake, cmd = listener.parse_wake_word(transcript)
        assert matched is expected_match, f"Failed for '{transcript}'"
        if matched:
            assert wake == expected_wake
            assert cmd == expected_cmd


def test_calculate_rms() -> None:
    mock_transcriber = MagicMock(spec=BaseAudioTranscriber)
    listener = LocalWakeWordListener(transcriber=mock_transcriber)

    # Pure silence
    silent_bytes = b"\x00" * 3200
    assert listener._calculate_rms(silent_bytes) == 0.0

    # Loud square wave
    loud_samples = [1000, -1000] * 800
    loud_bytes = struct.pack(f"<{len(loud_samples)}h", *loud_samples)
    rms = listener._calculate_rms(loud_bytes)
    assert math.isclose(rms, 1000.0, rel_tol=1e-3)


def test_pcm_to_wav_validity() -> None:
    mock_transcriber = MagicMock(spec=BaseAudioTranscriber)
    listener = LocalWakeWordListener(transcriber=mock_transcriber, sample_rate=16000)

    pcm_data = b"\x00" * 3200
    wav_data = listener._pcm_to_wav(pcm_data)

    assert wav_data.startswith(b"RIFF")
    buf = io.BytesIO(wav_data)
    with wave.open(buf, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        assert w.getnframes() == 1600


def test_wake_word_listener_flow() -> None:
    mock_transcriber = MagicMock(spec=BaseAudioTranscriber)
    mock_transcriber.transcribe.return_value = "oye vaquita qué hora es"

    listener = LocalWakeWordListener(
        transcriber=mock_transcriber,
        energy_threshold=200.0,
        silence_timeout_chunks=2,
    )

    loud_chunk = struct.pack("<1600h", *([800] * 1600))
    silent_chunk = struct.pack("<1600h", *([0] * 1600))

    # Simulate 6 loud chunks (0.6s speech) followed by 3 silent chunks
    fake_stream_chunks = [loud_chunk] * 6 + [silent_chunk] * 3 + [b""]

    fake_stdout = MagicMock()
    fake_stdout.read.side_effect = fake_stream_chunks

    fake_proc = MagicMock()
    fake_proc.stdout = fake_stdout
    fake_proc.terminate.return_value = None
    fake_proc.wait.return_value = 0

    triggered_events: list[WakeWordEvent] = []

    with patch.object(listener, "_spawn_capture_process", return_value=fake_proc):
        listener._running = True
        listener._loop(lambda evt: triggered_events.append(evt))

    assert len(triggered_events) == 1
    event = triggered_events[0]
    assert event.wake_word == "vaquita"
    assert event.command == "qué hora es"
    assert event.raw_transcript == "oye vaquita qué hora es"
