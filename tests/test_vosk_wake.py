import json
from pathlib import Path
import struct
import time
from unittest.mock import MagicMock, patch

import pytest

from vaquita.core.voice import BaseWakeWordDetector, WakeWordEvent, WakeWordModelError
from vaquita.factory import create_wake_word_listener
from vaquita.infrastructure.vosk_wake import (
    VoskWakeWordDetector,
    ensure_vosk_model,
)
from vaquita.infrastructure.wake_word import LocalWakeWordListener


def _make_fake_recognizer(
    accept_results: list[bool],
    partial_results: list[str] | None = None,
    final_results: list[str] | None = None,
    final_flush: str = '{"text": ""}',
) -> MagicMock:
    rec = MagicMock()
    rec.AcceptWaveform.side_effect = accept_results
    if partial_results is not None:
        rec.PartialResult.side_effect = partial_results
    else:
        rec.PartialResult.return_value = '{"partial": ""}'
    if final_results is not None:
        rec.Result.side_effect = final_results
    else:
        rec.Result.return_value = '{"text": ""}'
    rec.FinalResult.return_value = final_flush
    return rec


def test_vosk_detector_implements_protocol() -> None:
    fake_rec = _make_fake_recognizer([])
    detector = VoskWakeWordDetector(recognizer_factory=lambda *_: fake_rec)
    assert isinstance(detector, BaseWakeWordDetector)


def test_vosk_detector_triggers_on_completed_result_and_ignores_distractors() -> None:
    fake_rec = _make_fake_recognizer(
        accept_results=[False, True, True, True],
        final_results=[
            json.dumps({"text": "hola"}),
            json.dumps({"text": "hey"}),
            json.dumps({"text": "vaquita"}),
        ],
    )
    detector = VoskWakeWordDetector(recognizer_factory=lambda *_: fake_rec)

    assert detector.accept_audio(b"\x01\x00" * 1600) is None
    assert detector.accept_audio(b"\x01\x00" * 1600) is None
    assert detector.accept_audio(b"\x01\x00" * 1600) is None
    event = detector.accept_audio(b"\x01\x00" * 1600)
    assert event is not None
    assert event.wake_word == "vaquita"
    assert event.raw_transcript == "vaquita"
    fake_rec.Reset.assert_called_once()


def test_vosk_detector_respects_cooldown() -> None:
    fake_rec = _make_fake_recognizer(
        accept_results=[True, True],
        final_results=[
            json.dumps({"text": "vaquita"}),
            json.dumps({"text": "vaquita"}),
        ],
    )
    detector = VoskWakeWordDetector(
        cooldown_seconds=5.0,
        recognizer_factory=lambda *_: fake_rec,
    )

    first = detector.accept_audio(b"\x01\x00" * 1600)
    second = detector.accept_audio(b"\x01\x00" * 1600)
    assert first is not None
    assert second is None


def test_vosk_detector_flush_evaluates_final_result() -> None:
    fake_rec = _make_fake_recognizer(
        accept_results=[],
        final_flush=json.dumps({"text": "hola vaquita"}),
    )
    detector = VoskWakeWordDetector(recognizer_factory=lambda *_: fake_rec)

    event = detector.flush()
    assert event is not None
    assert event.wake_word == "vaquita"
    assert event.raw_transcript == "hola vaquita"
    fake_rec.Reset.assert_called_once()


def test_ensure_vosk_model_explicit_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "custom-model"
    model_dir.mkdir()
    assert ensure_vosk_model(model_path=model_dir) == model_dir

    with pytest.raises(WakeWordModelError, match="not found"):
        ensure_vosk_model(model_path=tmp_path / "missing")


def test_listener_with_streaming_detector_dispatches_async() -> None:
    detector = MagicMock(spec=BaseWakeWordDetector)
    detector.accept_audio.side_effect = [
        None,
        WakeWordEvent(wake_word="vaquita", command="", raw_transcript="vaquita"),
        None,
    ]
    detector.flush.return_value = None

    listener = LocalWakeWordListener(
        detector=detector,
        energy_threshold=200.0,
        silence_timeout_chunks=2,
    )

    loud_chunk = struct.pack("<1600h", *([800] * 1600))
    silent_chunk = struct.pack("<1600h", *([0] * 1600))

    fake_stdout = MagicMock()
    fake_stdout.read.side_effect = [loud_chunk, loud_chunk, silent_chunk, silent_chunk, b""]

    fake_proc = MagicMock()
    fake_proc.stdout = fake_stdout

    events: list[WakeWordEvent] = []
    with patch.object(listener, "_spawn_capture_process", return_value=fake_proc):
        listener.start(events.append)
        deadline = time.monotonic() + 2.0
        while not events and time.monotonic() < deadline:
            time.sleep(0.02)
        listener.stop()

    assert len(events) == 1
    assert events[0].wake_word == "vaquita"


def test_factory_create_wake_word_listener_accepts_custom_detector() -> None:
    custom_detector = MagicMock(spec=BaseWakeWordDetector)
    listener = create_wake_word_listener(detector=custom_detector)
    assert isinstance(listener, LocalWakeWordListener)
    assert listener.detector is custom_detector
