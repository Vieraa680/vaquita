from collections.abc import Callable, Sequence
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
import urllib.request
import zipfile

from vaquita.core.voice import (
    BaseWakeWordDetector,
    WakeWordEvent,
    WakeWordModelError,
)

DEFAULT_VOSK_MODEL_NAME = "vosk-model-small-es-0.42"
DEFAULT_VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{DEFAULT_VOSK_MODEL_NAME}.zip"
DEFAULT_MODELS_DIR = Path.home() / ".local" / "share" / "vaquita" / "models"

DEFAULT_WAKE_GRAMMAR: tuple[str, ...] = (
    "vaquita",
    "hola",
    "oye",
    "hey",
    "ok",
    "que",
    "si",
    "no",
    "bueno",
    "el",
    "la",
    "[unk]",
)

_WAKE_TOKEN = "vaquita"


def load_vosk_module() -> Any:
    """Import vosk lazily and suppress verbose Kaldi C++ logging."""
    try:
        import vosk  # type: ignore[import-untyped]
    except ImportError as exc:
        raise WakeWordModelError("vosk package is not installed.") from exc
    vosk.SetLogLevel(-1)
    return vosk


def ensure_vosk_model(
    model_path: Path | str | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
    model_name: str = DEFAULT_VOSK_MODEL_NAME,
    model_url: str = DEFAULT_VOSK_MODEL_URL,
) -> Path:
    """Resolve an existing local Vosk model directory or download the small Spanish model."""
    if model_path is not None:
        explicit_path = Path(model_path).expanduser()
        if not explicit_path.is_dir():
            raise WakeWordModelError(f"Vosk model directory not found: {explicit_path}")
        return explicit_path

    target_dir = models_dir / model_name
    if target_dir.is_dir():
        return target_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vaquita] Descargando modelo offline Vosk ({model_name}, ~40 MB) en {target_dir}...")

    tmp_zip: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=models_dir, suffix=".zip", delete=False) as tmp:
            tmp_zip = Path(tmp.name)

        urllib.request.urlretrieve(model_url, tmp_zip)
        with zipfile.ZipFile(tmp_zip, "r") as archive:
            archive.extractall(models_dir)
    except Exception as exc:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise WakeWordModelError(
            f"Failed to download Vosk model from {model_url}: {exc}"
        ) from exc
    finally:
        if tmp_zip is not None and tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)

    if not target_dir.is_dir():
        raise WakeWordModelError(
            f"Downloaded archive did not contain expected directory: {target_dir}"
        )
    return target_dir


def _extract_wake_phrase(text: str) -> str | None:
    tokens = [tok for tok in text.strip().lower().split() if tok != "[unk]"]
    if _WAKE_TOKEN in tokens:
        return " ".join(tokens)
    return None


class VoskWakeWordDetector(BaseWakeWordDetector):
    """Offline streaming wake word detector powered by Vosk KaldiRecognizer with restricted grammar."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        sample_rate: int = 16000,
        grammar: Sequence[str] = DEFAULT_WAKE_GRAMMAR,
        cooldown_seconds: float = 1.5,
        recognizer_factory: Callable[[Path, int, str], Any] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.cooldown_seconds = cooldown_seconds
        self._last_trigger = 0.0
        grammar_json = json.dumps(list(grammar))

        if recognizer_factory is not None:
            resolved_path = Path(model_path) if model_path is not None else Path(".")
            self._recognizer: Any = recognizer_factory(resolved_path, sample_rate, grammar_json)
            return

        vosk = load_vosk_module()
        resolved_model_path = ensure_vosk_model(model_path=model_path)
        model = vosk.Model(str(resolved_model_path))
        self._recognizer = vosk.KaldiRecognizer(model, float(sample_rate), grammar_json)

    def _evaluate_result(self, raw_json: str, field: str, completed: bool) -> WakeWordEvent | None:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        candidate = str(payload.get(field, ""))
        matched = _extract_wake_phrase(candidate)
        if matched is None:
            return None

        now = time.monotonic()
        if now - self._last_trigger < self.cooldown_seconds:
            if completed:
                self.reset()
            return None

        self._last_trigger = now
        self.reset()
        return WakeWordEvent(
            wake_word="vaquita",
            command="",
            raw_transcript=matched,
        )

    def accept_audio(self, pcm_bytes: bytes) -> WakeWordEvent | None:
        if not pcm_bytes:
            return None

        if not bool(self._recognizer.AcceptWaveform(pcm_bytes)):
            return None
        return self._evaluate_result(self._recognizer.Result(), "text", True)

    def flush(self) -> WakeWordEvent | None:
        raw_json = self._recognizer.FinalResult()
        event = self._evaluate_result(raw_json, "text", True)
        if event is None:
            self.reset()
        return event

    def reset(self) -> None:
        self._recognizer.Reset()
