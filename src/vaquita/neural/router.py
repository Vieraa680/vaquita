from collections.abc import Sequence
from importlib.resources import files
import json
from pathlib import Path

from vaquita.core.router import BaseIntentRouter, IntentRoute
from vaquita.neural.intent import FEATURE_SCHEMA_VERSION, IntentClassifier

DEFAULT_MODEL_DIR = Path.home() / ".config" / "vaquita" / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "intent.json"
DEFAULT_DATASET_PATH = DEFAULT_MODEL_DIR / "dataset.json"


def load_bundled_seed_dataset() -> list[tuple[str, str]]:
    data_path = Path(__file__).resolve().parent / "data" / "default_intents.json"
    if not data_path.is_file():
        try:
            resource = files("vaquita.neural.data").joinpath("default_intents.json")
            data_path = Path(str(resource))
        except (ModuleNotFoundError, TypeError):
            return []

    if not data_path.is_file():
        return []

    raw_data = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        return []

    return [
        (str(item["text"]), str(item["intent"]))
        for item in raw_data
        if isinstance(item, dict) and "text" in item and "intent" in item
    ]


SEED_DATASET = load_bundled_seed_dataset()


class NeuralIntentRouter(BaseIntentRouter):
    """Local neural intent router backed by IntentClassifier with persistent continuous learning."""

    def __init__(
        self,
        classifier: IntentClassifier,
        model_path: Path | None = None,
        dataset_path: Path | None = None,
        samples: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        self.classifier = classifier
        self.model_path = model_path
        self.dataset_path = dataset_path
        self._samples: list[tuple[str, str]] = list(
            samples if samples is not None else load_bundled_seed_dataset()
        )

    @property
    def samples(self) -> list[tuple[str, str]]:
        return list(self._samples)

    def route(self, user_input: str) -> IntentRoute:
        prediction = self.classifier.predict(user_input)
        return IntentRoute(
            intent=prediction.label,
            confidence=prediction.confidence,
            is_confident=prediction.is_confident,
            probabilities=prediction.probabilities,
            metadata={
                "known_tokens": prediction.known_tokens,
                "total_tokens": prediction.total_tokens,
            },
        )

    def save_dataset(self) -> None:
        if self.dataset_path is None:
            return
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"text": text, "intent": intent}
            for text, intent in self._samples
        ]
        self.dataset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def train_and_save(
        self,
        epochs: int = 300,
        learning_rate: float = 1.0,
        weight_decay: float = 0.002,
    ) -> list[float]:
        losses = self.classifier.fit(
            self._samples,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        if self.model_path is not None:
            self.classifier.save(self.model_path)
        self.save_dataset()
        return losses

    def add_samples(
        self,
        samples: Sequence[tuple[str, str]],
        retrain: bool = True,
        epochs: int = 120,
    ) -> None:
        if not samples:
            return

        existing = set(self._samples)
        for sample in samples:
            if sample not in existing:
                self._samples.append(sample)
                existing.add(sample)

        if retrain:
            self.train_and_save(epochs=epochs)
        else:
            self.save_dataset()

    @classmethod
    def from_path_or_seed(
        cls,
        model_path: Path | str | None = None,
        dataset_path: Path | str | None = None,
        auto_train: bool = True,
        seed: int = 42,
    ) -> "NeuralIntentRouter":
        resolved_model = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        resolved_dataset = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH

        seed_samples = load_bundled_seed_dataset()
        samples = list(seed_samples)
        seed_updated = False
        if resolved_dataset.is_file():
            try:
                raw_data = json.loads(resolved_dataset.read_text(encoding="utf-8"))
                if isinstance(raw_data, list):
                    loaded = [
                        (str(item["text"]), str(item["intent"]))
                        for item in raw_data
                        if isinstance(item, dict) and "text" in item and "intent" in item
                    ]
                    if loaded:
                        samples = loaded
                        existing = set(samples)
                        for sample in seed_samples:
                            if sample not in existing:
                                samples.append(sample)
                                existing.add(sample)
                                seed_updated = True
            except (json.JSONDecodeError, KeyError):
                pass
        else:
            seed_updated = True

        seed_intents = {intent for _, intent in seed_samples}
        if resolved_model.is_file():
            classifier = IntentClassifier.load(resolved_model)
            needs_retrain = (
                not seed_intents.issubset(set(classifier.labels))
                or classifier.schema_version < FEATURE_SCHEMA_VERSION
                or seed_updated
            )
            if auto_train and needs_retrain:
                fresh_classifier = IntentClassifier(hidden_size=12, seed=seed)
                router = cls(
                    classifier=fresh_classifier,
                    model_path=resolved_model,
                    dataset_path=resolved_dataset,
                    samples=samples,
                )
                router.train_and_save()
                return router
            return cls(
                classifier=classifier,
                model_path=resolved_model,
                dataset_path=resolved_dataset,
                samples=samples,
            )

        classifier = IntentClassifier(hidden_size=12, seed=seed)
        router = cls(
            classifier=classifier,
            model_path=resolved_model,
            dataset_path=resolved_dataset,
            samples=samples,
        )
        if auto_train:
            router.train_and_save()
        return router
