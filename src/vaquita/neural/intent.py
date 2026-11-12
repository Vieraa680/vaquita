from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

from vaquita.neural.autograd import Value
from vaquita.neural.mlp import MLP


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.findall(r"[a-z0-9]+", without_accents)


def softmax(logits: Sequence[Value]) -> list[Value]:
    if not logits:
        raise ValueError("logits must not be empty.")

    max_logit = max(logit.data for logit in logits)
    exps = [(logit - max_logit).exp() for logit in logits]
    total = exps[0]
    for exp_value in exps[1:]:
        total = total + exp_value
    return [exp_value / total for exp_value in exps]


class TextVectorizer:
    """Binary Bag-of-Words vectorizer with accent-insensitive tokenization."""

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self._index: dict[str, int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def set_vocabulary(self, vocab: Sequence[str]) -> None:
        if not vocab:
            raise ValueError("vocabulary must not be empty.")

        self.vocab = list(vocab)
        self._index = {token: idx for idx, token in enumerate(self.vocab)}

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts must not be empty.")

        tokens: set[str] = set()
        for text in texts:
            tokens.update(tokenize(text))

        if not tokens:
            raise ValueError("vocabulary must not be empty.")

        self.set_vocabulary(sorted(tokens))

    def analyze(self, text: str) -> tuple[list[float], int, int]:
        if not self.vocab:
            raise RuntimeError("TextVectorizer must be fitted before transform.")

        tokens = tokenize(text)
        vector = [0.0] * len(self.vocab)
        known_tokens = 0
        for token in tokens:
            idx = self._index.get(token)
            if idx is not None:
                vector[idx] = 1.0
                known_tokens += 1
        return vector, known_tokens, len(tokens)

    def transform(self, text: str) -> list[float]:
        vector, _, _ = self.analyze(text)
        return vector


@dataclass(frozen=True)
class IntentPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    known_tokens: int
    total_tokens: int
    is_confident: bool


class IntentClassifier:
    """Text intent classifier backed by TextVectorizer, MLP, and cross-entropy loss."""

    def __init__(self, hidden_size: int = 8, seed: int | None = 42) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")

        self._hidden_size = hidden_size
        self._seed = seed
        self._vectorizer = TextVectorizer()
        self._labels: list[str] = []
        self._mlp: MLP | None = None

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def vocab_size(self) -> int:
        return self._vectorizer.vocab_size

    def fit(
        self,
        samples: Sequence[tuple[str, str]],
        epochs: int = 120,
        learning_rate: float = 0.2,
        weight_decay: float = 0.005,
    ) -> list[float]:
        if not samples:
            raise ValueError("samples must not be empty.")
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")

        unique_labels = sorted({label for _, label in samples})
        if len(unique_labels) < 2:
            raise ValueError("At least two distinct intent labels are required.")

        self._labels = unique_labels
        label_to_index = {label: idx for idx, label in enumerate(self._labels)}

        self._vectorizer.fit([text for text, _ in samples])
        self._mlp = MLP(
            num_inputs=self._vectorizer.vocab_size,
            layer_sizes=[self._hidden_size, len(self._labels)],
            seed=self._seed,
        )

        encoded_samples = [
            (self._vectorizer.transform(text), label_to_index[label])
            for text, label in samples
        ]

        loss_history: list[float] = []
        for _ in range(epochs):
            total_loss = Value(0.0)
            for features, target_idx in encoded_samples:
                logits = self._mlp(features)
                probs = softmax(logits)
                total_loss = total_loss - probs[target_idx].log()

            mean_loss = total_loss * (1.0 / len(encoded_samples))
            self._mlp.zero_grad()
            mean_loss.backward()

            for param in self._mlp.parameters():
                param.data -= learning_rate * (param.grad + weight_decay * param.data)

            loss_history.append(mean_loss.data)

        return loss_history

    def predict(
        self,
        text: str,
        min_confidence: float = 0.65,
        min_coverage: float = 0.3,
    ) -> IntentPrediction:
        if self._mlp is None:
            raise RuntimeError("IntentClassifier must be fitted before predict.")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0.")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be between 0.0 and 1.0.")

        features, known_tokens, total_tokens = self._vectorizer.analyze(text)
        probs = softmax(self._mlp(features))
        probabilities = {
            label: prob.data
            for label, prob in zip(self._labels, probs, strict=True)
        }
        best_label = max(probabilities, key=lambda label: probabilities[label])
        best_confidence = probabilities[best_label]
        coverage = (known_tokens / total_tokens) if total_tokens > 0 else 0.0
        is_confident = (
            known_tokens > 0
            and coverage >= min_coverage
            and best_confidence >= min_confidence
        )
        resolved_label = best_label if is_confident else "unknown"

        return IntentPrediction(
            label=resolved_label,
            confidence=best_confidence,
            probabilities=probabilities,
            known_tokens=known_tokens,
            total_tokens=total_tokens,
            is_confident=is_confident,
        )

    def save(self, path: str | Path) -> None:
        if self._mlp is None:
            raise RuntimeError("IntentClassifier must be fitted before save.")

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "hidden_size": self._hidden_size,
            "labels": self._labels,
            "vocab": self._vectorizer.vocab,
            "layers": [
                {
                    "units": [
                        {
                            "weights": [weight.data for weight in unit.weights],
                            "bias": unit.bias.data,
                            "nonlin": unit.nonlin,
                        }
                        for unit in layer.units
                    ]
                }
                for layer in self._mlp.layers
            ],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Invalid classifier file format.")

        hidden_size = raw.get("hidden_size")
        labels = raw.get("labels")
        vocab = raw.get("vocab")
        layers_data = raw.get("layers")

        if (
            not isinstance(hidden_size, int)
            or not isinstance(labels, list)
            or not isinstance(vocab, list)
            or not isinstance(layers_data, list)
            or len(labels) < 2
            or not vocab
            or not layers_data
        ):
            raise ValueError("Invalid classifier metadata.")

        classifier = cls(hidden_size=hidden_size, seed=None)
        classifier._labels = [str(label) for label in labels]
        classifier._vectorizer.set_vocabulary([str(token) for token in vocab])

        mlp = MLP(
            num_inputs=classifier.vocab_size,
            layer_sizes=[hidden_size, len(classifier._labels)],
            seed=0,
        )
        if len(layers_data) != len(mlp.layers):
            raise ValueError("Layer count mismatch in saved classifier.")

        for layer, layer_data in zip(mlp.layers, layers_data, strict=True):
            units_data = layer_data.get("units")
            if not isinstance(units_data, list) or len(units_data) != len(layer.units):
                raise ValueError("Unit count mismatch in saved classifier.")

            for unit, unit_data in zip(layer.units, units_data, strict=True):
                weights = unit_data.get("weights")
                bias = unit_data.get("bias")
                nonlin = unit_data.get("nonlin")
                if (
                    not isinstance(weights, list)
                    or len(weights) != len(unit.weights)
                    or not isinstance(bias, (int, float))
                    or not isinstance(nonlin, bool)
                ):
                    raise ValueError("Invalid unit parameters in saved classifier.")

                for weight_node, weight_val in zip(unit.weights, weights, strict=True):
                    weight_node.data = float(weight_val)
                unit.bias.data = float(bias)
                unit.nonlin = nonlin

        classifier._mlp = mlp
        return classifier
