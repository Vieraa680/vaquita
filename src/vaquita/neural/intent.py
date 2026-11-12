from collections.abc import Sequence
from dataclasses import dataclass
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

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts must not be empty.")

        tokens: set[str] = set()
        for text in texts:
            tokens.update(tokenize(text))

        if not tokens:
            raise ValueError("vocabulary must not be empty.")

        self.vocab = sorted(tokens)
        self._index = {token: idx for idx, token in enumerate(self.vocab)}

    def transform(self, text: str) -> list[float]:
        if not self.vocab:
            raise RuntimeError("TextVectorizer must be fitted before transform.")

        vector = [0.0] * len(self.vocab)
        for token in tokenize(text):
            idx = self._index.get(token)
            if idx is not None:
                vector[idx] = 1.0
        return vector


@dataclass(frozen=True)
class IntentPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


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
    ) -> list[float]:
        if not samples:
            raise ValueError("samples must not be empty.")
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

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
                param.data -= learning_rate * param.grad

            loss_history.append(mean_loss.data)

        return loss_history

    def predict(self, text: str) -> IntentPrediction:
        if self._mlp is None:
            raise RuntimeError("IntentClassifier must be fitted before predict.")

        features = self._vectorizer.transform(text)
        probs = softmax(self._mlp(features))
        probabilities = {
            label: prob.data
            for label, prob in zip(self._labels, probs, strict=True)
        }
        best_label = max(probabilities, key=lambda label: probabilities[label])
        return IntentPrediction(
            label=best_label,
            confidence=probabilities[best_label],
            probabilities=probabilities,
        )
