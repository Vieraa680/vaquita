from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata

from vaquita.neural.autograd import Value
from vaquita.neural.mlp import MLP

FEATURE_SCHEMA_VERSION = 2


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_accents = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.findall(r"[a-z0-9]+", without_accents)


def _extract_prefix_stems(tokens: Sequence[str], length: int = 4) -> list[str]:
    return [f"p{length}:{token[:length]}" for token in tokens if len(token) >= length]


def softmax(logits: Sequence[Value]) -> list[Value]:
    if not logits:
        raise ValueError("logits must not be empty.")

    max_logit = max(logit.data for logit in logits)
    exps = [(logit - max_logit).exp() for logit in logits]
    total = exps[0]
    for exp_value in exps[1:]:
        total = total + exp_value
    return [exp_value / total for exp_value in exps]


def _softmax_floats(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)
    return [exp_val / total for exp_val in exps]


class TextVectorizer:
    """L2-normalized TF-IDF vectorizer combining word unigrams and shared 4-char prefix stems."""

    def __init__(self) -> None:
        self.vocab: list[str] = []
        self.idf_weights: list[float] = []
        self._index: dict[str, int] = {}
        self._mean_idf: float = 1.0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def set_vocabulary(
        self,
        vocab: Sequence[str],
        idf_weights: Sequence[float] | None = None,
    ) -> None:
        if not vocab:
            raise ValueError("vocabulary must not be empty.")

        self.vocab = list(vocab)
        if idf_weights is not None:
            if len(idf_weights) != len(self.vocab):
                raise ValueError("idf_weights length must match vocabulary size.")
            self.idf_weights = [float(w) for w in idf_weights]
        else:
            self.idf_weights = [1.0] * len(self.vocab)
        self._mean_idf = sum(self.idf_weights) / len(self.idf_weights)
        self._index = {token: idx for idx, token in enumerate(self.vocab)}

    def fit(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts must not be empty.")

        doc_freq: Counter[str] = Counter()
        word_vocab: set[str] = set()
        stem_doc_freq: Counter[str] = Counter()

        for text in texts:
            tokens = tokenize(text)
            unique_words = set(tokens)
            word_vocab.update(unique_words)
            doc_freq.update(unique_words)

            unique_stems = set(_extract_prefix_stems(tokens))
            stem_doc_freq.update(unique_stems)

        if not word_vocab:
            raise ValueError("vocabulary must not be empty.")

        shared_stems = {
            stem for stem, freq in stem_doc_freq.items() if freq >= 2
        }
        for stem in shared_stems:
            doc_freq[stem] = stem_doc_freq[stem]

        combined_vocab = sorted(word_vocab | shared_stems)
        num_docs = len(texts)
        idf = [
            math.log((num_docs + 1.0) / (doc_freq[feat] + 1.0)) + 1.0
            for feat in combined_vocab
        ]
        self.set_vocabulary(combined_vocab, idf_weights=idf)

    def analyze_detailed(self, text: str) -> tuple[list[float], int, int, float]:
        if not self.vocab:
            raise RuntimeError("TextVectorizer must be fitted before transform.")

        tokens = tokenize(text)
        stems = _extract_prefix_stems(tokens)
        counts = Counter(tokens)
        counts.update(stems)

        vector = [0.0] * len(self.vocab)
        known_word_tokens = 0
        known_idf_sum = 0.0
        for token in tokens:
            idx = self._index.get(token)
            if idx is not None:
                known_word_tokens += 1
                known_idf_sum += self.idf_weights[idx]

        matched_features = 0
        for feat, count in counts.items():
            idx = self._index.get(feat)
            if idx is not None:
                matched_features += 1
                tf = 1.0 + math.log(count)
                subword_scale = 0.4 if feat.startswith("p4:") else 1.0
                vector[idx] = tf * self.idf_weights[idx] * subword_scale

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]

        unknown_word_tokens = max(0, len(tokens) - known_word_tokens)
        denom = known_idf_sum + float(unknown_word_tokens) * self._mean_idf
        idf_coverage = (
            (known_idf_sum / denom)
            if denom > 0.0
            else (1.0 if matched_features > 0 else 0.0)
        )
        return vector, known_word_tokens, len(tokens), idf_coverage

    def analyze(self, text: str) -> tuple[list[float], int, int]:
        vector, known_tokens, total_tokens, _ = self.analyze_detailed(text)
        return vector, known_tokens, total_tokens

    def transform(self, text: str) -> list[float]:
        vector, _, _, _ = self.analyze_detailed(text)
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
        self.schema_version: int = FEATURE_SCHEMA_VERSION

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
        self.schema_version = FEATURE_SCHEMA_VERSION
        label_to_index = {label: idx for idx, label in enumerate(self._labels)}

        self._vectorizer.fit([text for text, _ in samples])
        self._mlp = MLP(
            num_inputs=self._vectorizer.vocab_size,
            layer_sizes=[self._hidden_size, len(self._labels)],
            seed=self._seed,
        )

        hidden_layer, output_layer = self._mlp.layers[0], self._mlp.layers[1]
        num_inputs = self._vectorizer.vocab_size
        num_hidden = len(hidden_layer.units)
        num_classes = len(output_layer.units)

        w1 = [[w.data for w in unit.weights] for unit in hidden_layer.units]
        b1 = [unit.bias.data for unit in hidden_layer.units]
        w2 = [[w.data for w in unit.weights] for unit in output_layer.units]
        b2 = [unit.bias.data for unit in output_layer.units]

        sparse_samples: list[tuple[list[tuple[int, float]], int]] = []
        for text, label in samples:
            vec = self._vectorizer.transform(text)
            nz = [(idx, val) for idx, val in enumerate(vec) if val != 0.0]
            sparse_samples.append((nz, label_to_index[label]))

        inv_n = 1.0 / len(sparse_samples)
        loss_history: list[float] = []

        for _ in range(epochs):
            dw1 = [[0.0] * num_inputs for _ in range(num_hidden)]
            db1 = [0.0] * num_hidden
            dw2 = [[0.0] * num_hidden for _ in range(num_classes)]
            db2 = [0.0] * num_classes
            total_loss = 0.0

            for nz, target_idx in sparse_samples:
                hidden_act = [0.0] * num_hidden
                for j in range(num_hidden):
                    acc = b1[j]
                    w1_j = w1[j]
                    for i, x_val in nz:
                        acc += w1_j[i] * x_val
                    hidden_act[j] = math.tanh(acc)

                logits = [0.0] * num_classes
                for k in range(num_classes):
                    acc = b2[k]
                    w2_k = w2[k]
                    for j in range(num_hidden):
                        acc += w2_k[j] * hidden_act[j]
                    logits[k] = acc

                probs = _softmax_floats(logits)
                total_loss -= math.log(max(probs[target_idx], 1e-15))

                dh = [0.0] * num_hidden
                for k in range(num_classes):
                    dz_k = (probs[k] - (1.0 if k == target_idx else 0.0)) * inv_n
                    db2[k] += dz_k
                    dw2_k = dw2[k]
                    w2_k = w2[k]
                    for j in range(num_hidden):
                        dw2_k[j] += dz_k * hidden_act[j]
                        dh[j] += w2_k[j] * dz_k

                for j in range(num_hidden):
                    da_j = dh[j] * (1.0 - hidden_act[j] * hidden_act[j])
                    db1[j] += da_j
                    dw1_j = dw1[j]
                    for i, x_val in nz:
                        dw1_j[i] += da_j * x_val

            for j in range(num_hidden):
                b1[j] -= learning_rate * (db1[j] + weight_decay * b1[j])
                w1_j = w1[j]
                dw1_j = dw1[j]
                for i in range(num_inputs):
                    w1_j[i] -= learning_rate * (dw1_j[i] + weight_decay * w1_j[i])

            for k in range(num_classes):
                b2[k] -= learning_rate * (db2[k] + weight_decay * b2[k])
                w2_k = w2[k]
                dw2_k = dw2[k]
                for j in range(num_hidden):
                    w2_k[j] -= learning_rate * (dw2_k[j] + weight_decay * w2_k[j])

            loss_history.append(total_loss * inv_n)

        for j, unit in enumerate(hidden_layer.units):
            unit.bias.data = b1[j]
            for i, weight_node in enumerate(unit.weights):
                weight_node.data = w1[j][i]

        for k, unit in enumerate(output_layer.units):
            unit.bias.data = b2[k]
            for j, weight_node in enumerate(unit.weights):
                weight_node.data = w2[k][j]

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

        features, known_tokens, total_tokens, idf_coverage = self._vectorizer.analyze_detailed(text)
        nz = [(idx, val) for idx, val in enumerate(features) if val != 0.0]

        hidden_layer, output_layer = self._mlp.layers[0], self._mlp.layers[1]
        hidden_act = []
        for unit in hidden_layer.units:
            acc = unit.bias.data
            weights = unit.weights
            for i, x_val in nz:
                acc += weights[i].data * x_val
            hidden_act.append(math.tanh(acc) if unit.nonlin else acc)

        logits = []
        for unit in output_layer.units:
            acc = unit.bias.data
            for h_val, weight_node in zip(hidden_act, unit.weights, strict=True):
                acc += weight_node.data * h_val
            logits.append(acc)

        probs = _softmax_floats(logits)
        probabilities = dict(zip(self._labels, probs, strict=True))
        best_label = max(probabilities, key=lambda label: probabilities[label])
        best_confidence = probabilities[best_label]
        raw_coverage = (known_tokens / total_tokens) if total_tokens > 0 else 0.0
        effective_coverage = max(raw_coverage, idf_coverage)
        is_confident = (
            bool(nz)
            and effective_coverage >= min_coverage
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
            "schema_version": self.schema_version,
            "hidden_size": self._hidden_size,
            "labels": self._labels,
            "vocab": self._vectorizer.vocab,
            "idf_weights": self._vectorizer.idf_weights,
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
        idf_weights = raw.get("idf_weights")
        layers_data = raw.get("layers")
        schema_version = raw.get("schema_version", 1)

        if (
            not isinstance(hidden_size, int)
            or not isinstance(labels, list)
            or not isinstance(vocab, list)
            or not isinstance(layers_data, list)
            or not isinstance(schema_version, int)
            or len(labels) < 2
            or not vocab
            or not layers_data
        ):
            raise ValueError("Invalid classifier metadata.")

        parsed_idf: list[float] | None = None
        if idf_weights is not None:
            if not isinstance(idf_weights, list) or len(idf_weights) != len(vocab):
                raise ValueError("Invalid idf_weights in saved classifier.")
            parsed_idf = [float(w) for w in idf_weights]

        classifier = cls(hidden_size=hidden_size, seed=None)
        classifier.schema_version = schema_version
        classifier._labels = [str(label) for label in labels]
        classifier._vectorizer.set_vocabulary(
            [str(token) for token in vocab],
            idf_weights=parsed_idf,
        )

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
