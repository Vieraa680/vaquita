from collections.abc import Sequence
import math
from random import Random

from vaquita.neural.autograd import Value
from vaquita.neural.intent import softmax


def _dot(left: Sequence[Value], right: Sequence[Value]) -> Value:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must be non-empty and of equal length.")

    total = left[0] * right[0]
    for a, b in zip(left[1:], right[1:], strict=True):
        total = total + a * b
    return total


def _linear_project(
    vector: Sequence[Value],
    matrix: Sequence[Sequence[Value]],
) -> list[Value]:
    return [_dot(vector, row) for row in matrix]


def _init_matrix(
    rows: int,
    cols: int,
    scale: float,
    rng: Random,
) -> list[list[Value]]:
    return [
        [Value(rng.uniform(-scale, scale)) for _ in range(cols)]
        for _ in range(rows)
    ]


class Embedding:
    """Lookup table mapping integer token or position IDs to learnable Value vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> None:
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be greater than zero.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        generator = rng if rng is not None else Random(seed)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.table = _init_matrix(num_embeddings, embedding_dim, scale, generator)

    def __call__(self, indices: Sequence[int]) -> list[list[Value]]:
        result: list[list[Value]] = []
        for idx in indices:
            if not 0 <= idx < self.num_embeddings:
                raise ValueError(
                    f"Embedding index {idx} out of range [0, {self.num_embeddings})."
                )
            result.append(list(self.table[idx]))
        return result

    def parameters(self) -> list[Value]:
        return [value for row in self.table for value in row]


class SelfAttentionHead:
    """Single scaled dot-product self-attention head with optional causal masking."""

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        causal: bool = False,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> None:
        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than zero.")
        if head_dim <= 0:
            raise ValueError("head_dim must be greater than zero.")

        generator = rng if rng is not None else Random(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.causal = causal

        self.w_query = _init_matrix(head_dim, embed_dim, scale, generator)
        self.w_key = _init_matrix(head_dim, embed_dim, scale, generator)
        self.w_value = _init_matrix(head_dim, embed_dim, scale, generator)

    def forward_with_weights(
        self,
        sequence: Sequence[Sequence[Value]],
    ) -> tuple[list[list[Value]], list[list[Value]]]:
        if not sequence:
            raise ValueError("sequence must not be empty.")
        for token_vec in sequence:
            if len(token_vec) != self.embed_dim:
                raise ValueError(
                    f"Expected embedding dim {self.embed_dim}, got {len(token_vec)}."
                )

        queries = [_linear_project(token, self.w_query) for token in sequence]
        keys = [_linear_project(token, self.w_key) for token in sequence]
        values = [_linear_project(token, self.w_value) for token in sequence]

        scale = 1.0 / math.sqrt(self.head_dim)
        seq_len = len(sequence)

        outputs: list[list[Value]] = []
        attention_weights: list[list[Value]] = []

        for query_idx in range(seq_len):
            limit = query_idx + 1 if self.causal else seq_len
            scores = [
                _dot(queries[query_idx], keys[key_idx]) * scale
                for key_idx in range(limit)
            ]
            probs = softmax(scores)

            row_weights = list(probs)
            if self.causal and limit < seq_len:
                row_weights.extend(Value(0.0) for _ in range(seq_len - limit))
            attention_weights.append(row_weights)

            attended: list[Value] = []
            for dim_idx in range(self.head_dim):
                acc = probs[0] * values[0][dim_idx]
                for key_idx in range(1, limit):
                    acc = acc + probs[key_idx] * values[key_idx][dim_idx]
                attended.append(acc)
            outputs.append(attended)

        return outputs, attention_weights

    def __call__(self, sequence: Sequence[Sequence[Value]]) -> list[list[Value]]:
        outputs, _ = self.forward_with_weights(sequence)
        return outputs

    def parameters(self) -> list[Value]:
        return [
            *[val for row in self.w_query for val in row],
            *[val for row in self.w_key for val in row],
            *[val for row in self.w_value for val in row],
        ]
