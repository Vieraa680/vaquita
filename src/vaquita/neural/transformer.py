from collections.abc import Sequence
from random import Random

from vaquita.neural.attention import Embedding, SelfAttentionHead
from vaquita.neural.autograd import Value
from vaquita.neural.intent import softmax
from vaquita.neural.mlp import Layer


class TransformerBlock:
    """Standard causal Transformer block with self-attention, feed-forward, and residual connections."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> None:
        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than zero.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")

        generator = rng if rng is not None else Random(seed)
        self.embed_dim = embed_dim
        self.attn = SelfAttentionHead(
            embed_dim=embed_dim,
            head_dim=embed_dim,
            causal=True,
            rng=generator,
        )
        self.mlp_up = Layer(embed_dim, hidden_dim, nonlin=True, rng=generator)
        self.mlp_down = Layer(hidden_dim, embed_dim, nonlin=False, rng=generator)

    def __call__(self, sequence: Sequence[Sequence[Value]]) -> list[list[Value]]:
        if not sequence:
            raise ValueError("sequence must not be empty.")

        attended = self.attn(sequence)
        res1 = [
            [x_val + a_val for x_val, a_val in zip(seq_vec, att_vec, strict=True)]
            for seq_vec, att_vec in zip(sequence, attended, strict=True)
        ]

        outputs: list[list[Value]] = []
        for vec in res1:
            ff_out = self.mlp_down(self.mlp_up(vec))
            outputs.append(
                [r_val + f_val for r_val, f_val in zip(vec, ff_out, strict=True)]
            )
        return outputs

    def parameters(self) -> list[Value]:
        return [
            *self.attn.parameters(),
            *self.mlp_up.parameters(),
            *self.mlp_down.parameters(),
        ]


class MiniTransformerLM:
    """Minimal autoregressive Transformer language model for sequential token generation."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        seed: int | None = None,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than zero.")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than zero.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        rng = Random(seed)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.token_emb = Embedding(vocab_size, embed_dim, rng=rng)
        self.pos_emb = Embedding(max_seq_len, embed_dim, rng=rng)
        self.blocks = [
            TransformerBlock(embed_dim, hidden_dim, rng=rng)
            for _ in range(num_layers)
        ]
        self.lm_head = Layer(embed_dim, vocab_size, nonlin=False, rng=rng)

    def forward(self, token_ids: Sequence[int]) -> list[list[Value]]:
        seq_len = len(token_ids)
        if seq_len == 0:
            raise ValueError("token_ids must not be empty.")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}."
            )

        tok_vecs = self.token_emb(token_ids)
        pos_vecs = self.pos_emb(list(range(seq_len)))

        current = [
            [t + p for t, p in zip(tv, pv, strict=True)]
            for tv, pv in zip(tok_vecs, pos_vecs, strict=True)
        ]

        for block in self.blocks:
            current = block(current)

        return [self.lm_head(vec) for vec in current]

    def compute_loss(
        self,
        input_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> Value:
        if len(input_ids) != len(target_ids):
            raise ValueError("input_ids and target_ids must have the same length.")
        if not input_ids:
            raise ValueError("input_ids must not be empty.")

        logits_seq = self.forward(input_ids)
        total_loss = Value(0.0)
        for logits, target in zip(logits_seq, target_ids, strict=True):
            if not 0 <= target < self.vocab_size:
                raise ValueError(f"Target ID {target} out of vocabulary range.")
            probs = softmax(logits)
            total_loss = total_loss - probs[target].log()

        return total_loss * (1.0 / len(input_ids))

    def parameters(self) -> list[Value]:
        return [
            *self.token_emb.parameters(),
            *self.pos_emb.parameters(),
            *[p for block in self.blocks for p in block.parameters()],
            *self.lm_head.parameters(),
        ]

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = 0.0

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
    ) -> list[int]:
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        result = list(prompt_ids)
        for _ in range(max_new_tokens):
            if len(result) >= self.max_seq_len:
                break
            logits_seq = self.forward(result)
            last_logits = logits_seq[-1]
            next_token = max(
                range(len(last_logits)),
                key=lambda idx: last_logits[idx].data,
            )
            result.append(next_token)

        return result
