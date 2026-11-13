import pytest

from vaquita.neural import MiniTransformerLM, TransformerBlock, Value


def test_transformer_block_preserves_dimensions() -> None:
    block = TransformerBlock(embed_dim=4, hidden_dim=8, seed=42)
    seq = [[Value(0.1 * (i + j)) for j in range(4)] for i in range(3)]

    out = block(seq)
    assert len(out) == 3
    assert len(out[0]) == 4


def test_mini_transformer_enforces_causal_invariance() -> None:
    model = MiniTransformerLM(
        vocab_size=6,
        max_seq_len=6,
        embed_dim=4,
        hidden_dim=8,
        num_layers=1,
        seed=42,
    )
    # First token is identical in both sequences (token 1); second token differs (2 vs 5).
    logits_a = model.forward([1, 2])
    logits_b = model.forward([1, 5])

    # Due to causal masking, logits at position 0 must be strictly identical regardless of future tokens.
    assert [v.data for v in logits_a[0]] == pytest.approx(
        [v.data for v in logits_b[0]], abs=1e-6
    )
    # But logits at position 1 must differ because the history differs.
    assert [v.data for v in logits_a[1]] != [v.data for v in logits_b[1]]


def test_mini_transformer_learns_autoregressive_sequence_and_generates() -> None:
    # Target sequence: 1 -> 2 -> 3 -> 4
    model = MiniTransformerLM(
        vocab_size=5,
        max_seq_len=6,
        embed_dim=4,
        hidden_dim=8,
        num_layers=1,
        seed=42,
    )
    inputs = [1, 2, 3]
    targets = [2, 3, 4]

    initial_loss = model.compute_loss(inputs, targets).data
    for _ in range(80):
        loss = model.compute_loss(inputs, targets)
        model.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.data -= 0.35 * param.grad

    final_loss = model.compute_loss(inputs, targets).data
    assert final_loss < initial_loss
    assert final_loss < 0.05

    # Autoregressive generation from prompt [1]
    generated = model.generate(prompt_ids=[1], max_new_tokens=3)
    assert generated == [1, 2, 3, 4]


def test_mini_transformer_validations() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        MiniTransformerLM(vocab_size=0, max_seq_len=4, embed_dim=4, hidden_dim=4)

    model = MiniTransformerLM(
        vocab_size=4, max_seq_len=3, embed_dim=4, hidden_dim=4, seed=1
    )

    with pytest.raises(ValueError, match="token_ids must not be empty"):
        model.forward([])

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.forward([0, 1, 2, 3])

    with pytest.raises(ValueError, match="prompt_ids must not be empty"):
        model.generate([], max_new_tokens=2)
