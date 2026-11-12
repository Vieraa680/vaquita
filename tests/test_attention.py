import pytest

from vaquita.neural import Embedding, MLP, SelfAttentionHead, Value, softmax


def test_embedding_maps_indices_and_validates_bounds() -> None:
    emb = Embedding(num_embeddings=5, embedding_dim=3, seed=42)
    vectors = emb([0, 2, 0])

    assert len(vectors) == 3
    assert len(vectors[0]) == 3
    assert [v.data for v in vectors[0]] == [v.data for v in vectors[2]]
    assert len(emb.parameters()) == 15

    with pytest.raises(ValueError, match="out of range"):
        emb([5])


def test_self_attention_weights_and_causal_mask() -> None:
    emb = Embedding(num_embeddings=4, embedding_dim=4, seed=7)
    head = SelfAttentionHead(embed_dim=4, head_dim=3, causal=True, seed=7)

    sequence = emb([0, 1, 2])
    outputs, weights = head.forward_with_weights(sequence)

    assert len(outputs) == 3
    assert len(outputs[0]) == 3
    assert len(weights) == 3

    for row_idx, row in enumerate(weights):
        assert sum(w.data for w in row) == pytest.approx(1.0)
        for col_idx in range(row_idx + 1, len(row)):
            assert row[col_idx].data == 0.0


def test_attention_with_positional_embeddings_solves_word_order_task() -> None:
    # Identical Bag-of-Words in all 4 commands; only word order decides if "luz" turns on (1) or off (0).
    vocab = {"enciende": 0, "apaga": 1, "luz": 2, "tele": 3}
    dataset = [
        (["enciende", "luz", "apaga", "tele"], 1),
        (["apaga", "tele", "enciende", "luz"], 1),
        (["apaga", "luz", "enciende", "tele"], 0),
        (["enciende", "tele", "apaga", "luz"], 0),
    ]

    embed_dim = 4
    seq_len = 4
    token_emb = Embedding(num_embeddings=len(vocab), embedding_dim=embed_dim, seed=11)
    pos_emb = Embedding(num_embeddings=seq_len, embedding_dim=embed_dim, seed=13)
    attn = SelfAttentionHead(embed_dim=embed_dim, head_dim=4, causal=False, seed=17)
    # Pool across positions (4 * 4 = 16 features) -> 2 classes (off / on)
    classifier = MLP(num_inputs=seq_len * 4, layer_sizes=[6, 2], seed=19)

    params = [
        *token_emb.parameters(),
        *pos_emb.parameters(),
        *attn.parameters(),
        *classifier.parameters(),
    ]

    losses: list[float] = []
    for _ in range(120):
        total_loss = Value(0.0)
        for words, label in dataset:
            tok_vecs = token_emb([vocab[w] for w in words])
            pos_vecs = pos_emb(list(range(seq_len)))
            combined = [
                [t + p for t, p in zip(tv, pv, strict=True)]
                for tv, pv in zip(tok_vecs, pos_vecs, strict=True)
            ]
            attended = attn(combined)
            residual = [
                [c + a for c, a in zip(cv, av, strict=True)]
                for cv, av in zip(combined, attended, strict=True)
            ]
            flattened = [val for vec in residual for val in vec]
            probs = softmax(classifier(flattened))
            total_loss = total_loss - probs[label].log()

        mean_loss = total_loss * (1.0 / len(dataset))
        for p in params:
            p.grad = 0.0
        mean_loss.backward()
        for p in params:
            p.data -= 0.4 * p.grad
        losses.append(mean_loss.data)

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.05

    for words, expected_label in dataset:
        tok_vecs = token_emb([vocab[w] for w in words])
        pos_vecs = pos_emb(list(range(seq_len)))
        combined = [
            [t + p for t, p in zip(tv, pv, strict=True)]
            for tv, pv in zip(tok_vecs, pos_vecs, strict=True)
        ]
        attended = attn(combined)
        residual = [
            [c + a for c, a in zip(cv, av, strict=True)]
            for cv, av in zip(combined, attended, strict=True)
        ]
        flattened = [val for vec in residual for val in vec]
        probs = softmax(classifier(flattened))
        predicted = 1 if probs[1].data > probs[0].data else 0
        assert predicted == expected_label
