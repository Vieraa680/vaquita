import pytest

from vaquita.neural import MLP, Neuron


def test_single_linear_neuron_cannot_solve_xor_while_mlp_can() -> None:
    xor_samples_scalar = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
    linear_neuron = Neuron(num_inputs=2, seed=42)
    linear_losses = linear_neuron.fit(
        xor_samples_scalar, epochs=300, learning_rate=0.05
    )
    assert linear_losses[-1] > 0.2

    xor_samples_vector = [(inputs, [target]) for inputs, target in xor_samples_scalar]
    mlp = MLP(num_inputs=2, layer_sizes=[4, 1], seed=42)
    mlp_losses = mlp.fit(xor_samples_vector, epochs=600, learning_rate=0.1)

    assert mlp_losses[-1] < 0.01
    assert mlp.predict([0.0, 0.0])[0] == pytest.approx(0.0, abs=0.1)
    assert mlp.predict([0.0, 1.0])[0] == pytest.approx(1.0, abs=0.1)
    assert mlp.predict([1.0, 0.0])[0] == pytest.approx(1.0, abs=0.1)
    assert mlp.predict([1.0, 1.0])[0] == pytest.approx(0.0, abs=0.1)


def test_mlp_learns_multi_output_classification_targets() -> None:
    # 3 distinct signal patterns mapped to 3 one-hot categories
    samples = [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([0.9, 0.1, 0.0], [1.0, 0.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.1, 0.9, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.1, 0.9], [0.0, 0.0, 1.0]),
    ]
    mlp = MLP(num_inputs=3, layer_sizes=[4, 3], seed=7)
    losses = mlp.fit(samples, epochs=400, learning_rate=0.1)

    assert losses[-1] < 0.02

    pred_cat_0 = mlp.predict([0.95, 0.05, 0.0])
    pred_cat_1 = mlp.predict([0.0, 0.95, 0.05])
    pred_cat_2 = mlp.predict([0.05, 0.0, 0.95])

    assert pred_cat_0.index(max(pred_cat_0)) == 0
    assert pred_cat_1.index(max(pred_cat_1)) == 1
    assert pred_cat_2.index(max(pred_cat_2)) == 2


def test_mlp_validates_configuration_and_inputs() -> None:
    with pytest.raises(ValueError, match="num_inputs"):
        MLP(num_inputs=0, layer_sizes=[2, 1])

    with pytest.raises(ValueError, match="layer_sizes"):
        MLP(num_inputs=2, layer_sizes=[])

    mlp = MLP(num_inputs=2, layer_sizes=[3, 2], seed=1)
    # (2*3 + 3) + (3*2 + 2) = 9 + 8 = 17 trainable parameters
    assert len(mlp.parameters()) == 17

    with pytest.raises(ValueError, match="Expected 2 inputs"):
        mlp.predict([1.0])

    with pytest.raises(ValueError, match="Expected 2 targets"):
        mlp.fit([([1.0, 2.0], [1.0])], epochs=5)
