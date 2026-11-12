import pytest

from vaquita.neural import Neuron


def test_neuron_learns_single_input_linear_rule() -> None:
    # Target rule: y = 2 * x + 1
    samples = [
        ([0.0], 1.0),
        ([1.0], 3.0),
        ([2.0], 5.0),
        ([3.0], 7.0),
    ]
    neuron = Neuron(num_inputs=1, seed=42)

    losses = neuron.fit(samples, epochs=300, learning_rate=0.02)

    assert losses[-1] < losses[0]
    assert losses[-1] < 1e-4
    assert neuron.predict([4.0]) == pytest.approx(9.0, abs=0.05)


def test_neuron_learns_multi_input_weighted_rule() -> None:
    # Target rule: priority_score = 3 * urgency + 1.5 * impact + 0.5
    samples = [
        ([0.0, 0.0], 0.5),
        ([1.0, 0.0], 3.5),
        ([0.0, 1.0], 2.0),
        ([1.0, 1.0], 5.0),
        ([2.0, 1.0], 8.0),
        ([1.0, 2.0], 6.5),
    ]
    neuron = Neuron(num_inputs=2, seed=7)

    losses = neuron.fit(samples, epochs=400, learning_rate=0.02)

    assert losses[-1] < 1e-4
    assert neuron.predict([2.0, 2.0]) == pytest.approx(9.5, abs=0.05)


def test_neuron_validates_dimensions_and_hyperparameters() -> None:
    with pytest.raises(ValueError, match="num_inputs"):
        Neuron(num_inputs=0)

    neuron = Neuron(num_inputs=2, seed=1)

    with pytest.raises(ValueError, match="Expected 2 inputs"):
        neuron.predict([1.0])

    with pytest.raises(ValueError, match="learning_rate"):
        neuron.train_step([1.0, 2.0], target=1.0, learning_rate=0.0)

    with pytest.raises(ValueError, match="samples"):
        neuron.fit([], epochs=10)

    with pytest.raises(ValueError, match="epochs"):
        neuron.fit([([1.0, 2.0], 1.0)], epochs=0)
