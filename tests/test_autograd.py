import pytest

from vaquita.neural import Value


def test_value_backward_matches_manual_linear_neuron_gradients() -> None:
    x1 = Value(2.0, label="x1")
    x2 = Value(-3.0, label="x2")
    w1 = Value(0.5, label="w1")
    w2 = Value(1.5, label="w2")
    b = Value(0.2, label="b")
    target = 1.0

    prediction = x1 * w1 + x2 * w2 + b
    error = prediction - target
    loss = error**2
    loss.backward()

    # Manual check: prediction = 1.0 - 4.5 + 0.2 = -3.3, error = -4.3
    expected_grad_pred = 2.0 * (-3.3 - target)
    assert prediction.data == pytest.approx(-3.3)
    assert w1.grad == pytest.approx(expected_grad_pred * x1.data)
    assert w2.grad == pytest.approx(expected_grad_pred * x2.data)
    assert b.grad == pytest.approx(expected_grad_pred)


def test_value_accumulates_gradients_when_node_is_reused() -> None:
    a = Value(3.0)
    # y = a + a * a = a + a^2 -> dy/da = 1 + 2a = 7
    y = a + a * a
    y.backward()

    assert y.data == pytest.approx(12.0)
    assert a.grad == pytest.approx(7.0)


def test_value_tanh_and_relu_activations() -> None:
    x = Value(0.0)
    y_tanh = x.tanh()
    y_tanh.backward()
    # d/dx tanh(0) = 1 - 0^2 = 1
    assert y_tanh.data == pytest.approx(0.0)
    assert x.grad == pytest.approx(1.0)

    neg = Value(-2.0)
    y_neg = neg.relu()
    y_neg.backward()
    assert y_neg.data == 0.0
    assert neg.grad == 0.0

    pos = Value(4.0)
    y_pos = pos.relu()
    y_pos.backward()
    assert y_pos.data == 4.0
    assert pos.grad == 1.0


def test_value_subtraction_and_reverse_operators() -> None:
    a = Value(2.0)
    b = Value(5.0)
    # out = (10 - a) * 3 - b = 24 - 5 = 19
    out = (10.0 - a) * 3.0 - b
    out.backward()

    assert out.data == pytest.approx(19.0)
    assert a.grad == pytest.approx(-3.0)
    assert b.grad == pytest.approx(-1.0)


def test_value_division_exp_and_log() -> None:
    x = Value(2.0)
    # y = log(exp(x) / 2.0) = x - log(2) -> dy/dx = 1
    y = (x.exp() / 2.0).log()
    y.backward()

    assert y.data == pytest.approx(2.0 - pytest.importorskip("math").log(2.0))
    assert x.grad == pytest.approx(1.0)

    with pytest.raises(ValueError, match="strictly positive"):
        Value(0.0).log()
