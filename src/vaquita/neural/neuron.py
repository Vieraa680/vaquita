from collections.abc import Sequence
from random import Random


class Neuron:
    """Single linear artificial neuron trained via stochastic gradient descent."""

    def __init__(self, num_inputs: int, seed: int | None = None) -> None:
        if num_inputs <= 0:
            raise ValueError("num_inputs must be greater than zero.")

        rng = Random(seed)
        self.weights: list[float] = [rng.uniform(-1.0, 1.0) for _ in range(num_inputs)]
        self.bias: float = rng.uniform(-1.0, 1.0)

    @property
    def num_inputs(self) -> int:
        return len(self.weights)

    def _validate_inputs(self, inputs: Sequence[float]) -> None:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(inputs)}."
            )

    def predict(self, inputs: Sequence[float]) -> float:
        self._validate_inputs(inputs)
        return sum(
            weight * value for weight, value in zip(self.weights, inputs, strict=True)
        ) + self.bias

    def train_step(
        self,
        inputs: Sequence[float],
        target: float,
        learning_rate: float = 0.01,
    ) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        prediction = self.predict(inputs)
        error = prediction - target
        loss = error**2

        grad_output = 2.0 * error
        for idx, value in enumerate(inputs):
            self.weights[idx] -= learning_rate * grad_output * value
        self.bias -= learning_rate * grad_output

        return loss

    def fit(
        self,
        samples: Sequence[tuple[Sequence[float], float]],
        epochs: int = 100,
        learning_rate: float = 0.01,
    ) -> list[float]:
        if not samples:
            raise ValueError("samples must not be empty.")
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        loss_history: list[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for inputs, target in samples:
                epoch_loss += self.train_step(
                    inputs=inputs,
                    target=target,
                    learning_rate=learning_rate,
                )
            loss_history.append(epoch_loss / len(samples))

        return loss_history
