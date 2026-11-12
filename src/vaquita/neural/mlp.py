from collections.abc import Sequence
import math
from random import Random

from vaquita.neural.autograd import Value


class NeuralUnit:
    """Single differentiable neuron backed by autograd Value nodes."""

    def __init__(
        self,
        num_inputs: int,
        nonlin: bool = True,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> None:
        if num_inputs <= 0:
            raise ValueError("num_inputs must be greater than zero.")

        generator = rng if rng is not None else Random(seed)
        scale = 1.0 / math.sqrt(num_inputs)
        self.weights: list[Value] = [
            Value(generator.uniform(-scale, scale)) for _ in range(num_inputs)
        ]
        self.bias = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, inputs: Sequence[float | Value]) -> Value:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(inputs)}."
            )

        activation = self.bias
        for weight, value in zip(self.weights, inputs, strict=True):
            if isinstance(value, (int, float)) and value == 0.0:
                continue
            activation = activation + weight * value
        return activation.tanh() if self.nonlin else activation

    def parameters(self) -> list[Value]:
        return [*self.weights, self.bias]


class Layer:
    """Dense layer of parallel NeuralUnit instances."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        nonlin: bool = True,
        seed: int | None = None,
        rng: Random | None = None,
    ) -> None:
        if num_outputs <= 0:
            raise ValueError("num_outputs must be greater than zero.")

        generator = rng if rng is not None else Random(seed)
        self.units: list[NeuralUnit] = [
            NeuralUnit(num_inputs, nonlin=nonlin, rng=generator)
            for _ in range(num_outputs)
        ]

    def __call__(self, inputs: Sequence[float | Value]) -> list[Value]:
        return [unit(inputs) for unit in self.units]

    def parameters(self) -> list[Value]:
        return [param for unit in self.units for param in unit.parameters()]


class MLP:
    """Multi-Layer Perceptron with tanh hidden layers and linear output layer."""

    def __init__(
        self,
        num_inputs: int,
        layer_sizes: Sequence[int],
        seed: int | None = None,
    ) -> None:
        if num_inputs <= 0:
            raise ValueError("num_inputs must be greater than zero.")
        if not layer_sizes:
            raise ValueError("layer_sizes must not be empty.")

        rng = Random(seed)
        sizes = [num_inputs, *layer_sizes]
        self.output_size = layer_sizes[-1]
        self.layers: list[Layer] = [
            Layer(
                sizes[idx],
                sizes[idx + 1],
                nonlin=(idx != len(layer_sizes) - 1),
                rng=rng,
            )
            for idx in range(len(layer_sizes))
        ]

    def __call__(self, inputs: Sequence[float | Value]) -> list[Value]:
        current = self.layers[0](inputs)
        for layer in self.layers[1:]:
            current = layer(current)
        return current

    def predict(self, inputs: Sequence[float]) -> list[float]:
        return [output.data for output in self(inputs)]

    def parameters(self) -> list[Value]:
        return [param for layer in self.layers for param in layer.parameters()]

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad = 0.0

    def fit(
        self,
        samples: Sequence[tuple[Sequence[float], Sequence[float]]],
        epochs: int = 300,
        learning_rate: float = 0.1,
    ) -> list[float]:
        if not samples:
            raise ValueError("samples must not be empty.")
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        loss_history: list[float] = []
        for _ in range(epochs):
            total_loss = Value(0.0)
            for inputs, targets in samples:
                if len(targets) != self.output_size:
                    raise ValueError(
                        f"Expected {self.output_size} targets, got {len(targets)}."
                    )
                outputs = self(inputs)
                for prediction, target in zip(outputs, targets, strict=True):
                    total_loss = total_loss + (prediction - target) ** 2

            mean_loss = total_loss * (1.0 / len(samples))
            self.zero_grad()
            mean_loss.backward()

            for param in self.parameters():
                param.data -= learning_rate * param.grad

            loss_history.append(mean_loss.data)

        return loss_history
