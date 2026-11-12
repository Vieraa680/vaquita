from collections.abc import Callable
import math


class Value:
    """Scalar value that tracks operations to compute gradients via backpropagation."""

    def __init__(
        self,
        data: float,
        children: tuple["Value", ...] = (),
        op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.label = label
        self._prev = children
        self._op = op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        prefix = f"{self.label}: " if self.label else ""
        return f"Value({prefix}data={self.data:.4f}, grad={self.grad:.4f})"

    @staticmethod
    def _ensure_value(other: "Value | float") -> "Value":
        return other if isinstance(other, Value) else Value(float(other))

    def __add__(self, other: "Value | float") -> "Value":
        rhs = self._ensure_value(other)
        out = Value(self.data + rhs.data, (self, rhs), "+")

        def _backward() -> None:
            self.grad += out.grad
            rhs.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-self._ensure_value(other))

    def __rsub__(self, other: float) -> "Value":
        return self._ensure_value(other) + (-self)

    def __mul__(self, other: "Value | float") -> "Value":
        rhs = self._ensure_value(other)
        out = Value(self.data * rhs.data, (self, rhs), "*")

        def _backward() -> None:
            self.grad += rhs.data * out.grad
            rhs.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __truediv__(self, other: "Value | float") -> "Value":
        return self * (self._ensure_value(other) ** -1)

    def __rtruediv__(self, other: float) -> "Value":
        return self._ensure_value(other) * (self**-1)

    def __pow__(self, exponent: int | float) -> "Value":
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            local_grad = 0.0 if exponent == 0 else exponent * (self.data ** (exponent - 1))
            self.grad += local_grad * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        value = math.exp(self.data)
        out = Value(value, (self,), "exp")

        def _backward() -> None:
            self.grad += value * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0.0:
            raise ValueError("log input must be strictly positive.")

        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        activated = math.tanh(self.data)
        out = Value(activated, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - activated**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        activated = self.data if self.data > 0.0 else 0.0
        out = Value(activated, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
