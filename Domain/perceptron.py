import random

class Perceptron:
    def __init__(self, num_features: int, activation_function):
        self._num_features = num_features
        self._activation_function = activation_function

        self._weights = [0.0] * num_features
        self._bias = 0.0

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def weights(self) -> list[float]:
        return self._weights.copy()

    @property
    def bias(self) -> float:
        return self._bias

    def set_parameters(self, weights: list[float], bias: float) -> None:
        if len(weights) != self._num_features:
            raise ValueError("Weights length does not match num_features.")

        self._weights = weights.copy()
        self._bias = float(bias)

    def randomize_parameters(self, low: float = -0.5, high: float = 0.5) -> None:
        self._weights = [random.uniform(low, high) for _ in range(self._num_features)]
        self._bias = random.uniform(low, high)

    def update_weights(self, delta_w: list[float]) -> None:
        if len(delta_w) != self._num_features:
            raise ValueError("delta_w length does not match num_features.")

        for i in range(self._num_features):
            self._weights[i] += float(delta_w[i])

    def update_bias(self, delta_b: float) -> None:
        self._bias += float(delta_b)

    def weighted_sum(self, inputs: list[float]) -> float:
        if len(inputs) != self._num_features:
            raise ValueError(f"Expected {self._num_features} features, got {len(inputs)}")

        s = 0.0
        for i in range(self._num_features):
            s += self._weights[i] * inputs[i]
        return s

    def predict(self, features: list[float]) -> int:
        net = self.weighted_sum(features) + self._bias
        return self._activation_function(net)