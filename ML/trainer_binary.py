from Domain.perceptron import Perceptron


class BinaryPerceptronTrainer:
    def __init__(self, perceptron: Perceptron, learning_rate: float, max_epoch: int):
        self._perceptron = perceptron
        self._learning_rate = learning_rate
        self._max_epoch = max_epoch

        self._num_epoch = 0
        self._num_updates = 0
        self._converged = False
        self._accuracy = 0.0

    @property
    def num_epoch(self) -> int:
        return self._num_epoch

    @property
    def num_updates(self) -> int:
        return self._num_updates

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def accuracy(self) -> float:
        return self._accuracy

    def train(self, x: list[list[float]], y01: list[int], random_init: bool = True) -> None:
        if len(x) == 0:
            return
        if len(x) != len(y01):
            raise ValueError("X and y01 must have the same length.")

        self._validate_binary_labels(y01)

        if random_init:
            self._perceptron.randomize_parameters(-0.5, 0.5)

        self._num_updates = 0
        self._converged = False
        self._num_epoch = 0

        for epoch in range(1, self._max_epoch + 1):
            errors_in_epoch = 0

            for i in range(len(x)):
                x_i = x[i]
                desired = int(y01[i])

                actual = self._perceptron.predict(x_i)
                if actual not in (0, 1):
                    raise ValueError("Activation must output {0,1}.")

                error = desired - actual

                if error != 0:
                    errors_in_epoch += 1
                    self._apply_update(x_i, error)
                    self._num_updates += 1

            self._num_epoch = epoch

            if errors_in_epoch == 0:
                self._converged = True
                break

        self._accuracy = self._compute_accuracy(x, y01)

    def predict(self, x_new: list[float]) -> int:
        pred = self._perceptron.predict(x_new)
        if pred not in (0, 1):
            raise ValueError("Activation must output {0,1}.")
        return pred

    def _apply_update(self, x: list[float], error: int) -> None:
        lr = self._learning_rate

        delta_w = []
        for j in range(self._perceptron.num_features):
            delta_w.append(lr * error * x[j])

        delta_b = lr * error

        self._perceptron.update_weights(delta_w)
        self._perceptron.update_bias(delta_b)

    def _validate_binary_labels(self, y01: list[int]) -> None:
        if sorted(set(int(v) for v in y01)) != [0, 1]:
            raise ValueError("Binary perceptron requires labels {0,1}.")

    def _compute_accuracy(self, x: list[list[float]], y01: list[int]) -> float:
        correct = 0
        for i in range(len(x)):
            pred = self._perceptron.predict(x[i])
            if pred == int(y01[i]):
                correct += 1
        return correct / len(x)