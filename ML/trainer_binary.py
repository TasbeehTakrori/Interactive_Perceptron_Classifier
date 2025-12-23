from Domain.perceptron import Perceptron


class BinaryPerceptronTrainer:
    def __init__(
            self,
            perceptron: Perceptron,
            learning_rate: float,
            max_epoch: int):
        self._perceptron = perceptron
        self._learning_rate = learning_rate
        self._max_epoch = max_epoch

        self._num_epoch = 0


    def train(self, features: list[list], ):