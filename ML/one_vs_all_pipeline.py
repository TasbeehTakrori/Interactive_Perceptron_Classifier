from Domain.perceptron import Perceptron
from ML.trainer_binary import BinaryPerceptronTrainer
from ML.preprocessor import Preprocessor


class OneVsAllPipeline:
    def __init__(
        self,
        preprocessor: Preprocessor,
        learning_rate: float,
        max_epoch: int,
        activation_function
    ):
        self._preprocessor = preprocessor
        self._learning_rate = learning_rate
        self._max_epoch = max_epoch
        self._activation_function = activation_function

        self._perceptrons = {}
        self._class_ids = []

    @property
    def class_ids(self) -> list[int]:
        return self._class_ids.copy()

    def train_one_vs_all(self, features: list[list[float]], labels: list) -> None:
        if len(features) == 0:
            return
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")

        x = self._preprocessor.normalize_inputs(features)
        y_ids = self._preprocessor.encode_labels(labels)

        self._class_ids = sorted(set(y_ids))
        if len(self._class_ids) < 2:
            raise ValueError("One-vs-All requires at least 2 classes.")

        num_features = len(x[0])
        self._perceptrons = {}

        for class_id in self._class_ids:
            y01 = [1 if y == class_id else 0 for y in y_ids]

            perceptron = Perceptron(
                num_features=num_features,
                activation_function=self._activation_function
            )

            trainer = BinaryPerceptronTrainer(
                perceptron=perceptron,
                learning_rate=self._learning_rate,
                max_epoch=self._max_epoch
            )

            trainer.train(x, y01, random_init=True)

            self._perceptrons[class_id] = perceptron

    def predict(self, x_new: list[float]):
        if not self._perceptrons:
            raise ValueError("Model is not trained yet.")

        x = self._preprocessor.transform_inputs([x_new])[0]

        best_class = None
        best_score = None

        for class_id, model in self._perceptrons.items():
            s = model.score(x)
            if best_score is None or s > best_score:
                best_score = s
                best_class = class_id

        return self._preprocessor.id_to_label.get(best_class, best_class)

    def accuracy(self, features: list[list[float]], labels: list) -> float:
        if len(features) == 0:
            return 0.0
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")

        correct = 0
        for x, y in zip(features, labels):
            if self.predict(x) == y:
                correct += 1
        return correct / len(features)