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
        self._accuracy = 0.0

    @property
    def class_ids(self) -> list[int]:
        return self._class_ids.copy()

    @property
    def accuracy(self) -> float:
        return float(self._accuracy)

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

        self._accuracy = self._compute_accuracy_from_normalized(x, y_ids)

    def predict(self, x_new: list[float]):
        """Predict label (raw) for a raw input x_new."""
        if not self._perceptrons:
            raise ValueError("Model is not trained yet.")

        x_norm = self._preprocessor.transform_inputs([x_new])[0]
        best_class_id = self._predict_id_from_normalized(x_norm)
        return self._preprocessor.id_to_label.get(best_class_id, best_class_id)


    def _predict_id_from_normalized(self, x_norm: list[float]) -> int:
        best_class = None
        best_score = None

        for class_id, model in self._perceptrons.items():
            s = model.score(x_norm)
            if best_score is None or s > best_score:
                best_score = s
                best_class = class_id

        if best_class is None:
            raise RuntimeError("Prediction failed.")
        return int(best_class)

    def _compute_accuracy_from_normalized(self, X_norm: list[list[float]], y_ids: list[int]) -> float:
        if len(X_norm) == 0:
            return 0.0
        if len(X_norm) != len(y_ids):
            raise ValueError("X_norm and y_ids must have the same length.")

        correct = 0
        for x_norm, y_id in zip(X_norm, y_ids):
            pred_id = self._predict_id_from_normalized(x_norm)
            if pred_id == int(y_id):
                correct += 1
        return correct / len(X_norm)