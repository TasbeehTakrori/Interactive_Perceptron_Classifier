from ML.preprocessor import Preprocessor
from ML.trainer_binary import BinaryPerceptronTrainer


class BinaryPerceptronPipeline:
    def __init__(self, preprocessor: Preprocessor, trainer: BinaryPerceptronTrainer):
        self._preprocessor = preprocessor
        self._trainer = trainer

        self._binary_to_class_id = None
        self._class_id_to_binary = None

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @property
    def trainer(self) -> BinaryPerceptronTrainer:
        return self._trainer

    def fit(self, features: list[list[float]], labels: list) -> None:
        if len(features) == 0:
            return
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")

        X = self._preprocessor.normalize_inputs(features)
        y_ids = self._preprocessor.encode_labels(labels)

        y01 = self._to_binary_01_with_mapping(y_ids)

        self._trainer.train(X, y01, random_init=True)

    def predict(self, x_new: list[float]):
        x_fixed = self._preprocessor.transform_inputs([x_new])[0]
        pred01 = self._trainer.predict(x_fixed)

        class_id = self._binary_to_class_id[pred01]
        return self._preprocessor.id_to_label.get(class_id, class_id)

    def _to_binary_01_with_mapping(self, y_ids: list[int]) -> list[int]:
        unique = sorted(set(int(v) for v in y_ids))
        if len(unique) != 2:
            raise ValueError("Binary classifier requires exactly 2 classes.")

        low_id, high_id = unique[0], unique[1]

        self._class_id_to_binary = {low_id: 0, high_id: 1}
        self._binary_to_class_id = {0: low_id, 1: high_id}

        return [self._class_id_to_binary[int(v)] for v in y_ids]