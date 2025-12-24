# ML/binary_pipeline.py

from ML.preprocessor import Preprocessor
from ML.trainer_binary import BinaryPerceptronTrainer


class BinaryPerceptronPipeline:

    def __init__(self, preprocessor: Preprocessor, trainer: BinaryPerceptronTrainer):
        self._preprocessor = preprocessor
        self._trainer = trainer

        # mapping between model output (0/1 or -1/+1) and class ids (0..k)
        self._model_out_to_class_id: dict[int, int] | None = None
        self._class_id_to_model_out: dict[int, int] | None = None

        self._label_mode: str | None = None  # "01" for step, "pm1" for sign

    @property
    def preprocessor(self) -> Preprocessor:
        return self._preprocessor

    @property
    def trainer(self) -> BinaryPerceptronTrainer:
        return self._trainer

    @property
    def label_mode(self) -> str | None:
        return self._label_mode

    def fit(self, features: list[list[float]], labels: list) -> None:
        if len(features) == 0:
            return
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length.")

        X = self._preprocessor.normalize_inputs(features)

        y_ids = self._preprocessor.encode_labels(labels)

        expected_outputs = self._detect_activation_outputs()

        y_model = self._map_two_classes_to_expected_outputs(y_ids, expected_outputs)

        self._trainer.train(X, y_model, random_init=True)

    def predict(self, x_new: list[float]):
        if self._model_out_to_class_id is None:
            raise ValueError("Pipeline is not fitted yet.")

        x_fixed = self._preprocessor.transform_inputs([x_new])[0]
        pred_out = int(self._trainer.predict(x_fixed))

        if pred_out not in self._model_out_to_class_id:
            raise ValueError(
                f"Model predicted value {pred_out} which is not in allowed outputs {list(self._model_out_to_class_id.keys())}."
            )

        class_id = self._model_out_to_class_id[pred_out]
        return self._preprocessor.id_to_label.get(class_id, class_id)


    def _detect_activation_outputs(self) -> tuple[int, int]:

        act = self._trainer._perceptron._activation_function

        neg = int(act(-1))
        pos = int(act(+1))

        if (neg, pos) == (0, 1):
            self._label_mode = "01"
            return 0, 1

        if (neg, pos) == (-1, 1):
            self._label_mode = "pm1"
            return -1, 1

        raise ValueError(
            f"Unsupported activation outputs: activation(-1)={neg}, activation(+1)={pos}. "
            f"Expected (0,1) for Step or (-1,1) for Sign."
        )

    def _map_two_classes_to_expected_outputs(
        self, y_ids: list[int], expected_outputs: tuple[int, int]
    ) -> list[int]:

        unique = sorted(set(int(v) for v in y_ids))
        if len(unique) != 2:
            raise ValueError("Binary classifier requires exactly 2 classes.")

        low_id, high_id = unique[0], unique[1]
        out_low, out_high = expected_outputs[0], expected_outputs[1]

        # class_id -> model_output
        self._class_id_to_model_out = {low_id: out_low, high_id: out_high}

        # model_output -> class_id
        self._model_out_to_class_id = {out_low: low_id, out_high: high_id}

        return [self._class_id_to_model_out[int(v)] for v in y_ids]