class Preprocessor:
    def __init__(self):
        self._min_values = None
        self._max_values = None

        self._label_to_id = {}
        self._id_to_label = {}

    @property
    def label_to_id(self) -> dict:
        return self._label_to_id.copy()

    @property
    def id_to_label(self) -> dict:
        return self._id_to_label.copy()

    @property
    def min_values(self):
        return None if self._min_values is None else self._min_values.copy()

    @property
    def max_values(self):
        return None if self._max_values is None else self._max_values.copy()

    def normalize_inputs(self, input_data: list[list[float]]) -> list[list[float]]:
        if len(input_data) == 0:
            return input_data

        num_features = len(input_data[0])
        self._min_values = [float("inf")] * num_features
        self._max_values = [float("-inf")] * num_features

        for row in input_data:
            if len(row) != num_features:
                raise ValueError("All samples must have the same number of features.")

            for j in range(num_features):
                value = float(row[j])
                if value < self._min_values[j]:
                    self._min_values[j] = value
                if value > self._max_values[j]:
                    self._max_values[j] = value

        return self.transform_inputs(input_data)

    def transform_inputs(self, input_data: list[list[float]]) -> list[list[float]]:
        if self._min_values is None or self._max_values is None:
            return input_data

        num_features = len(self._min_values)
        input_data_norm = []

        for row in input_data:
            if len(row) != num_features:
                raise ValueError("Sample has wrong number of features.")

            new_row = []
            for j in range(num_features):
                mn = self._min_values[j]
                mx = self._max_values[j]
                value = float(row[j])

                if mx == mn:
                    new_row.append(0.0)
                else:
                    new_row.append((value - mn) / (mx - mn))

            input_data_norm.append(new_row)

        return input_data_norm

    def encode_labels(self, y: list) -> list[int]:
        encoded = []
        for label in y:
            encoded.append(self._encode_label(label))
        return encoded

    def _encode_label(self, label):
        if isinstance(label, float) and not label.is_integer():
            raise ValueError("Numeric label must be an integer class id.")

        if isinstance(label, (int, float)):
            v = int(label)
            if v not in self._id_to_label:
                self._id_to_label[v] = v
                self._label_to_id[v] = v
            return v

        if label not in self._label_to_id:
            new_id = len(self._label_to_id)
            self._label_to_id[label] = new_id
            self._id_to_label[new_id] = label

        return self._label_to_id[label]

    def transform_label(self, label) -> int:
        if isinstance(label, float) and not label.is_integer():
            raise ValueError("Numeric label must be an integer class id.")

        if isinstance(label, (int, float)):
            v = int(label)
            if v not in self._id_to_label:
                raise ValueError("Unknown label (not seen in training).")
            return v

        if label not in self._label_to_id:
            raise ValueError("Unknown label (not seen in training).")

        return self._label_to_id[label]