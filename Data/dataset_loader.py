import csv


def load_csv_dataset(file_path: str) -> tuple[list[list[float]], list]:
    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or len(fieldnames) < 2:
            raise ValueError("CSV must contain at least 1 feature column + label column.")

        label_col = fieldnames[-1]
        feature_cols = fieldnames[:-1]

        X = []
        y = []

        for row in reader:
            X.append([float(row[c]) for c in feature_cols])
            y.append(int(row[label_col]))

        return X, y