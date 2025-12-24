import random

from Utils.activations import step
from ML.preprocessor import Preprocessor
from ML.one_vs_all_pipeline import OneVsAllPipeline
from Data.dataset_loader import load_csv_dataset


def test_one_vs_all_weather_activity():
    random.seed(1)

    X, y = load_csv_dataset("Data/real/weather_activity.csv")

    prep = Preprocessor()
    ova = OneVsAllPipeline(
        preprocessor=prep,
        learning_rate=0.1,
        max_epoch=100,
        activation_function=step
    )

    ova.train_one_vs_all(X, y)

    acc = ova.accuracy(X, y)

    print("\n" + "=" * 45)
    print("Dataset : Weather -> Activity")
    print("Path    : Data/real/weather_activity.csv")
    print("Classes :", [prep.id_to_label[c] for c in ova.class_ids])
    print("Accuracy:", acc)

    for xi, yi in zip(X, y):
        pred = ova.predict(xi)
        print(f"X={xi} -> pred={pred}, y={yi}")

    assert acc >= 0.9


if __name__ == "__main__":
    test_one_vs_all_weather_activity()