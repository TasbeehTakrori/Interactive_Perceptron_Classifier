from Utils.activations import step
from Domain.perceptron import Perceptron
from ML.preprocessor import Preprocessor
from ML.trainer_binary import BinaryPerceptronTrainer
from Data.dataset_loader import load_csv_dataset
import random
from Data.registry import DATASETS


def run_dataset(name: str, path: str) -> None:
    X, y = load_csv_dataset(path)

    p = Perceptron(num_features=len(X[0]), activation_function=step)
    prep = Preprocessor()
    trainer = BinaryPerceptronTrainer(
        perceptron=p,
        preprocessor=prep,
        learning_rate=0.1,
        max_epoch=50
    )

    trainer.train(X, y)

    print("\n" + "=" * 45)
    print("Dataset:", name)
    print("Path   :", path)
    print("Converged:", trainer.converged)
    print("Epochs   :", trainer.num_epoch)
    print("Accuracy :", trainer.accuracy)
    print("Weights  :", p.weights)
    print("Bias     :", p.bias)

    for xi, yi in zip(X, y):
        pred = trainer.predict(xi)
        print(f"X={xi} -> pred={pred}, y={yi}")


def main():
    random.seed(1)
    for name, path in DATASETS.items():
        run_dataset(name, path)


if __name__ == "__main__":
    main()