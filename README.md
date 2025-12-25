# Interactive Perceptron Classifier

An educational yet fully functional implementation of the **Perceptron algorithm**, supporting:

- Binary classification
- Multi-class classification using **One-vs-All**
- Two activation functions: **Step** and **Sign**
- Interactive **GUI** for training, prediction, and visualization
- Clean architecture separating Domain, ML logic, Data, and GUI

---

## ✨ Features

- **Binary Perceptron**
  - Supports Step `{0,1}` and Sign `{-1,+1}` activations
  - Reports convergence, epochs, updates, and accuracy

- **One-vs-All (OVA) Multi-class Classification**
  - Automatically adapts targets to the chosen activation
  - Trains one perceptron per class
  - Provides per-class training statistics

- **GUI (Tkinter + Matplotlib)**
  - Load built-in or custom CSV datasets
  - Edit datasets interactively
  - Train models with configurable parameters
  - Visualize:
    - Data points
    - Decision boundaries (2D)
    - Predicted test point (highlighted)

- **Clean ML Pipeline**
  - Automatic normalization
  - Label encoding
  - Clear separation of concerns

---

## 🧠 Supported Activations

| Activation | Output Range |
|-----------|--------------|
| Step | `{0, 1}`      |
| Sign | `{-1, +1}`    |

Both activations are supported in:
- Binary classification
- One-vs-All classification

---

## 📂 Project Structure

```
project/
├── Domain/
│   └── perceptron.py
├── ML/
│   ├── preprocessor.py
│   ├── trainer_binary.py
│   ├── binary_pipeline.py
│   └── one_vs_all_pipeline.py
├── Utils/
│   └── activations.py
├── Data/
│   ├── dataset_loader.py
│   ├── registry.py
│   └── real/
├── GUI/
│   └── main_window.py
├── tests/
│   ├── test_binary.py
│   └── test_ova.py
└── README.md
```

---

## 📊 Dataset Format

CSV files must follow this structure:

```csv
x1,x2,...,label
1.2,3.4,ClassA
2.1,1.8,ClassB
```

- Features must be numeric
- Labels can be strings or numbers
- Number of features: **2 to 4** (GUI plotting available for 2D)

---

## 🚀 Running the GUI

```bash
python GUI/main_window.py
```

---

## 🧪 Running Tests (CLI)

### Binary tests
```bash
python tests/test_binary.py
```

### One-vs-All tests
```bash
python tests/test_ova.py
```

---

## 📈 Visualization

For 2D datasets, the GUI displays:
- Normalized data points
- Decision boundary lines
- Last predicted test point (red X)

All plots are rendered in **normalized feature space [0,1]** for stable axes.

---

## 📝 Notes

- Perceptron convergence depends on linear separability.
- In One-vs-All mode, individual classifiers may not converge even if overall accuracy is high.
  This is expected and correctly reported.

---

## 📜 License

For educational and academic use.
"""

Path("/mnt/data/README.md").write_text(readme, encoding="utf-8")
print("Saved /mnt/data/README.md")
