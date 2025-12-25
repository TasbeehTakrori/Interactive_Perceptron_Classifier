# GUI/main_window.py
import os
import sys
import csv
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

if __package__ is None:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from Utils.activations import step, sign
from Domain.perceptron import Perceptron
from ML.preprocessor import Preprocessor
from ML.trainer_binary import BinaryPerceptronTrainer
from ML.binary_pipeline import BinaryPerceptronPipeline
from ML.one_vs_all_pipeline import OneVsAllPipeline
from Data.dataset_loader import load_csv_dataset
from Data.registry import DATASETS

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class InteractivePerceptronGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Interactive Perceptron Classifier")
        self.root.minsize(1180, 760)

        self._style = ttk.Style()
        try:
            self._style.theme_use("clam")
        except Exception:
            pass

        self._apply_modern_style()

        # ML objects (re-created on every Train)
        self._preprocessor = Preprocessor()
        self._binary_pipeline: BinaryPerceptronPipeline | None = None
        self._ova_pipeline: OneVsAllPipeline | None = None
        self._last_test_point_norm = None

        # UI / data state
        self._current_num_features = 2
        self._last_train_X_raw = None
        self._last_train_y = None
        self._last_mode = "Binary"

        self._build_ui()
        self._refresh_feature_columns(self._current_num_features)
        self._refresh_test_entries(self._current_num_features)
        self._refresh_dataset_dropdown()

    # ---------------- STYLE ----------------

    def _apply_modern_style(self):
        self._style.configure("Header.TLabel", font=("TkDefaultFont", 16, "bold"))
        self._style.configure("SubHeader.TLabel", font=("TkDefaultFont", 10))
        self._style.configure("Card.TLabelframe", padding=10)
        self._style.configure("Card.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        self._style.configure("Primary.TButton", font=("TkDefaultFont", 10, "bold"), padding=6)
        self._style.configure("TButton", padding=6)
        self._style.configure("TCombobox", padding=4)
        self._style.configure("TSpinbox", padding=4)

    # ---------------- UI BUILD ----------------

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        container = ttk.Frame(self.root, padding=14)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        # Top header
        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Interactive Perceptron Classifier", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Binary + One-vs-All | Train, Predict, and Visualize (2D)",
            style="SubHeader.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Main layout: left table / right controls
        main = ttk.Frame(container)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        # LEFT: Training data card
        left = ttk.Labelframe(main, text="Training Dataset", style="Card.TLabelframe")
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)

        # Dataset selector + quick actions
        topbar = ttk.Frame(left)
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        topbar.columnconfigure(1, weight=1)

        ttk.Label(topbar, text="Built-in dataset:").grid(row=0, column=0, sticky="w")
        self.var_dataset = tk.StringVar(value="")
        self.ds_combo = ttk.Combobox(topbar, textvariable=self.var_dataset, values=[], state="readonly")
        self.ds_combo.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        self.ds_combo.bind("<<ComboboxSelected>>", lambda e: self._load_selected_dataset_into_table())

        ttk.Button(topbar, text="Load CSV", command=self._load_csv).grid(row=0, column=2, padx=4)
        ttk.Button(topbar, text="Save CSV", command=self._save_csv).grid(row=0, column=3, padx=4)

        # Feature count & table operations
        opbar = ttk.Frame(left)
        opbar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        opbar.columnconfigure(10, weight=1)

        ttk.Label(opbar, text="Features:").grid(row=0, column=0, sticky="w")
        self.var_num_features = tk.IntVar(value=2)
        spin = ttk.Spinbox(
            opbar, from_=2, to=4, width=6,
            textvariable=self.var_num_features,
            command=self._on_change_num_features
        )
        spin.grid(row=0, column=1, padx=(8, 12), sticky="w")

        ttk.Button(opbar, text="Add row", command=self._add_row).grid(row=0, column=2, padx=4)
        ttk.Button(opbar, text="Remove selected", command=self._remove_row).grid(row=0, column=3, padx=4)
        ttk.Button(opbar, text="Clear table", command=self._clear_table).grid(row=0, column=4, padx=4)

        # Table
        self.tree = ttk.Treeview(left, columns=("x1", "x2", "label"), show="headings", height=14)
        self.tree.grid(row=2, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        vsb.grid(row=2, column=1, sticky="ns")
        hsb = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        hsb.grid(row=3, column=0, sticky="ew")

        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.bind("<Double-1>", self._on_double_click_cell)

        # RIGHT TOP: Controls card
        right_top = ttk.Labelframe(main, text="Training Controls", style="Card.TLabelframe")
        right_top.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        right_top.columnconfigure(1, weight=1)

        ttk.Label(right_top, text="Mode:").grid(row=0, column=0, sticky="w", pady=4)
        self.var_mode = tk.StringVar(value="Binary")
        mode_combo = ttk.Combobox(right_top, textvariable=self.var_mode, values=["Binary", "One-vs-All"], state="readonly")
        mode_combo.grid(row=0, column=1, sticky="ew", pady=4)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_mode_changed())

        ttk.Label(right_top, text="Learning rate:").grid(row=1, column=0, sticky="w", pady=4)
        self.var_lr = tk.StringVar(value="0.1")
        ttk.Entry(right_top, textvariable=self.var_lr).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(right_top, text="Max epochs:").grid(row=2, column=0, sticky="w", pady=4)
        self.var_epochs = tk.StringVar(value="100")
        ttk.Entry(right_top, textvariable=self.var_epochs).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(right_top, text="Activation:").grid(row=3, column=0, sticky="w", pady=4)
        self.var_activation = tk.StringVar(value="Step")
        act_combo = ttk.Combobox(right_top, textvariable=self.var_activation, values=["Step", "Sign"], state="readonly")
        act_combo.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Separator(right_top).grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Button(right_top, text="Train", style="Primary.TButton", command=self._train).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=4
        )
        ttk.Button(right_top, text="Reset model", command=self._reset_model).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=4
        )

        # RIGHT BOTTOM: Predict + Plot card
        right_bottom = ttk.Labelframe(main, text="Predict + Plot", style="Card.TLabelframe")
        right_bottom.grid(row=1, column=1, sticky="nsew")
        right_bottom.columnconfigure(0, weight=1)
        right_bottom.rowconfigure(6, weight=1)

        self.lbl_status = ttk.Label(right_bottom, text="Status: Not trained yet.")
        self.lbl_status.grid(row=0, column=0, sticky="w", pady=(0, 6))

        # Test input row
        testbar = ttk.Frame(right_bottom)
        testbar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(testbar, text="Test sample:").grid(row=0, column=0, sticky="w")

        self._test_entries = []
        for j in range(4):
            ent = ttk.Entry(testbar, width=10)
            ent.grid(row=0, column=1 + j, padx=4)
            self._test_entries.append(ent)

        ttk.Button(testbar, text="Predict", command=self._predict_from_test).grid(row=0, column=5, padx=(10, 0))

        self.lbl_pred = ttk.Label(testbar, text="Predicted: -", font=("TkDefaultFont", 10, "bold"))
        self.lbl_pred.grid(row=0, column=6, padx=(10, 0))

        # Plot options
        plot_opts = ttk.Frame(right_bottom)
        plot_opts.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        plot_opts.columnconfigure(1, weight=1)

        ttk.Label(plot_opts, text="Plot:").grid(row=0, column=0, sticky="w")
        self.var_plot_mode = tk.StringVar(value="Points")
        plot_combo = ttk.Combobox(plot_opts, textvariable=self.var_plot_mode, values=["Points", "Decision lines"], state="readonly")
        plot_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        plot_combo.bind("<<ComboboxSelected>>", lambda e: self._render_plot())

        # Plot canvas
        self.fig = Figure(figsize=(4.4, 3.1), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_bottom)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew", pady=(2, 8))

        # Log
        ttk.Label(right_bottom, text="Log:").grid(row=4, column=0, sticky="w")
        self.txt_log = tk.Text(right_bottom, height=7, wrap="word")
        self.txt_log.grid(row=5, column=0, sticky="nsew")
        self.txt_log.insert("end", "Ready.\n")
        self.txt_log.configure(state="disabled")

        self._clear_plot()

    # ---------------- TABLE OPS ----------------

    def _refresh_feature_columns(self, num_features: int):
        cols = [f"x{i+1}" for i in range(num_features)] + ["label"]
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=110, anchor="center")
        self._current_num_features = num_features

    def _on_change_num_features(self):
        n = int(self.var_num_features.get())
        self._refresh_feature_columns(n)
        self._refresh_test_entries(n)

    def _refresh_test_entries(self, num_features: int):
        for j, ent in enumerate(self._test_entries):
            if j < num_features:
                ent.configure(state="normal")
            else:
                ent.delete(0, "end")
                ent.configure(state="disabled")

    def _add_row(self):
        n = self._current_num_features
        values = ["0.0"] * n + ["0"]
        self.tree.insert("", "end", values=values)

    def _remove_row(self):
        sel = self.tree.selection()
        for iid in sel:
            self.tree.delete(iid)

    def _clear_table(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)

    def _on_double_click_cell(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        row_id = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        col_index = int(col.replace("#", "")) - 1

        x, y, w, h = self.tree.bbox(row_id, col)
        value = self.tree.item(row_id, "values")[col_index]

        entry = ttk.Entry(self.tree)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, value)
        entry.focus_set()

        def save_edit(_=None):
            new_val = entry.get().strip()
            vals = list(self.tree.item(row_id, "values"))
            vals[col_index] = new_val
            self.tree.item(row_id, values=vals)
            entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def _load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            X, y = load_csv_dataset(path)
        except Exception as e:
            messagebox.showerror("CSV load error", str(e))
            return
        self._load_data_into_table(X, y)
        self._log(f"Loaded CSV: {path}")

    def _save_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        X, y = self._read_table_as_xy()
        if X is None:
            return

        headers = [f"x{i+1}" for i in range(len(X[0]))] + ["label"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for xi, yi in zip(X, y):
                w.writerow(list(xi) + [yi])

        self._log(f"Saved CSV: {path}")

    # ---------------- DATASETS ----------------

    def _refresh_dataset_dropdown(self):
        names = list(DATASETS.keys())
        self.ds_combo["values"] = names
        if names:
            self.var_dataset.set(names[0])
            self._load_selected_dataset_into_table()

    def _load_selected_dataset_into_table(self):
        name = self.var_dataset.get().strip()
        if not name:
            return
        path = DATASETS.get(name)
        if not path:
            return
        try:
            X, y = load_csv_dataset(path)
        except Exception as e:
            messagebox.showerror("Dataset load error", str(e))
            return

        self._load_data_into_table(X, y)
        self._log(f"Loaded built-in dataset: {name}")

    def _load_data_into_table(self, X, y):
        if not X:
            return
        n_features = len(X[0])
        if n_features < 2 or n_features > 4:
            messagebox.showerror("Invalid dataset", "Dataset must have 2..4 features.")
            return

        self.var_num_features.set(n_features)
        self._refresh_feature_columns(n_features)
        self._refresh_test_entries(n_features)
        self._reset_model(clear_log=False)

        self._clear_table()
        for xi, yi in zip(X, y):
            row = [str(v) for v in xi] + [str(yi)]
            self.tree.insert("", "end", values=row)

        self._reset_model(clear_log=False)

    # ---------------- READ TABLE ----------------

    def _read_table_as_xy(self):
        rows = []
        for iid in self.tree.get_children():
            rows.append(self.tree.item(iid, "values"))

        if not rows:
            messagebox.showwarning("No data", "Please add training rows first.")
            return None, None

        n_features = self._current_num_features

        X = []
        y = []
        for r in rows:
            if len(r) != n_features + 1:
                messagebox.showerror("Bad row", "Row has wrong number of columns.")
                return None, None
            try:
                xi = [float(r[i]) for i in range(n_features)]
            except Exception:
                messagebox.showerror("Bad value", f"Features must be numeric: {r}")
                return None, None
            yi = r[-1]
            X.append(xi)
            y.append(yi)

        return X, y

    # ---------------- MODEL OPS ----------------

    def _get_activation(self):
        name = self.var_activation.get()
        return step if name == "Step" else sign

    def _train(self):
        X, y = self._read_table_as_xy()
        if X is None:
            return

        try:
            lr = float(self.var_lr.get())
            max_epoch = int(self.var_epochs.get())
        except Exception:
            messagebox.showerror("Invalid settings", "Learning rate must be float and max epochs must be int.")
            return

        mode = self.var_mode.get()
        act_name = self.var_activation.get()
        act = self._get_activation()

        # New training run => new preprocessor
        self._preprocessor = Preprocessor()
        self._binary_pipeline = None
        self._ova_pipeline = None
        self._last_test_point_norm = None

        self._last_train_X_raw = X
        self._last_train_y = y
        self._last_mode = mode

        self._log("=" * 70)
        self._log(f"TRAIN | mode={mode} | lr={lr} | max_epoch={max_epoch} | activation={act_name}")

        if mode == "One-vs-All":
            self._log(f"OVA targets will match activation ({act_name}): "
                      f"{'{0,1}' if act_name == 'Step' else '{-1,+1}'}")

        try:
            if mode == "Binary":
                p = Perceptron(num_features=len(X[0]), activation_function=act)
                trainer = BinaryPerceptronTrainer(perceptron=p, learning_rate=lr, max_epoch=max_epoch)
                pipeline = BinaryPerceptronPipeline(preprocessor=self._preprocessor, trainer=trainer)

                pipeline.fit(X, y)

                self._binary_pipeline = pipeline
                self._ova_pipeline = None

                self.lbl_status.configure(
                    text=f"Status: Trained (Binary) | epochs={trainer.num_epoch} | updates={trainer.num_updates} | "
                         f"converged={trainer.converged} | acc={trainer.accuracy:.3f}"
                )

                self._log(
                    f"epochs={trainer.num_epoch}, updates={trainer.num_updates}, converged={trainer.converged}, acc={trainer.accuracy:.3f}")
                self._log(f"weights={p.weights}")
                self._log(f"bias={p.bias}")

            else:
                ova = OneVsAllPipeline(
                    preprocessor=self._preprocessor,
                    learning_rate=lr,
                    max_epoch=max_epoch,
                    activation_function=act
                )

                ova.train_one_vs_all(X, y)

                self.lbl_status.configure(
                    text=f"Status: Trained (One-vs-All) | classes={len(ova.class_ids)} | acc={ova.accuracy:.3f}"
                )

                self._log(f"classes={len(ova.class_ids)} | acc={ova.accuracy:.3f}")
                self._log(f"class_ids={ova.class_ids}")

                # --- NEW: show per-class training stats ---
                stats = ova.train_stats
                if stats:
                    # summary
                    epochs_list = [stats[c]["epochs"] for c in ova.class_ids]
                    updates_list = [stats[c]["updates"] for c in ova.class_ids]
                    conv_list = [1 if stats[c]["converged"] else 0 for c in ova.class_ids]

                    self._log(f"OVA summary: avg_epochs={sum(epochs_list) / len(epochs_list):.2f}, "
                              f"total_updates={sum(updates_list)}, "
                              f"converged_models={sum(conv_list)}/{len(conv_list)}")

                    # per-class lines
                    for class_id in ova.class_ids:
                        label = self._preprocessor.id_to_label.get(class_id, class_id)
                        s = stats[class_id]
                        self._log(f"  class={label} | epochs={s['epochs']} | updates={s['updates']} | "
                                  f"converged={s['converged']} | acc_binary={s['acc_binary']:.3f}")

                self._ova_pipeline = ova
                self._binary_pipeline = None

                self.lbl_status.configure(
                    text=f"Status: Trained (One-vs-All) | classes={len(ova.class_ids)} | acc={ova.accuracy:.3f}"
                )
                self._log(f"classes={len(ova.class_ids)} | acc={ova.accuracy:.3f}")
                self._log(f"class_ids={ova.class_ids}")

        except Exception as e:
            messagebox.showerror("Training error", str(e))
            self._log(f"ERROR: {e}")
            return

        self._render_plot()

    def _predict_from_test(self):
        mode = self.var_mode.get()
        n = self._current_num_features

        vals = []
        for j in range(n):
            ent = self._test_entries[j]
            try:
                vals.append(float(ent.get().strip()))
            except Exception:
                messagebox.showerror("Invalid input", "Please enter numeric feature values for the test sample.")
                return

        try:
            if mode == "Binary":
                if self._binary_pipeline is None:
                    messagebox.showwarning("Not trained", "Train the model first.")
                    return
                pred_label = self._binary_pipeline.predict(vals)
                self.lbl_pred.configure(text=f"Predicted: {pred_label}")
                self._log(f"PREDICT(Binary): x={vals} -> {pred_label}")

            else:
                if self._ova_pipeline is None:
                    messagebox.showwarning("Not trained", "Train the model first.")
                    return
                pred_label = self._ova_pipeline.predict(vals)
                self.lbl_pred.configure(text=f"Predicted: {pred_label}")
                self._log(f"PREDICT(OVA): x={vals} -> {pred_label}")
            # store test point (normalized) for plotting
            x_norm = self._preprocessor.transform_inputs([vals])[0]
            self._last_test_point_norm = x_norm

            self._render_plot()

        except Exception as e:
            messagebox.showerror("Predict error", str(e))
            self._log(f"ERROR: {e}")

    def _reset_model(self, clear_log=True):
        self._binary_pipeline = None
        self._ova_pipeline = None
        self._last_train_X_raw = None
        self._last_train_y = None
        self._last_test_point_norm = None
        self.lbl_status.configure(text="Status: Not trained yet.")
        self.lbl_pred.configure(text="Predicted: -")
        self._clear_plot()
        if clear_log:
            self._log("Model reset.")

    def _on_mode_changed(self):
        self._reset_model(clear_log=False)
        self._log(f"Mode changed to: {self.var_mode.get()}")

    # ---------------- LOG ----------------

    def _log(self, msg: str):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    # ---------------- PLOT ----------------

    def _clear_plot(self):
        self.ax.clear()
        self.ax.set_title("Plot (train first)")
        self.canvas.draw()

    def _render_plot(self):
        self.ax.clear()

        if self._last_train_X_raw is None or self._last_train_y is None:
            self.ax.set_title("Plot (train first)")
            self.canvas.draw()
            return

        X_raw = self._last_train_X_raw
        y = self._last_train_y
        n_features = len(X_raw[0])

        if n_features != 2:
            self.ax.set_title("Plot available for 2 features only")
            self.canvas.draw()
            return

        if self._binary_pipeline is None and self._ova_pipeline is None:
            self.ax.set_title("Plot (train first)")
            self.canvas.draw()
            return

        # plot in normalized space using the SAME preprocessor
        X_norm = self._preprocessor.transform_inputs(X_raw)

        plot_mode = self.var_plot_mode.get()
        mode = self._last_mode

        # stable axes always [0,1]
        self.ax.set_xlim(-0.05, 1.05)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.grid(True, linestyle="--", alpha=0.35)

        uniq = list(dict.fromkeys(y))

        # points
        for lab in uniq:
            px = [X_norm[i][0] for i in range(len(X_norm)) if y[i] == lab]
            py = [X_norm[i][1] for i in range(len(X_norm)) if y[i] == lab]
            self.ax.scatter(px, py, label=str(lab), s=45)

        # decision lines
        if plot_mode == "Decision lines":
            if mode == "Binary" and self._binary_pipeline is not None:
                model = self._binary_pipeline.trainer._perceptron
                self._plot_line_for_model(model, label="boundary")

            if mode == "One-vs-All" and self._ova_pipeline is not None:
                for class_id, model in self._ova_pipeline._perceptrons.items():
                    lab = self._preprocessor.id_to_label.get(class_id, class_id)
                    self._plot_line_for_model(model, label=str(lab))

        title = "Binary" if mode == "Binary" else "One-vs-All"
        self.ax.set_title(f"{title} (Normalized space 0..1)")
        self.ax.set_xlabel("x1 (norm)")
        self.ax.set_ylabel("x2 (norm)")

        if self._last_test_point_norm is not None:
            tx, ty = self._last_test_point_norm[0], self._last_test_point_norm[1]
            self.ax.scatter(
                [tx], [ty],
                c="red",
                marker="X",
                s=160,
                linewidths=2,
                edgecolors="black",
                zorder=10,
                label="Test point"
            )

        self.ax.legend(loc="best", fontsize=8)

        self.canvas.draw()

    def _plot_line_for_model(self, model: Perceptron, label: str):
        w = model.weights
        b = model.bias
        if len(w) < 2:
            return

        w1, w2 = float(w[0]), float(w[1])

        xmin, xmax = 0.0, 1.0

        if abs(w2) < 1e-12:
            if abs(w1) < 1e-12:
                return
            x0 = -b / w1
            self.ax.axvline(x=x0, linestyle="--", linewidth=1.5, label=f"line:{label}")
            return

        x_line = [xmin, xmax]
        y_line = [-(w1 * x + b) / w2 for x in x_line]
        self.ax.plot(x_line, y_line, linestyle="--", linewidth=1.5, label=f"line:{label}")


def main():
    random.seed(1)
    root = tk.Tk()
    InteractivePerceptronGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()