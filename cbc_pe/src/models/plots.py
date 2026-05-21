import numpy as np
import matplotlib.pyplot as plt


def plot_pred_vs_true(y_true, y_pred, label_names, split_name):
    for j, label in enumerate(label_names):
        true = y_true[:, j]
        pred = y_pred[:, j]

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())

        plt.figure(figsize=(5, 5))
        plt.scatter(true, pred, s=10, alpha=0.5)
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel(f"True {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"{split_name}: pred vs true — {label}")
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_residuals(y_true, y_pred, label_names, split_name):
    residual = y_pred - y_true

    for j, label in enumerate(label_names):
        plt.figure(figsize=(6, 4))
        plt.hist(residual[:, j], bins=50, alpha=0.8)
        plt.axvline(0.0, linestyle="--")
        plt.xlabel(f"Residual pred - true ({label})")
        plt.ylabel("Count")
        plt.title(f"{split_name}: residual distribution — {label}")
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_residual_vs_true(y_true, y_pred, label_names, split_name):
    residual = y_pred - y_true

    for j, label in enumerate(label_names):
        plt.figure(figsize=(6, 4))
        plt.scatter(y_true[:, j], residual[:, j], s=10, alpha=0.5)
        plt.axhline(0.0, linestyle="--")
        plt.xlabel(f"True {label}")
        plt.ylabel("Residual pred - true")
        plt.title(f"{split_name}: residual vs true — {label}")
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_abs_error_vs_quantity(quantity, y_true, y_pred, label_names, quantity_name, split_name):
    abs_error = np.abs(y_pred - y_true)

    for j, label in enumerate(label_names):
        plt.figure(figsize=(6, 4))
        plt.scatter(quantity, abs_error[:, j], s=10, alpha=0.5)
        plt.xlabel(quantity_name)
        plt.ylabel(f"Absolute error in {label}")
        plt.title(f"{split_name}: abs error vs {quantity_name} — {label}")
        plt.grid(True, alpha=0.3)
        plt.show()

