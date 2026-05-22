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
        plt.plot([lo, hi], [lo, hi], linestyle="-.", color="brick")
        plt.xlabel(f"True {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"{split_name}: pred vs true — {label}")
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_pred_vs_true_density(
    y_true,
    y_pred,
    label_names,
    split_name,
    bins=80,
    cmap="viridis",
):
    """
    Density plot for predicted vs true values.

    Useful when scatter plots become too dense.
    """
    for j, label in enumerate(label_names):
        true = y_true[:, j]
        pred = y_pred[:, j]

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())

        plt.figure(figsize=(5.5, 5))
        h = plt.hist2d(
            true,
            pred,
            bins=bins,
            range=[[lo, hi], [lo, hi]],
            cmap=cmap,
        )

        plt.plot([lo, hi], [lo, hi], linestyle="-.", color="firebrick", linewidth=1.5)

        plt.xlabel(f"True {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"{split_name}: pred vs true density — {label}")
        plt.colorbar(label="count")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

##############################################################################

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


def plot_residual_vs_true_density(
    y_true,
    y_pred,
    label_names,
    split_name,
    gridsize=60,
    cmap="viridis",
):
    residual = y_pred - y_true

    for j, label in enumerate(label_names):
        true = y_true[:, j]
        res = residual[:, j]

        plt.figure(figsize=(6, 4.8))
        hb = plt.hexbin(
            true,
            res,
            gridsize=gridsize,
            mincnt=1,
            cmap=cmap,
        )

        plt.axhline(0.0, linestyle="--", color="firebrick", linewidth=1.5)

        plt.xlabel(f"True {label}")
        plt.ylabel("Residual: pred - true")
        plt.title(f"{split_name}: residual vs true density — {label}")
        plt.colorbar(hb, label="count")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

##############################################################################

def plot_mean_abs_error_vs_true_pred(
        
    y_true,
    y_pred,
    label_names,
    split_name,
    gridsize=60,
    cmap="magma",
):
    """
    Hexbin plot in true-predicted space, colored by mean absolute error.

    This shows where the model makes larger mistakes, not just where points are dense.
    """
    abs_error = np.abs(y_pred - y_true)

    for j, label in enumerate(label_names):
        true = y_true[:, j]
        pred = y_pred[:, j]
        err = abs_error[:, j]

        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())

        plt.figure(figsize=(5.8, 5.2))
        hb = plt.hexbin(
            true,
            pred,
            C=err,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            mincnt=1,
            cmap=cmap,
        )

        plt.plot([lo, hi], [lo, hi], linestyle="-.", color="white", linewidth=1.3)

        plt.xlabel(f"True {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"{split_name}: mean abs error map — {label}")
        plt.colorbar(hb, label=f"mean |error| in {label}")
        plt.grid(True, alpha=0.15)
        plt.tight_layout()
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


def plot_abs_error_vs_quantity_density(
    quantity,
    y_true,
    y_pred,
    label_names,
    quantity_name,
    split_name,
    gridsize=60,
    cmap="viridis",
):
    abs_error = np.abs(y_pred - y_true)

    quantity = np.asarray(quantity)

    for j, label in enumerate(label_names):
        err = abs_error[:, j]

        plt.figure(figsize=(6, 4.8))
        hb = plt.hexbin(
            quantity,
            err,
            gridsize=gridsize,
            mincnt=1,
            cmap=cmap,
        )

        plt.xlabel(quantity_name)
        plt.ylabel(f"|error| in {label}")
        plt.title(f"{split_name}: abs error vs {quantity_name} density — {label}")
        plt.colorbar(hb, label="count")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

##############################################################################

def plot_binned_error_vs_quantity(
    quantity,
    y_true,
    y_pred,
    label_names,
    quantity_name,
    split_name,
    n_bins=12,
):
    quantity = np.asarray(quantity)
    abs_error = np.abs(y_pred - y_true)

    bin_edges = np.linspace(quantity.min(), quantity.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for j, label in enumerate(label_names):
        mean_err = []
        median_err = []
        counts = []

        for k in range(n_bins):
            lo = bin_edges[k]
            hi = bin_edges[k + 1]

            if k == n_bins - 1:
                mask = (quantity >= lo) & (quantity <= hi)
            else:
                mask = (quantity >= lo) & (quantity < hi)

            counts.append(mask.sum())

            if mask.sum() == 0:
                mean_err.append(np.nan)
                median_err.append(np.nan)
            else:
                mean_err.append(abs_error[mask, j].mean())
                median_err.append(np.median(abs_error[mask, j]))

        mean_err = np.array(mean_err)
        median_err = np.array(median_err)
        counts = np.array(counts)

        plt.figure(figsize=(6.5, 4.5))
        plt.plot(bin_centers, mean_err, marker="o", label="mean |error|")
        plt.plot(bin_centers, median_err, marker="s", label="median |error|")

        plt.xlabel(quantity_name)
        plt.ylabel(f"|error| in {label}")
        plt.title(f"{split_name}: binned abs error vs {quantity_name} — {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"\n{label} — counts per bin:")
        print(counts)