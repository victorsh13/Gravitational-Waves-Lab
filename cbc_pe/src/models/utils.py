from __future__ import annotations
import random
import numpy as np
import torch

from src.models.evaluate import regression_metrics




def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Más reproducible, aunque puede reducir algo el rendimiento.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def summarize_prediction_file(pred_path):
    data = np.load(pred_path, allow_pickle=True)

    pred_val = data["pred_val"]
    y_val = data["y_val"]
    label_names = data["label_names"].tolist()

    metrics_val = regression_metrics(y_val, pred_val, label_names, "val_std")

    row = {
        "file": pred_path.name,
        "val_MSE_global": metrics_val["MSE"].mean(),
        "val_MAE_global": metrics_val["MAE"].mean(),
    }

    for _, r in metrics_val.iterrows():
        label = r["label"]
        row[f"{label}_MSE"] = r["MSE"]
        row[f"{label}_MAE"] = r["MAE"]
        row[f"{label}_R2"] = r["R2"]

    return row