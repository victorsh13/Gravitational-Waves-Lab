import numpy as np
import torch    



# To compare the predictions of the model with the true values
@torch.no_grad()
def predict_on_loader(model, loader, device):
    """
    This function is used to predict the values of the model on a given loader.
    It returns the predicted values and the true values.
    Parameters:
    - model (torch.nn.Module): The model to be used for prediction.
    - loader (torch.utils.data.DataLoader): The loader containing the data to be predicted.
    - device (torch.device): The device on which the model and data are located.
    Returns:
    - preds (np.ndarray): The predicted values.
    - targets (np.ndarray): The true values.
    """
    model.eval()

    preds = []
    targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)

        pred = model(X_batch)

        preds.append(pred.cpu().numpy())
        targets.append(y_batch.numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    return preds, targets


# To obtain the mse errors per label
@torch.no_grad()
def compute_error_metrics(pred, y_true, label_names):
    """
    Computes the mean squared error (MSE) and mean absolute error (MAE) for each label.
    Returns a dictionary with the global MSE and a dictionary with the MSE and MAE per label.
    Parameters:
    - pred (np.ndarray): The predicted values.
    - y_true (np.ndarray): The true values.
    - label_names (list): A list of strings representing the names of the labels.
    Returns:
    - metrics (dict): A dictionary containing the global MSE and the MSE and MAE per label.
        The keys of the dictionary are "global_mse" and "per_label", respectively.
        The values of the "per_label" key are dictionaries with the keys "mse" and "mae",
        representing the MSE and MAE for each label.
    """
    mse = np.mean((pred - y_true) ** 2, axis=0)
    mae = np.mean(np.abs(pred - y_true), axis=0)

    metrics = {
        "global_mse": float(np.mean(mse)),
        "global_mae": float(np.mean(mae)),
        "per_label": {},
    }

    for i, name in enumerate(label_names):
        metrics["per_label"][name] = {
            "mse": float(mse[i]),
            "mae": float(mae[i]),
        }

    return metrics


def evaluate_global_mse(mse_per_label):
    return float(np.mean(mse_per_label))


@torch.no_grad()
def extract_predictions_and_embeddings(model, loader, device):
    """
    This function is used to extract the predictions and embeddings from the model on a given loader.
    It returns the predicted values, the embeddings, and the true values.
    Parameters:
    - model (torch.nn.Module): The model to be used for prediction.
    - loader (torch.utils.data.DataLoader): The loader containing the data to be predicted.
    - device (torch.device): The device on which the model and data are located.
    Returns:
    - preds (np.ndarray): The predicted values.
    - embs (np.ndarray): The embeddings.
    - targets (np.ndarray): The true values.
    """
    model.eval()

    all_pred = []
    all_emb = []
    all_y = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)

        pred, emb = model(X_batch, return_embedding=True)

        all_pred.append(pred.cpu().numpy())
        all_emb.append(emb.cpu().numpy())
        all_y.append(y_batch.numpy())

    pred = np.concatenate(all_pred, axis=0)
    emb = np.concatenate(all_emb, axis=0)
    y = np.concatenate(all_y, axis=0)

    return pred, emb, y

