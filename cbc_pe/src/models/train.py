import numpy as np
import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train() # Red a modo entrenamiento

    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device) # Mueve los datos a la GPU/CPU
        y_batch = y_batch.to(device)  

        optimizer.zero_grad() # Borra los gradientes y los hace cero

        pred = model(X_batch) # Aplica el modelo y obtiene las predicciones
        loss = loss_fn(pred, y_batch) # Calcula el error de la predicción con la verdad

        loss.backward() # Calcula los gradientes de la función de error
        optimizer.step() # Actualiza los pesos del modelo

        batch_size = X_batch.shape[0]
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    mean_loss = total_loss / n_samples
    return mean_loss



# We don't want to compute the gradient calculation, just validation, no training
@torch.no_grad() 

def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    mean_loss = total_loss / n_samples
    return mean_loss

# To obtain the mse errors per label
@torch.no_grad()
def evaluate_per_label_mse(model, loader, device):
    model.eval()

    all_pred = []
    all_y = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch)

        all_pred.append(pred.cpu().numpy())
        all_y.append(y_batch.numpy())

    pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    mse_per_label = np.mean((pred - y_true) ** 2, axis=0)
    mae_per_label = np.mean(np.abs(pred - y_true), axis=0)

    return mse_per_label, mae_per_label

# To compare the predictions of the model with the true values
@torch.no_grad()
def predict_on_loader(model, loader, device):
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