import numpy as np
import torch
import torch.nn as nn
import time
import copy

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



def train_model(
    model,
    train_loader,
    val_loader,
    device,
    y_mean,
    y_std,
    model_config: dict,
    seed: int | None = None,
    batch_size: int | None = None,
    max_epochs: int = 100,
    patience: int = 15,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_fn = nn.MSELoss() #MSE Used in baseline
    #loss_fn = nn.SmoothL1Loss(beta=1)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_checkpoint = {}
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss": float(train_loss),
                "best_val_loss": float(best_val_loss),
                "y_mean": torch.as_tensor(y_mean, dtype=torch.float32),
                "y_std": torch.as_tensor(y_std, dtype=torch.float32),
                "model_config": copy.deepcopy(model_config),
                "training_config": {
                    "seed": seed,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
            }

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss = {train_loss:.6f} | "
            f"val_loss = {val_loss:.6f} | "
            f"best_val = {best_val_loss:.6f}"
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time

    best_checkpoint["elapsed_seconds"] = elapsed
    best_checkpoint["history"] = history

    return best_checkpoint, history

