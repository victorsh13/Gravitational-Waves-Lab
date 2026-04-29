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