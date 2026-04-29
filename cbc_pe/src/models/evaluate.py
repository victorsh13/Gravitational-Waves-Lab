import torch

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