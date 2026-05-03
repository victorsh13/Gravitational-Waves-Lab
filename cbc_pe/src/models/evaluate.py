import numpy as np
import torch    

@torch.no_grad()
def extract_predictions_and_embeddings(model, loader, device):
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

