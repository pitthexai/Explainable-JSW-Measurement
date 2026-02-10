import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def masked_mse_loss(predictions, targets, mask):
    loss = (predictions - targets)**2
    loss = loss * mask
    return loss.sum()/mask.sum()

def compute_metrics(preds, targets, mask):
    mse = np.sum(((preds-targets)**2)*mask)/np.sum(mask)
    mae = np.sum(np.abs((preds-targets)*mask))/np.sum(mask)
    return mse, mae