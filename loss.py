
import torch

def mse_loss(output, target):
    return ((output - target) ** 2).mean()
