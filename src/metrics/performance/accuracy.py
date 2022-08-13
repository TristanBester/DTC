import torch


def accuracy(preds, target):
    return (preds == target).to(torch.float).mean().item()
