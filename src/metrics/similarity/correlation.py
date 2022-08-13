import torch


def correlation_based_similarity(x, y):
    t = torch.vstack((x.squeeze(2), y.squeeze(2)))
    p = torch.corrcoef(t)[0, 1]
    return torch.sqrt(2 * (1 - p))
