import torch


def euclidean_distance(x, y):
    """
    Return (x.shape[0], y.shape[0]) matrix where each element is d(x_i, y_i) 
    where x_i is the i-th time series in x => x_i = x[i].
    """
    a = x.repeat(1, 1, y.shape[0]).permute(0, 2, 1)
    b = y.repeat(x.shape[0], 1, 1).reshape(a.shape)
    return torch.sqrt(torch.sum((a - b) ** 2, dim=2))

