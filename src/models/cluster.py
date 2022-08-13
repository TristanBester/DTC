import torch
import torch.nn as nn


class ClusteringLayer(nn.Module):
    def __init__(self, centroids, metric, alpha=1) -> None:
        super().__init__()
        self.centroids = nn.Parameter(centroids)
        self.metric = metric
        self.alpha = alpha

    def students_t_distribution_kernel(self, x, alpha):
        num = torch.pow((1 + x / alpha), -(alpha + 1) / 2)
        denom = num.sum(dim=1).reshape(-1, 1).repeat(1, self.centroids.shape[0])
        return num / denom

    def forward(self, x):
        D = self.metric(x, self.centroids)
        Q = self.students_t_distribution_kernel(D, self.alpha)
        return Q
