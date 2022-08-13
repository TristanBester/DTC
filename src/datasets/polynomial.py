import numpy as np
import torch
from torch.utils.data import Dataset


class PolynomialDataset(Dataset):
    def __init__(self, X_path, Y_path) -> None:
        super().__init__()

        with open(X_path + "/X.npy", "rb") as f:
            self.X = np.load(f)

        with open(Y_path + "/Y.npy", "rb") as f:
            self.Y = np.load(f)

    def __len__(self):
        return self.X.shape[0] / self.ba

    def __getitem__(self, index):
        x = torch.tensor(self.X[index]).to(torch.float).unsqueeze(-1)
        y = torch.tensor(self.Y[index]).to(torch.float).unsqueeze(-1)
        return x, y

