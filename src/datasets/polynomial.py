import numpy as np
import torch
from torch.utils.data import Dataset

import wandb


class PolynomialDataset(Dataset):
    def __init__(self, wandb_run, artifact_name_X, artifact_name_Y) -> None:
        super().__init__()

        artifact_X = wandb_run.use_artifact(artifact_name_X, type="dataset")
        artifact_Y = wandb_run.use_artifact(artifact_name_Y, type="dataset")

        X_path = artifact_X.download()
        Y_path = artifact_Y.download()

        with open(X_path + "/X.npy", "rb") as f:
            self.X = np.load(f)

        with open(Y_path + "/Y.npy", "rb") as f:
            self.Y = np.load(f)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.X[index]).to(torch.float32).unsqueeze(-1)
        y = torch.tensor(self.Y[index]).to(torch.float32).unsqueeze(-1)

        return x, y


if __name__ == "__main__":
    run = wandb.init(project="DTC", name="test")
    dataset = PolynomialDataset(
        run,
        "tristanbester1/DTC/polynomial_dataset_X:v0",
        "tristanbester1/DTC/polynomial_dataset_Y:v0",
    )

    for (i, x) in dataset:
        print(i, x)

