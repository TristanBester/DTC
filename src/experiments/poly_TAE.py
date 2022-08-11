import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.datasets.polynomial import PolynomialDataset
from src.models.TAE import TAE


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def incremental_average(ave, n_val, n):
    if ave is None:
        return n_val

    ave = ave + (n_val - ave) / float(n)
    return ave


def train_one_epoch(model, optimizer, criterion, data_loader, device, scheduler):
    model.train()

    ave_loss = None
    pbar = tqdm(data_loader, leave=True, total=len(data_loader))

    for n, (x, _) in enumerate(pbar):
        x = x.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, x)
        loss.backward()
        optimizer.step()

        ave_loss = incremental_average(
            ave_loss, loss.item(), (n + 1) * data_loader.batch_size
        )
        pbar.set_description(f"loss - {round(ave_loss, 4)}")
    return ave_loss


def validate(model, criterion, data_loader, device):
    model.eval()

    ave_loss = 0
    pbar = tqdm(data_loader, leave=True, total=len(data_loader))

    with torch.no_grad():
        for _, (x, _) in enumerate(pbar):
            x = x.to(device)

            outputs = model(x)
            loss = criterion(outputs, x)

            ave_loss += loss.item()
    return ave_loss / len(data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default="test")
    args = parser.parse_args()

    run = wandb.init(project="DTC", name=args.run_name)
    dataset = PolynomialDataset(
        run,
        "tristanbester1/DTC/polynomial_dataset_X:v0",
        "tristanbester1/DTC/polynomial_dataset_Y:v0",
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    model = TAE(
        input_dim=1,
        seq_len=100,
        cnn_kernel=10,
        cnn_stride=3,
        mp_kernel=10,
        mp_stride=3,
        lstm_hidden_dim=8,
        upsample_scale=2,
        deconv_kernel=10,
        deconv_stride=6,
    )

    device = torch.device(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, threshold=0.01, verbose=True,
    )

    tae_artifact = wandb.Artifact(f"TAE-{wandb.run.id}", type="model")

    lowest_val = 999

    for i in range(1000):
        train_loss = train_one_epoch(
            model, optimizer, criterion, train_loader, device, None,
        )
        val_loss = validate(model, criterion, test_loader, device)
        scheduler.step(val_loss)

        wandb.log(
            {"train_loss": train_loss, "test_loss": val_loss, "lr": get_lr(optimizer)}
        )

        if val_loss < lowest_val:
            torch.save(model.state_dict(), f"./TAE_best.pt")
            tae_artifact.add_file("./TAE_best.pt", "model.pt")
            wandb.log_artifact(tae_artifact, aliases=["latest", "best"])

