import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from datasets import PolynomialDataset
from metrics.performance import accuracy
from metrics.similarity import euclidean_distance
from models import ClusteringLayer, Decoder, Encoder

# ENCODER_CONFIG = {
#     "input_dim": 1,
#     "seq_len": 100,
#     "cnn_kernel": 10,
#     "cnn_stride": 3,
#     "mp_kernel": 10,
#     "mp_stride": 3,
#     "lstm_hidden_dim": 1,
# }

# DECODER_CONFIG = {
#     "seq_len": 100,
#     "cnn_kernel": 14,
#     "cnn_stride": 2,
#     "mp_kernel": 5,
#     "mp_stride": 2,
#     "upsample_scale": 2,
#     "input_dim": 1,
#     "hidden_dim": 1,
#     "deconv_kernel": 10,
#     "deconv_stride": 6,
# }


ENCODER_CONFIG = {
    "input_dim": 1,
    "seq_len": 100,
    "cnn_kernel": 10,
    "cnn_stride": 2,
    "mp_kernel": 4,
    "mp_stride": 2,
    "lstm_hidden_dim": 1,
}

DECODER_CONFIG = {
    "seq_len": 100,
    "cnn_kernel": 10,
    "cnn_stride": 2,
    "mp_kernel": 4,
    "mp_stride": 2,
    "upsample_scale": 2,
    "input_dim": 1,
    "hidden_dim": 1,
    "deconv_kernel": 14,
    "deconv_stride": 2,
}


class EarlyStopping:
    def __init__(self, patience=1):
        self.min_loss = float("inf")
        self.patience = patience
        self.patience_counter = 0

    def __call__(self, epoch_loss):
        if epoch_loss is None:
            return False

        if epoch_loss < self.min_loss:
            self.min_loss = epoch_loss
            self.patience_counter = 0
            return False

        if self.patience_counter > self.patience:
            return True

        self.patience_counter += 1
        return False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_autoencoder_one_epoch(model, optimizer, criterion, data_loader, device):
    model.train()
    total_loss = 0
    batch_counter = 0

    for x, _ in data_loader:
        x = x.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_counter += 1
    return total_loss / batch_counter


def evaluate_autoencoder(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0
    batch_counter = 0

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            outputs = model(x)
            loss = criterion(outputs, x)

            total_loss += loss.item()
            batch_counter += 1
    return total_loss / batch_counter


def init_centroids(encoder, data_loader, device, metric, n_clusters):
    encoder.eval()
    latent = []

    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            latent.append(encoder(x))
    latent = torch.cat(latent)

    similarity_matrix = metric(latent, latent)
    clustering_assignments = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="precomputed", linkage="complete",
    ).fit_predict(similarity_matrix)

    centroids = []
    for i in np.unique(clustering_assignments):
        centroid = latent[clustering_assignments == i].mean(dim=0).unsqueeze(0)
        centroids.append(centroid)
    centroids = torch.cat(centroids)
    return centroids


def target_distribution(Q):
    F = Q.sum(dim=0)
    num = (Q ** 2) / F
    denom = num.sum(dim=1).reshape(-1, 1).repeat(1, Q.shape[-1])
    return num / denom


def train_dtc_one_epoch(
    encoder,
    decoder,
    clustering,
    ae_optimizer,
    cl_optimizer,
    ae_criterion,
    cl_criterion,
    data_loader,
    device,
):
    encoder.train()
    decoder.train()
    clustering.train()

    ae_total_loss = 0
    cl_total_loss = 0
    batch_counter = 0

    for x, _ in data_loader:
        x = x.to(device)

        ae_optimizer.zero_grad()
        cl_optimizer.zero_grad()

        latent = encoder(x)
        x_prime = decoder(latent)

        Q = clustering(latent)
        P = target_distribution(Q)
        log_Q, log_P = torch.log(Q), torch.log(P)

        ae_loss = ae_criterion(x, x_prime)
        cl_loss = cl_criterion(log_Q, log_P)

        ae_loss.backward(retain_graph=True)
        cl_loss.backward()

        ae_optimizer.step()
        cl_optimizer.step()

        ae_total_loss += ae_loss.item()
        cl_total_loss += cl_loss.item()
        batch_counter += 1
    return (
        ae_total_loss / batch_counter,
        cl_total_loss / batch_counter,
    )


def evaluate_dtc(
    encoder,
    decoder,
    clustering,
    ae_criterion,
    cl_criterion,
    perf_metric,
    data_loader,
    device,
):
    encoder.eval()
    decoder.eval()
    clustering.eval()

    ae_total_loss = 0
    cl_total_loss = 0
    perf_total_score = 0
    batch_counter = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device).flatten()

            latent = encoder(x)
            x_prime = decoder(latent)

            Q = clustering(latent)
            P = target_distribution(Q)
            log_Q, log_P = torch.log(Q), torch.log(P)

            ae_loss = ae_criterion(x, x_prime)
            cl_loss = cl_criterion(log_Q, log_P)

            ae_total_loss += ae_loss.item()
            cl_total_loss += cl_loss.item()

            cluster_preds = torch.argmax(Q, dim=1).flatten()

            perf_total_score += perf_metric(cluster_preds, y)
            batch_counter += 1

    return (
        ae_total_loss / batch_counter,
        cl_total_loss / batch_counter,
        perf_total_score / batch_counter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default="test")
    args = parser.parse_args()

    wandb_run = wandb.init(project="DTC", name=args.run_name)

    artifact_X = wandb_run.use_artifact(
        "tristanbester1/DTC/polynomial_dataset_X:v0", type="dataset",
    )
    artifact_Y = wandb_run.use_artifact(
        "tristanbester1/DTC/polynomial_dataset_Y:v0", type="dataset",
    )

    X_path = artifact_X.download()
    Y_path = artifact_Y.download()

    dataset = PolynomialDataset(X_path=X_path, Y_path=Y_path,)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device(args.device)

    encoder = Encoder(**ENCODER_CONFIG)
    decoder = Decoder(**DECODER_CONFIG)
    autoencoder = nn.Sequential(encoder, decoder)
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    stop = EarlyStopping(patience=1)
    val_loss = None

    print("Autoencoder pretraining started.")

    while not stop(val_loss):
        train_loss = train_autoencoder_one_epoch(
            autoencoder, ae_optimizer, ae_criterion, train_loader, device
        )
        val_loss = evaluate_autoencoder(autoencoder, ae_criterion, test_loader, device)
        wandb.log(
            {"AE_pretrain_train_loss": train_loss, "AE_pretrain_val_loss": val_loss,}
        )

    print("Autoencoder pretrainined completed.")

    print("Initialising cluster centroids.")

    centroids = init_centroids(
        autoencoder[0], train_loader, device, euclidean_distance, 3
    )

    print("Cluster centroids initialised.")

    clustering_layer = ClusteringLayer(
        centroids=centroids, metric=euclidean_distance, alpha=1
    )
    cl_criterion = nn.KLDivLoss(log_target=True, reduction="batchmean")
    cl_optimizer = optim.Adam(params=clustering_layer.parameters(), lr=0.001)

    ae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        ae_optimizer, factor=0.5, patience=2, threshold=0.001, verbose=True,
    )
    cl_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        cl_optimizer, factor=0.5, patience=2, threshold=0.001, verbose=True,
    )

    encoder_artifact = wandb.Artifact(f"encoder-{wandb.run.id}", type="model")
    decoder_artifact = wandb.Artifact(f"decoder-{wandb.run.id}", type="model")
    cluster_artifact = wandb.Artifact(f"CL-{wandb.run.id}", type="model")

    lowest_cl_val_loss = float("inf")

    print("DTC training started.")

    for i in tqdm(range(1000)):
        ae_train_loss, cl_train_loss = train_dtc_one_epoch(
            encoder=encoder,
            decoder=decoder,
            clustering=clustering_layer,
            ae_optimizer=ae_optimizer,
            cl_optimizer=cl_optimizer,
            ae_criterion=ae_criterion,
            cl_criterion=cl_criterion,
            data_loader=train_loader,
            device=device,
        )

        ae_val_loss, cl_val_loss, val_acc = evaluate_dtc(
            encoder=encoder,
            decoder=decoder,
            clustering=clustering_layer,
            ae_criterion=ae_criterion,
            cl_criterion=cl_criterion,
            perf_metric=accuracy,
            data_loader=test_loader,
            device=device,
        )

        ae_scheduler.step(ae_val_loss)
        cl_scheduler.step(cl_val_loss)

        wandb.log(
            {
                "DTC_AE_MSE_train_loss": ae_train_loss,
                "DTC_CL_KLDiv_train_loss": cl_train_loss,
                "DTC_AE_MSE_val_loss": ae_val_loss,
                "DTC_ACL_KLDiv_val_loss": ae_val_loss,
                "DTC_cluster_acc_val_loss": val_acc,
                "DTC_AE_lr": get_lr(ae_optimizer),
                "DTC_CL_lr": get_lr(cl_optimizer),
            }
        )

        if cl_val_loss < lowest_cl_val_loss:
            torch.save(encoder.state_dict(), f"./encoder.pt")
            torch.save(decoder.state_dict(), f"./decoder.pt")
            torch.save(clustering_layer.state_dict(), f"./CL.pt")
            lowest_cl_val_loss = cl_val_loss

    encoder_artifact.add_file("./encoder.pt", "model_encoder.pt")
    decoder_artifact.add_file("./decoder.pt", "model_decoder.pt")
    cluster_artifact.add_file("./CL.pt", "model_cluster.pt")

    wandb.log_artifact(encoder_artifact, aliases=["latest", "best"])
    wandb.log_artifact(decoder_artifact, aliases=["latest", "best"])
    wandb.log_artifact(cluster_artifact, aliases=["latest", "best"])

    print("DTC training complete.")

