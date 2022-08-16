import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from datasets import PolynomialDataset
from metrics.performance import accuracy
from metrics.similarity import complexity_invariant_similarity, euclidean_distance
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
#     "cnn_kernel": 10,
#     "cnn_stride": 3,
#     "mp_kernel": 10,
#     "mp_stride": 3,
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
    optimizer,
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

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device).flatten()

        optimizer.zero_grad()

        latent = encoder(x)
        x_prime = decoder(latent)

        Q = clustering(latent)
        P = target_distribution(Q)
        log_Q, log_P = torch.log(Q), torch.log(P)

        ae_loss = ae_criterion(x, x_prime)
        cl_loss = cl_criterion(log_Q, log_P)

        dtc_loss = ae_loss + cl_loss

        dtc_loss.backward()

        optimizer.step()

        ae_total_loss += ae_loss.item()
        cl_total_loss += cl_loss.item()

        batch_counter += 1
    return (
        ae_total_loss / batch_counter,
        cl_total_loss / batch_counter,
    )


def evaluate_dtc(
    encoder, decoder, clustering, ae_criterion, cl_criterion, data_loader, device,
):
    encoder.eval()
    decoder.eval()
    clustering.eval()

    ae_total_loss = 0
    cl_total_loss = 0
    batch_counter = 0

    labels = []
    predictions = []
    Qs = []
    Ps = []

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

            predictions.append(cluster_preds)
            labels.append(y)
            Qs.append(Q)
            Ps.append(P)

            batch_counter += 1

    return (
        ae_total_loss / batch_counter,
        cl_total_loss / batch_counter,
        predictions,
        labels,
        Qs,
        Ps,
    )


def permute(preds, Qs, Ps, labels, option=None):
    acc_orig = (preds == labels).to(torch.float).mean().item()

    preds = (preds + 1) % 3
    acc_one = (preds == labels).to(torch.float).mean().item()

    preds = (preds + 1) % 3
    acc_two = (preds == labels).to(torch.float).mean().item()

    if option is None:
        if np.argmax([acc_orig, acc_one, acc_two]) == 0:
            # print("##### 1 #####")
            return acc_orig, Qs, Ps, 0
        elif np.argmax([acc_orig, acc_one, acc_two]) == 1:
            # print("##### 2 #####")
            Qs = torch.cat([Qs[:, 0], Qs[:, 2], Qs[:, 1],]).reshape(3, Qs.shape[0]).T
            Ps = torch.cat([Ps[:, 0], Ps[:, 2], Ps[:, 1],]).reshape(3, Qs.shape[0]).T
            return acc_one, Qs, Ps, 1
        else:
            # print("##### 3 #####")
            Qs = torch.cat([Qs[:, 2], Qs[:, 1], Qs[:, 0],]).reshape(3, Qs.shape[0]).T
            Ps = torch.cat([Ps[:, 2], Ps[:, 1], Ps[:, 0],]).reshape(3, Qs.shape[0]).T
            return acc_two, Ps, Qs, 2
    else:
        if option == 0:
            return acc_orig, Qs, Ps, 0
        elif option == 1:
            Qs = torch.cat([Qs[:, 0], Qs[:, 2], Qs[:, 1],]).reshape(3, Qs.shape[0]).T
            Ps = torch.cat([Ps[:, 0], Ps[:, 2], Ps[:, 1],]).reshape(3, Qs.shape[0]).T
            return acc_one, Qs, Ps, 1
        else:
            Qs = torch.cat([Qs[:, 2], Qs[:, 1], Qs[:, 0],]).reshape(3, Qs.shape[0]).T
            Ps = torch.cat([Ps[:, 2], Ps[:, 1], Ps[:, 0],]).reshape(3, Qs.shape[0]).T
            return acc_two, Ps, Qs, 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run_name", type=str, default="test")
    args = parser.parse_args()

    wandb_run = wandb.init(project="DTC", name=args.run_name)

    # artifact_X = wandb_run.use_artifact(
    #     "tristanbester1/DTC/polynomial_dataset_X:v0", type="dataset",
    # )
    # artifact_Y = wandb_run.use_artifact(
    #     "tristanbester1/DTC/polynomial_dataset_Y:v0", type="dataset",
    # )

    # X_path = artifact_X.download()
    # Y_path = artifact_Y.download()

    # dataset = PolynomialDataset(X_path=X_path, Y_path=Y_path,)
    dataset = PolynomialDataset(
        X_path="/Users/tristan/Documents/CS/Research/DTC/artifacts/polynomial_dataset_X:v0",
        Y_path="/Users/tristan/Documents/CS/Research/DTC/artifacts/polynomial_dataset_Y:v0",
    )

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=dataset, batch_size=300)
    test_loader = DataLoader(dataset=dataset, batch_size=300)

    device = torch.device(args.device)

    encoder = Encoder(**ENCODER_CONFIG)
    decoder = Decoder(**DECODER_CONFIG)
    autoencoder = nn.Sequential(encoder, decoder)
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    stop = EarlyStopping(patience=0)
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
        print(val_loss)

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
    dtc_optimizer = optim.Adam(params=clustering_layer.parameters(), lr=0.01)

    dtc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        dtc_optimizer, factor=0.5, patience=50, threshold=0.0001, verbose=True,
    )

    encoder_artifact = wandb.Artifact(f"encoder-{wandb.run.id}", type="model")
    decoder_artifact = wandb.Artifact(f"decoder-{wandb.run.id}", type="model")
    cluster_artifact = wandb.Artifact(f"CL-{wandb.run.id}", type="model")

    lowest_loss = float("inf")

    print("DTC training started.")
    option = None

    all_centroids = []
    all_centroids.append(clustering_layer.centroids.detach().numpy().copy())

    all_ae_preds = []

    for i in range(1000):
        ae_train_loss, cl_train_loss = train_dtc_one_epoch(
            encoder=encoder,
            decoder=decoder,
            clustering=clustering_layer,
            optimizer=dtc_optimizer,
            ae_criterion=ae_criterion,
            cl_criterion=cl_criterion,
            data_loader=train_loader,
            device=device,
        )

        ae_val_loss, cl_val_loss, preds, labels, Qs, Ps = evaluate_dtc(
            encoder=encoder,
            decoder=decoder,
            clustering=clustering_layer,
            ae_criterion=ae_criterion,
            cl_criterion=cl_criterion,
            data_loader=test_loader,
            device=device,
        )

        dtc_scheduler.step(ae_val_loss + cl_val_loss)

        preds = torch.cat(preds)
        Qs = torch.cat(Qs)
        Ps = torch.cat(Ps)
        labels = torch.cat(labels).to(torch.long)

        preds, Qs, Ps, option = permute(preds, Qs, Ps, labels, None)

        preds_Q = torch.argmax(Qs, dim=1).flatten()
        preds_P = torch.argmax(Ps, dim=1).flatten()

        acc_Q = (preds_Q == labels).to(torch.float).mean().item()
        acc_P = (preds_P == labels).to(torch.float).mean().item()

        print(acc_P)

        l_Q = F.cross_entropy(Qs, labels).item()
        l_P = F.cross_entropy(Ps, labels).item()

        maxes_Q = Qs.max(dim=1).values.mean().item()
        maxes_P = Ps.max(dim=1).values.mean().item()

        wandb.log(
            {
                "AE_loss": ae_val_loss,
                "CL_loss": cl_val_loss,
                "DTC_loss": ae_val_loss + cl_val_loss,
                "ACC_Q": acc_Q,
                "ACC_P": acc_P,
                "Cross_Entropy_Q": l_Q,
                "Cross_Entropy_P": l_P,
                "Ave_Max_Proba_Q": maxes_Q,
                "Ave_Max_Proba_P": maxes_P,
            }
        )

        if i % 100 == 0:
            all_centroids.append(clustering_layer.centroids.detach().numpy().copy())

            for x, y in train_loader:
                l = encoder(x)
                all_ae_preds.append((l, y))

        if ae_val_loss + cl_val_loss < lowest_loss:
            print(f"Saving - {i}")
            lowest_loss = ae_val_loss + cl_val_loss
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

    with open("centroids.pkl", "wb") as f:
        pickle.dump(all_centroids, f)

    with open("preds.pkl", "wb") as f:
        pickle.dump(all_ae_preds, f)

    centroids_artifact = wandb.Artifact(f"centroids-{wandb.run.id}", type="audit-data")
    ae_latent_artifact = wandb.Artifact(f"latent-{wandb.run.id}", type="audit-data")

    centroids_artifact.add_file("./centroids.pkl", "centroids.pkl")
    ae_latent_artifact.add_file("./preds.pkl", "latent.pkl")

    wandb.log_artifact(centroids_artifact)
    wandb.log_artifact(ae_latent_artifact)
