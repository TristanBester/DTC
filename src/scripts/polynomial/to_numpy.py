import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    run = wandb.init(project="DTC", name="Convert X and Y to NumPy")

    # Download polynomial dataset as run input
    artifact = run.use_artifact("polynomial_dataset:v0")
    poly_data_dir = artifact.download()
    poly_data_path = poly_data_dir + "/polynomial.csv"

    # Setup paths
    data_dir = os.environ.get("DATA_BASE_PATH")
    X_path = Path(data_dir) / "X.npy"
    Y_path = Path(data_dir) / "Y.npy"

    df = pd.read_csv(poly_data_path)

    # Split features and target
    X = df.drop("label", axis=1).values
    Y = df[["label"]].values

    # Serialise
    np.save(X_path, X)
    np.save(Y_path, Y)

    # Push to remote
    artifact_X = wandb.Artifact("polynomial_dataset_X", type="dataset")
    artifact_Y = wandb.Artifact("polynomial_dataset_Y", type="dataset")
    artifact_X.add_file(X_path)
    artifact_Y.add_file(Y_path)
    run.log_artifact(artifact_X)
    run.log_artifact(artifact_Y)
