import os
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv


def get_linear(X):
    return 2 * X + np.random.randn(len(X)) * 0.05


def get_quadratic(X):
    return X**2 + np.random.randn(len(X)) * 0.05


def get_cubic(X):
    return X**3 + np.random.randn(len(X)) * 0.05


if __name__ == "__main__":
    load_dotenv()

    data_dir = os.environ.get("DATA_BASE_PATH")
    data_path = Path(data_dir) / "polynomial" / "polynomial.csv"

    # Function inputs
    X = np.linspace(-1, 1, 100)

    # Calculate function outputs
    linear_series = [get_linear(X) for _ in range(100)]
    quadratic_series = [get_quadratic(X) for _ in range(100)]
    cubic_series = [get_cubic(X) for _ in range(100)]

    # Aggregate data
    df = pd.DataFrame(data=(linear_series + quadratic_series + cubic_series), columns=X)
    # Label based on degree of polynomial
    df["label"] = [0] * 100 + [1] * 100 + [2] * 100

    # Serialise
    df.to_csv(data_path, index=False)

    # Push to remote
    run = wandb.init(project="DTC", name="Generate polynomial dataset")
    artifact = wandb.Artifact("polynomial_dataset", type="dataset")
    artifact.add_file(data_path)
    run.log_artifact(artifact)
