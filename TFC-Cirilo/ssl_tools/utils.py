from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml


def read_dataset(
    dataset_path: Path,
    feature_columns_prefix: List[str] = (
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ),
    label_column: str = "standard activity code",
) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the dataset from the specified path and returns it as a pandas DataFrame.

    Args:
        dataset_path (Path): The path to the dataset.
        feature_columns_prefix (List[str], optional): The prefix of the feature columns. Defaults to ("accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z").
        label_column (str, optional): The label column. Defaults to "label".

    Returns:
        Tuple[np.ndarray, np.ndarray]: The features and labels.
    """
    dataset_path = Path(dataset_path)

    # Read the dataset
    df = pd.read_csv(dataset_path)

    feature_cols = [
        c
        for prefix in feature_columns_prefix
        for c in df.columns
        if c.startswith(prefix)
    ]

    X = df[feature_cols].values
    y = df[label_column].values
    return X, y


def read_train_val_test(
    dataset_path: Path,
    train_file: str = "train.csv",
    validation_file: str = "validation.csv",
    test_file: str = "test.csv",
    feature_columns_prefix: List[str] = (
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ),
    label_column: str = "standard activity code",
):
    dataset_path = Path(dataset_path)
    datasets = dict()
    for name, f in [
        ("train", train_file),
        ("validation", validation_file),
        ("test", test_file),
    ]:
        if f is None:
            datasets[name] = [None, None]
        else:
            X, y = read_dataset(dataset_path / f, feature_columns_prefix, label_column)
            datasets[name] = (X.reshape(-1, 6, 60), y)
    return datasets


def load_yaml(path: Path):
    path = Path(path)
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_data(dataset_name: str, dataset_location_path: str = "dataset_locations.yaml"):
    data_paths = load_yaml(dataset_location_path)
    return read_train_val_test(data_paths[dataset_name])


def plot_sample_sensors(
    data: np.ndarray,
    labels: List[str] = (
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ),
    title: str = "Sample sensors",
    x_axis_title: str = "Time",
    y_axis_title: str = "Value",
):
    # using pliotly
    fig = go.Figure()
    for i in range(data.shape[0]):
        fig.add_trace(go.Scatter(y=data[i, :], mode="lines", name=labels[i]))
    fig.update_layout(title=title, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    return fig