import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from typing import Dict, Tuple

__all__ = ["load_wine_dataset", "load_iris_dataset"]

# Global configuration dictionaries for data
DATA_PARAMS: Dict[str, float] = {
    "test_size": 0.25,
    "validation_split": 0.5,
}


def load_wine_dataset(
    batch_size: int, num_workers: int, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load and preprocess wine dataset.
    data = load_wine()
    X: torch.Tensor = preprocess_data(data.data)
    y: torch.Tensor = torch.tensor(data.target, dtype=torch.long)

    # Split data into training, validation, and test sets.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=DATA_PARAMS["test_size"], random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=DATA_PARAMS["validation_split"],
        random_state=seed,
    )

    train_dataset: TensorDataset = TensorDataset(X_train, y_train)
    val_dataset: TensorDataset = TensorDataset(X_val, y_val)
    test_dataset: TensorDataset = TensorDataset(X_test, y_test)

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def load_iris_dataset(
    batch_size: int, num_workers: int, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Load and preprocess iris dataset.
    data = load_iris()
    X: torch.Tensor = preprocess_data(data.data)
    y: torch.Tensor = torch.tensor(data.target, dtype=torch.long)

    # Split data into training, validation, and test sets.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=DATA_PARAMS["test_size"], random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=DATA_PARAMS["validation_split"],
        random_state=seed,
    )

    train_dataset: TensorDataset = TensorDataset(X_train, y_train)
    val_dataset: TensorDataset = TensorDataset(X_val, y_val)
    test_dataset: TensorDataset = TensorDataset(X_test, y_test)

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader: DataLoader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def preprocess_data(X: np.ndarray) -> torch.Tensor:
    r"""Normalize data to the range [-1, 1].

    Args:
        X (np.ndarray): Input features.

    Returns:
        torch.Tensor: Normalized features as a float32 tensor.
    """
    X_tensor: torch.Tensor = torch.tensor(X, dtype=torch.float32)
    # Compute min and max along each feature dimension.
    min_vals: torch.Tensor = X_tensor.min(dim=0, keepdim=True)[0]
    max_vals: torch.Tensor = X_tensor.max(dim=0, keepdim=True)[0]
    X_normalized: torch.Tensor = 2 * (X_tensor - min_vals) / (max_vals - min_vals) - 1
    return X_normalized
