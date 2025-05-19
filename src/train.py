import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import cudaq

from typing import Any, Dict
from tqdm import tqdm

from torchquantum.models import QNN, HybridQNN

from src.classic_models import ParamMatchedFCN
from src.train_functions import train_step, val_step, test_step
from src.utils import Accuracy, get_unique_filename, set_seed, count_parameters
from src.data import load_wine_dataset, load_iris_dataset


# Global configuration dictionaries for data, training, and model parameters.
IN_FEATURES = 4
OUT_FEATURES = 3

TRAINING_PARAMS: Dict[str, Any] = {
    "batch_size": 8,
    "num_workers": 3,
    "learning_rate": 0.1,
    "epochs": 20,
    "seed": 42,
}

QNN_PARAMS: Dict[str, Any] = {
    "in_features": IN_FEATURES,
    "out_features": OUT_FEATURES,
    "num_layers": 3,
    "shots": 1028,
    "feature_map": "z",
    "var_form": "efficientsu2",
    "reupload": False,
}

FCN_PARAMS: Dict[str, int] = {
    "in_features": IN_FEATURES,
    "out_features": OUT_FEATURES,
}

HYBRID_PARAMS: Dict[str, Any] = {
    "in_features": IN_FEATURES,
    "out_features": OUT_FEATURES,
    "num_layers": 1,
    "shots": 1028,
    "feature_map": "z",
    "var_form": "efficientsu2",
    "reupload": True,
}


def main() -> None:
    r"""Main training and evaluation loop.

    Loads the wine dataset, preprocesses the data, splits it into train, validation,
    and test sets, and trains both a quantum neural network (QNN) and a classical
    fully connected network (FCN). Logs training and validation metrics to TensorBoard,
    and saves the QNN model state after training.
    """
    # Set random seeds for reproducibility.
    set_seed(TRAINING_PARAMS["seed"])

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set target for cudaq based on device type.
    if device.type == "cuda":
        cudaq.set_target("nvidia")
    else:
        cudaq.set_target("qpp-cpu")

    train_loader, val_loader, test_loader = load_iris_dataset(
        batch_size=TRAINING_PARAMS["batch_size"],
        num_workers=TRAINING_PARAMS["num_workers"],
        seed=TRAINING_PARAMS["seed"],
    )

    # Initialize models.
    model_qnn: nn.Module = QNN(
        in_features=QNN_PARAMS["in_features"],
        out_features=QNN_PARAMS["out_features"],
        num_layers=QNN_PARAMS["num_layers"],
        backend=None,
        shots=QNN_PARAMS["shots"],
        feature_map=QNN_PARAMS["feature_map"],
        var_form=QNN_PARAMS["var_form"],
        reupload=QNN_PARAMS["reupload"],
    )
    d: int = count_parameters(model_qnn)
    print(f"Quantum model parameter count:", d)

    model_fcn: nn.Module = ParamMatchedFCN(
        in_features=FCN_PARAMS["in_features"],
        out_features=FCN_PARAMS["out_features"],
        d=d,
    )
    real_d: int = count_parameters(model_fcn)
    print(f"Classical model parameter count:", real_d)

    model_hybrid: nn.Module = HybridQNN(
        in_features=HYBRID_PARAMS["in_features"],
        out_features=HYBRID_PARAMS["out_features"],
        num_layers=HYBRID_PARAMS["num_layers"],
        backend=None,
        shots=HYBRID_PARAMS["shots"],
        feature_map=HYBRID_PARAMS["feature_map"],
        var_form=HYBRID_PARAMS["var_form"],
        reupload=HYBRID_PARAMS["reupload"],
    )
    d: int = count_parameters(model_hybrid)
    print(f"Hybrid model parameter count:", d)

    model_qnn.to(device)
    model_fcn.to(device)
    model_hybrid.to(device)

    # Define loss function and optimizers.
    criterion = nn.CrossEntropyLoss()
    optimizer_qnn = optim.Adam(
        model_qnn.parameters(), lr=TRAINING_PARAMS["learning_rate"], weight_decay=0.01
    )
    optimizer_fcn = optim.Adam(
        model_fcn.parameters(), lr=TRAINING_PARAMS["learning_rate"], weight_decay=0.01
    )
    optimizer_hybrid = optim.Adam(
        model_hybrid.parameters(),
        lr=TRAINING_PARAMS["learning_rate"],
        weight_decay=0.01,
    )

    # Create TensorBoard writers with unique log directories.
    writer_qnn = SummaryWriter(get_unique_filename("runs/quantum_neural_network"))
    writer_fcn = SummaryWriter(get_unique_filename("runs/multilayer_perceptron"))
    # writer_hybrid = SummaryWriter(
    #     get_unique_filename("runs/hybrid_quantum_neural_network")
    # )
    print(
        "TensorBoard logs saved to: runs/quantum_neural_network, runs/multilayer_perceptron "
        "and runs/hybrid_quantum_neural_network"
    )
    print("Run 'tensorboard --logdir=runs' to view the logs.")

    # Initialize accuracy trackers.
    acc_train_qnn = Accuracy()
    acc_val_qnn = Accuracy()
    acc_test_qnn = Accuracy()
    acc_train_fcn = Accuracy()
    acc_val_fcn = Accuracy()
    acc_test_fcn = Accuracy()
    acc_train_hybrid = Accuracy()
    acc_val_hybrid = Accuracy()
    acc_test_hybrid = Accuracy()

    # Initial evaluation on the test set.
    test_step(model_qnn, test_loader, criterion, acc_test_qnn, device)
    test_step(model_fcn, test_loader, criterion, acc_test_fcn, device)
    test_step(model_hybrid, test_loader, criterion, acc_test_hybrid, device)
    print("Initial test evaluation logged to TensorBoard.")

    # # Training loop.
    for epoch in tqdm(range(1, TRAINING_PARAMS["epochs"] + 1)):
        acc_train_qnn.reset()
        train_step(
            model_qnn,
            train_loader,
            criterion,
            optimizer_qnn,
            acc_train_qnn,
            writer_qnn,
            epoch,
            device,
        )
        acc_val_qnn.reset()
        val_step(
            model_qnn, val_loader, criterion, acc_val_qnn, writer_qnn, epoch, device
        )

        acc_train_fcn.reset()
        train_step(
            model_fcn,
            train_loader,
            criterion,
            optimizer_fcn,
            acc_train_fcn,
            writer_fcn,
            epoch,
            device,
        )
        acc_val_fcn.reset()
        val_step(
            model_fcn, val_loader, criterion, acc_val_fcn, writer_fcn, epoch, device
        )

    #     acc_train_hybrid.reset()
    #     train_step(
    #         model_hybrid,
    #         train_loader,
    #         criterion,
    #         optimizer_hybrid,
    #         acc_train_hybrid,
    #         writer_hybrid,
    #         epoch,
    #         device,
    #     )
    #     acc_val_hybrid.reset()
    #     val_step(
    #         model_hybrid,
    #         val_loader,
    #         criterion,
    #         acc_val_hybrid,
    #         writer_hybrid,
    #         epoch,
    #         device,
    #     )

    # Final test evaluation.
    acc_test_qnn.reset()
    test_step(model_qnn, test_loader, criterion, acc_test_qnn, device)
    acc_test_fcn.reset()
    test_step(model_fcn, test_loader, criterion, acc_test_fcn, device)
    acc_test_hybrid.reset()
    # test_step(model_hybrid, test_loader, criterion, acc_test_hybrid, device)
    print(f"Final test accuracy (QNN): {acc_test_qnn.compute():.4f}")
    print(f"Final test accuracy (MLP): {acc_test_fcn.compute():.4f}")
    # print(f"Final test accuracy (Hybrid): {acc_test_hybrid.compute():.4f}")

    # Save the quantum model state.
    torch.save(model_qnn.state_dict(), "models/qnn_state_dict.pth")
    torch.save(model_fcn.state_dict(), "models/mlp_state_dict.pth")
    # torch.save(model_hybrid.state_dict(), "models/hybrid_state_dict.pth")


if __name__ == "__main__":
    main()
