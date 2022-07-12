import logging
import warnings
import argparse
from argparse import Namespace
from functools import partial

import torch
import torchvision
import numpy as np
import pytorch_lightning as pl
import ConfigSpace.hyperparameters as CSH
from torch import Generator
from torchvision import transforms
from sklearn.gaussian_process.kernels import Matern
from matplotlib.backends.backend_pdf import PdfPages

from bayesopt.bayesopt import BayesOpt
from bayesopt.resnet import ResNetModel, ResNetFashionMnistModule


def optimize(
    val_size: float = 0.3,
    max_epochs: int = 10,
    batch_size: int = 128,
    patience: int = 0,
    lr: float = 1e-3,
    generator: Generator = None,
) -> float:
    """Trains and evaluates a ResNet on the FashionMNIST classification task.

    Splits the training data into training and validation sets. Trains the
    ResNet until the early stopping criterion is met or the maximum number of
    epochs is reached. Reports the val. performance in terms of its F1 score.

    Parameters
    ----------
    val_size : float
        Fraction of the number of training samples used as validation data.
    max_epochs : int
        Number of epochs the neural network is trained at most.
    batch_size : int
        Batch size of training & validation data loaders.
    patience : int
        Maximum number of validation checks with no improvement before the
        training is stopped early.
    lr : float
        Learning rate of the SGD optimizer.
    generator: Generator
        Random number generator used to train / test split.

    Returns
    -------
    float
        Final F1 score of the ResNet on the validation data.
    """
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    transform = transforms.ToTensor()
    data = torchvision.datasets.FashionMNIST(
        args.store_dir, download=True, train=True, transform=transform
    )

    train = int(np.floor((1.0 - val_size) * len(data)))
    val = int(np.ceil(val_size * len(data)))
    train, val = torch.utils.data.random_split(
        data, (train, val), generator=generator
    )

    train = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val = torch.utils.data.DataLoader(val, batch_size=batch_size)

    model = ResNetModel(num_classes=10)
    module = ResNetFashionMnistModule(model, num_classes=10, lr=lr)

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_f1_score", mode="max", patience=patience
    )

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        max_epochs=max_epochs,
        callbacks=[early_stopping],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=None,
    )
    trainer.validate(module, val, verbose=False)
    trainer.fit(module, train, val)

    f1_score = trainer.logged_metrics["val_f1_score"].item()
    return f1_score


def main(args: Namespace) -> None:
    """Optimize the learning rate of a ResNet on FasionMNIST.

    Uses Bayesian Optimization to find a learning rate that maximizes the val.
    F1 score of a ResNet architecture. Uses a single train / val. split for
    each attempt.

    Parameters
    ----------
    args : Namespace
        Specifies arguments of the optimization script.
    """
    # Define search space of LR over log scale.
    lr = CSH.UniformFloatHyperparameter(
        "lr", lower=args.lower, upper=args.upper, log=True
    )

    # Define parameters of acquisition function, minimization and surrogate.
    acquisition = {"acquisition": "expected_improvement", "xi": 0.01}
    minimizer_kwargs = {"method": "L-BFGS-B", "nrestarts": 25}
    kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
    surrogate = {"n_restarts_optimizer": 10, "kernel": kernel}

    # Random number generator used to train / test split.
    generator = Generator()
    generator.manual_seed(args.seed)

    # Define parameters of evaluation function except for learning rate.
    objective = {
        "val_size": 0.3,
        "max_epochs": args.max_epochs,
        "batch_size": 128,
        "patience": 0,
        "generator": generator,
    }
    objective = partial(optimize, **objective)

    bayesopt = BayesOpt(
        space=lr,
        objective=objective,
        acquisition=acquisition,
        minimizer_kwargs=minimizer_kwargs,
        surrogate=surrogate,
        log_freq=1,
        seed=args.seed,
    )

    # Store output of Bayesian Optimization in .pdf file.
    with PdfPages(args.output) as pdf:
        bayesopt.optimize(1)

        for _ in range(args.niters - 1):
            fig = bayesopt.render()
            pdf.savefig(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lower", type=float, default=1e-5, help="Lower bound of LR."
    )
    parser.add_argument(
        "--upper", type=float, default=1e-1, help="Upper bound of LR."
    )
    parser.add_argument(
        "--niters", type=int, default=10, help="Num. of BO iterations."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Max. Num. of training epochs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bayesopt.pdf",
        help="File that stores the BO plots.",
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default="fashion-mnist",
        help="Directory that stores the FashionMNIST data.",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=0, help="Num. of GPUs."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    main(args)
