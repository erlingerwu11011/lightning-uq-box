from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import Trainer
from torch import nn

from lightning_uq_box.datamodules import TabularRegressionDataModule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import (
    DeepEnsembleRegression,
    MCDropoutRegression,
    MVERegression,
    NLL,
    QuantileRegression,
)


def build_datamodule(df: pd.DataFrame, batch_size: int) -> TabularRegressionDataModule:
    """Build tabular datamodule for 5 inputs + last column as target."""
    return TabularRegressionDataModule(
        dataframe=df,
        n_input_features=5,
        target_column=-1,
        batch_size=batch_size,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )


def train_and_predict_mc_dropout(
    dm: TabularRegressionDataModule, max_epochs: int
) -> list[dict[str, torch.Tensor]]:
    model = MLP(n_inputs=5, n_hidden=[64, 64], n_outputs=1, dropout_p=0.1)
    uq_model = MCDropoutRegression(
        model=model,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        loss_fn=NLL(),
        num_mc_samples=30,
        burnin_epochs=20,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(uq_model, datamodule=dm)
    return trainer.predict(uq_model, datamodule=dm)


def train_and_predict_quantile(
    dm: TabularRegressionDataModule, max_epochs: int
) -> list[dict[str, torch.Tensor]]:
    quantiles = [0.1, 0.5, 0.9]
    model = MLP(
        n_inputs=5,
        n_hidden=[64, 64],
        n_outputs=len(quantiles),
        activation_fn=nn.ReLU(),
    )
    uq_model = QuantileRegression(
        model=model,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        quantiles=quantiles,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(uq_model, datamodule=dm)
    return trainer.predict(uq_model, datamodule=dm)


def train_and_predict_deep_ensemble(
    dm: TabularRegressionDataModule,
    max_epochs: int,
    n_ensembles: int,
) -> list[dict[str, torch.Tensor]]:
    trained_members: list[dict[str, str | MVERegression]] = []

    for i in range(n_ensembles):
        base_model = MLP(n_inputs=5, n_hidden=[64, 64], n_outputs=2, dropout_p=0.0)
        member = MVERegression(
            model=base_model,
            optimizer=partial(torch.optim.Adam, lr=1e-3),
            burnin_epochs=20,
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(member, datamodule=dm)

        ckpt_path = Path(f"deep_ensemble_member_{i}.ckpt")
        trainer.save_checkpoint(str(ckpt_path))
        trained_members.append({"base_model": member, "ckpt_path": str(ckpt_path)})

    ensemble = DeepEnsembleRegression(trained_members)
    return Trainer(accelerator="auto", devices=1, logger=False).predict(
        ensemble, datamodule=dm
    )


def summarize_predictions(
    pred_batches: list[dict[str, torch.Tensor]], dm: TabularRegressionDataModule
) -> None:
    pred_mean = np.concatenate([batch["pred"].detach().cpu().numpy() for batch in pred_batches])
    pred_uct = np.concatenate(
        [batch["pred_uct"].detach().cpu().numpy() for batch in pred_batches]
    )

    pred_mean_real = dm.y_scaler.inverse_transform(pred_mean)

    print("pred_mean_real shape:", pred_mean_real.shape)
    print("pred_uct shape:", pred_uct.shape)
    print("pred_mean_real[:5]:\n", pred_mean_real[:5])
    print("pred_uct[:5]:\n", pred_uct[:5])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to your real Excel file, e.g. C:/Users/13377/Desktop/test1.xlsx",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mc_dropout",
        choices=["mc_dropout", "quantile", "deep_ensemble"],
        help="Select UQ method",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-ensembles", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Excel file not found: {data_path}")

    # Always use the user-provided real file path (no synthetic fallback)
    df = pd.read_excel(data_path)
    dm = build_datamodule(df=df, batch_size=args.batch_size)

    if args.method == "mc_dropout":
        pred_batches = train_and_predict_mc_dropout(dm, max_epochs=args.max_epochs)
    elif args.method == "quantile":
        pred_batches = train_and_predict_quantile(dm, max_epochs=args.max_epochs)
    else:
        pred_batches = train_and_predict_deep_ensemble(
            dm,
            max_epochs=args.max_epochs,
            n_ensembles=args.n_ensembles,
        )

    summarize_predictions(pred_batches, dm)


if __name__ == "__main__":
    main()
