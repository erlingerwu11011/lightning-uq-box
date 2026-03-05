# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Datamodule for tabular regression data sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .utils import collate_fn_tensordataset


class TabularRegressionDataModule(LightningDataModule):
    """Datamodule for tabular regression with train/val/test split and scaling.

    By default it uses the first ``n_input_features`` columns as model inputs and the
    last column as target.
    """

    def __init__(
        self,
        data_path: str | Path,
        n_input_features: int = 5,
        target_column: str | int = -1,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        batch_size: int = 64,
        scale_inputs: bool = True,
        scale_target: bool = True,
        sheet_name: str | int = 0,
    ) -> None:
        """Initialize datamodule from CSV or Excel data source.

        Args:
            data_path: path to source data file (.xlsx/.xls/.csv).
            n_input_features: number of leading columns used as input features.
            target_column: target column selector, either column name or index.
            test_size: test split ratio out of full data.
            val_size: validation split ratio out of full data.
            random_state: random seed for splitting.
            batch_size: dataloader batch size.
            scale_inputs: whether to standardize input features.
            scale_target: whether to standardize target values.
            sheet_name: Excel sheet selector when reading Excel files.
        """
        super().__init__()

        self.data_path = Path(data_path)
        self.n_input_features = n_input_features
        self.target_column = target_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.scale_inputs = scale_inputs
        self.scale_target = scale_target
        self.sheet_name = sheet_name

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def setup(self, stage: str | None = None) -> None:
        """Prepare train/val/test datasets."""
        del stage

        df = self._load_dataframe()

        if isinstance(self.target_column, str):
            y = df[self.target_column].to_numpy().reshape(-1, 1)
        else:
            y = df.iloc[:, self.target_column].to_numpy().reshape(-1, 1)

        X = df.iloc[:, : self.n_input_features].to_numpy()

        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=self.test_size + self.val_size,
            random_state=self.random_state,
        )

        holdout_test_ratio = self.test_size / (self.test_size + self.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout,
            y_holdout,
            test_size=holdout_test_ratio,
            random_state=self.random_state,
        )

        if self.scale_inputs:
            X_train = self.x_scaler.fit_transform(X_train)
            X_val = self.x_scaler.transform(X_val)
            X_test = self.x_scaler.transform(X_test)

        if self.scale_target:
            y_train = self.y_scaler.fit_transform(y_train)
            y_val = self.y_scaler.transform(y_val)
            y_test = self.y_scaler.transform(y_test)

        self.train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        self.val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        self.test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tensordataset,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def _load_dataframe(self) -> pd.DataFrame:
        suffix = self.data_path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        if suffix == ".csv":
            return pd.read_csv(self.data_path)

        raise ValueError(
            "Unsupported file extension. Use one of: .xlsx, .xls, .csv"
        )
