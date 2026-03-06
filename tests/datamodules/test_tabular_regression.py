# Copyright (c) 2023 lightning-uq-box. All rights reserved.
# Licensed under the Apache License 2.0.

"""Tests for tabular regression datamodule."""

from pathlib import Path

import pandas as pd
import pytest

from lightning_uq_box.datamodules import TabularRegressionDataModule


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    """Create a small in-test sample dataset with 5 inputs and 1 output."""
    data = {
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
        "x3": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "x4": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        "x5": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        "y": [3.0, 3.4, 3.6, 3.8, 4.0, 4.2, 4.6, 4.8, 5.0, 5.2],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_csv(tmp_path: Path, sample_dataframe: pd.DataFrame) -> Path:
    path = tmp_path / "sample_soft_sensor.csv"
    sample_dataframe.to_csv(path, index=False)
    return path


def test_tabular_regression_datamodule_setup(sample_csv: Path) -> None:
    dm = TabularRegressionDataModule(
        data_path=sample_csv,
        n_input_features=5,
        target_column=-1,
        val_size=0.2,
        test_size=0.2,
        batch_size=4,
        random_state=0,
    )

    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    assert set(batch.keys()) == {"input", "target"}
    assert batch["input"].shape[-1] == 5
    assert batch["target"].shape[-1] == 1


def test_invalid_split_ratio_raises(sample_csv: Path) -> None:
    dm = TabularRegressionDataModule(
        data_path=sample_csv,
        val_size=0.5,
        test_size=0.5,
    )

    with pytest.raises(ValueError, match=r"test_size \+ val_size must be < 1.0"):
        dm.setup()


def test_tabular_regression_datamodule_from_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    dm = TabularRegressionDataModule(
        dataframe=sample_dataframe,
        n_input_features=5,
        target_column=-1,
        val_size=0.2,
        test_size=0.2,
        batch_size=4,
        random_state=0,
    )

    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    assert batch["input"].shape[-1] == 5
    assert batch["target"].shape[-1] == 1
