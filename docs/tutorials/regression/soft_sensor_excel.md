# Soft Sensor Regression from Excel Data (5 Inputs, 1 Output)

下面给一个**完整可运行**的脚本：你先用

```python
import pandas as pd

df = pd.read_excel(r"C:\Users\13377\Desktop\test1.xlsx")
```

然后在同一个脚本里切换不同 UQ 方法（MC Dropout / Quantile Regression / Deep Ensemble）进行训练和预测。

> 这 3 个只是**示例**，不是只能用这 3 个。`lightning-uq-box` 里还有很多回归 UQ 方法（如 `LaplaceRegression`、`SNGPRegression`、`CARDRegression`、`SWAGRegression` 等），只是为了给你一个最容易直接跑通的起点，这里先放了最常见的三种。

## 1) 新建脚本 `run_soft_sensor_uq.py`

```python
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


def build_datamodule(df: pd.DataFrame, batch_size: int = 64) -> TabularRegressionDataModule:
    """5个输入变量 + 最后一列输出变量。"""
    return TabularRegressionDataModule(
        dataframe=df,
        n_input_features=5,
        target_column=-1,
        batch_size=batch_size,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )


def train_and_predict_mc_dropout(dm: TabularRegressionDataModule, max_epochs: int):
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
    preds = trainer.predict(uq_model, datamodule=dm)
    return preds


def train_and_predict_quantile(dm: TabularRegressionDataModule, max_epochs: int):
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
    preds = trainer.predict(uq_model, datamodule=dm)
    return preds


def train_and_predict_deep_ensemble(
    dm: TabularRegressionDataModule,
    max_epochs: int,
    n_ensembles: int,
):
    trained_members = []

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
    preds = Trainer(accelerator="auto", devices=1, logger=False).predict(
        ensemble, datamodule=dm
    )
    return preds


def summarize_predictions(pred_batches, dm: TabularRegressionDataModule):
    pred_mean = np.concatenate([batch["pred"].detach().cpu().numpy() for batch in pred_batches])
    pred_uct = np.concatenate(
        [batch["pred_uct"].detach().cpu().numpy() for batch in pred_batches]
    )

    # 恢复到原始输出量纲（默认scale_target=True）
    pred_mean_real = dm.y_scaler.inverse_transform(pred_mean)

    print("pred_mean_real shape:", pred_mean_real.shape)
    print("pred_uct shape:", pred_uct.shape)
    print("pred_mean_real[:5]:\n", pred_mean_real[:5])
    print("pred_uct[:5]:\n", pred_uct[:5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="mc_dropout",
        choices=["mc_dropout", "quantile", "deep_ensemble"],
        help="选择不确定性量化方法",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-ensembles", type=int, default=3)
    args = parser.parse_args()

    # 你给的读取方式
    df = pd.read_excel(r"C:\Users\13377\Desktop\test1.xlsx")

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
```

## 2) 一步步运行

### Step A. 安装依赖（首次）

```bash
pip install -e .
pip install pandas openpyxl
```

> 如果你已经通过 `pip install lightning-uq-box` 或 `pip install git+https://github.com/lightning-uq-box/lightning-uq-box.git` 安装过库本体，只需要确保 `pandas/openpyxl` 可用即可。

### Step B. 运行 MC Dropout

```bash
python run_soft_sensor_uq.py --method mc_dropout --max-epochs 120
```

### Step C. 运行 Quantile Regression

```bash
python run_soft_sensor_uq.py --method quantile --max-epochs 120
```

### Step D. 运行 Deep Ensemble

```bash
python run_soft_sensor_uq.py --method deep_ensemble --max-epochs 80 --n-ensembles 3
```

## 3) 关键参数说明

- `n_input_features=5`：前 5 列作为输入。
- `target_column=-1`：最后一列作为输出。
- `--method`：切换 UQ 方法（`mc_dropout` / `quantile` / `deep_ensemble`）。
- `--max-epochs`：训练轮数；你可先用小一点（如 30）检查流程，再增大。

## 4) 不止这 3 种：如何扩展到其他 UQ 方法

不是只能切换 `mc_dropout`、`quantile`、`deep_ensemble`。上面的脚本是为了“你现在就能在 `test1.xlsx` 跑起来”而给的最小完整版本。

如果要换其他方法，通常只需要两步：

1. 从 `lightning_uq_box.uq_methods` 导入目标方法；
2. 按该方法要求构建模型输出维度和 wrapper 参数，然后保持 `Trainer.fit(...)` / `Trainer.predict(...)` 不变。

例如（伪代码）：

```python
from lightning_uq_box.uq_methods import LaplaceRegression

# 先训练一个确定性回归器，再构建 LaplaceRegression
# 之后依旧是 trainer.fit(...) + trainer.predict(...)
```

建议你优先按以下顺序尝试：

- 想先快速上手：`MCDropoutRegression`
- 想直接给分位数区间：`QuantileRegression`
- 想提升稳健性：`DeepEnsembleRegression`
- 想做后验近似：`LaplaceRegression`
- 想结合 GP 风格头部：`SNGPRegression`

## 5) 常见问题

- 如果报 `ModuleNotFoundError: openpyxl`：执行 `pip install openpyxl`。
- 如果你的输出列不是最后一列：把 `target_column=-1` 改成列名（例如 `"y"`）或者目标列索引。
- 如果数据量很小，建议减小 `batch_size`（比如 16/32）。
