# Soft Sensor Regression from Excel Data (5 Inputs, 1 Output)

如果你的软测量数据保存在 Excel，且前 5 列为输入、最后 1 列为输出，可以直接用 `TabularRegressionDataModule`。

## 1) 准备数据模块（真正的代码改动已在库内提供）

```python
from lightning_uq_box.datamodules import TabularRegressionDataModule

# 你的文件路径
dm = TabularRegressionDataModule(
    data_path=r"C:\Users\13377\Desktop\test1.xlsx",
    n_input_features=5,   # 前5列输入
    target_column=-1,     # 最后一列输出
    batch_size=64,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
)
```

## 2) 训练一个不确定性感知回归器（MC Dropout）

```python
import torch
from lightning import Trainer

from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import MCDropoutRegression

model = MLP(n_inputs=5, n_hidden=[64, 64], n_outputs=1, dropout_p=0.1)

uq_model = MCDropoutRegression(
    model=model,
    optimizer=torch.optim.Adam,
    lr=1e-3,
)

trainer = Trainer(
    max_epochs=200,
    accelerator="auto",
    devices=1,
    enable_checkpointing=False,
    logger=False,
)

trainer.fit(uq_model, datamodule=dm)
```

## 3) 预测并还原到工程量纲

```python
import numpy as np

pred_dict = trainer.predict(uq_model, datamodule=dm)

pred_mean = np.concatenate([b["pred"].cpu().numpy() for b in pred_dict], axis=0)
pred_uct = np.concatenate([b["pred_uct"].cpu().numpy() for b in pred_dict], axis=0)

# 还原到原始输出量纲（前提是 scale_target=True，默认就是 True）
pred_mean_real = dm.y_scaler.inverse_transform(pred_mean)

print("Prediction shape:", pred_mean_real.shape)
print("Uncertainty shape:", pred_uct.shape)
```

## 说明

- 若输出列不是最后一列，可将 `target_column` 设为列名（如 `"y"`）或其他列索引。
- 当前 `TabularRegressionDataModule` 支持 `.xlsx/.xls/.csv`。
