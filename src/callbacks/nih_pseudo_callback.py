"""
NIH CXR pseudo label writer callback. Adapted from

"""

from typing import Literal
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter

from src.utils import flatten


class NIHWriter(BasePredictionWriter):
    def __init__(
        self,
        nih_train_df_path: str,
        nih_pseudo_train_df_path: str,
        write_interval: Literal["epoch", "batch", "batch_and_epoch"],
        num_classes: int = 40,
    ):
        super().__init__(write_interval)
        self.nih_train_df_path = Path(nih_train_df_path)
        self.nih_pseudo_train_df_path = Path(nih_pseudo_train_df_path)
        self.num_classes = num_classes

        self.nih_pseudo_train_df_path.parent.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        preds = predictions.float().squeeze(0).detach().cpu().numpy()

        save_dir = Path(self.nih_pseudo_train_df_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "nih_preds.npy", preds)

        n = self.num_classes

        nih_train_df = pd.read_csv(self.nih_train_df_path)
        labels = np.array(nih_train_df.iloc[:, -n:].values).astype(np.float32)

        # Flatten batch_indices to get the actual row indices
        batch_indices = list(flatten(batch_indices))
        batch_indices = np.array(batch_indices).astype(int)

        # Replace the original labels with the pseudo labels only
        # if labels value is -1 (unmentioned and non-existent labels)
        predicted_rows = labels[batch_indices]  # Get all predicted rows at once
        mask = predicted_rows == -1.0
        predicted_rows[mask] = preds[mask]
        labels[batch_indices] = predicted_rows

        nih_train_df.iloc[:, -n:] = labels
        nih_train_df.to_csv(self.nih_pseudo_train_df_path, index=False)

        print(f"NIH pseudo labels saved to {self.nih_pseudo_train_df_path}")
