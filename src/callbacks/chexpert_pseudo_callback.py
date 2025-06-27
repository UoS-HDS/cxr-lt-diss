"""
Chexpert pseudo label writer callback. Adapted from

"""

from typing import Literal
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter

from src.util import flatten


class ChexpertWriter(BasePredictionWriter):
    def __init__(
        self,
        chexpert_train_df_path: str,
        chexpert_pseudo_train_df_path: str,
        write_interval: Literal["epoch", "batch", "batch_and_epoch"],
    ):
        super().__init__(write_interval)
        self.chexpert_train_df_path = chexpert_train_df_path
        self.chexpert_pseudo_train_df_path = chexpert_pseudo_train_df_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        preds = predictions.float().squeeze(0).detach().cpu().numpy()

        save_dir = Path(self.chexpert_pseudo_train_df_path).parent
        np.save(save_dir / "chexpert_preds.npy", preds)

        chexpert_train_df = pd.read_csv(self.chexpert_train_df_path)
        org = np.array(chexpert_train_df.iloc[:, -40:].values).astype(np.float32)

        # Flatten batch_indices to get the actual row indices
        batch_indices = list(flatten(batch_indices))
        batch_indices = np.array(batch_indices).astype(int)

        # Replace the original labels with the pseudo labels only
        # if org value is -1 (unmentioned and non-existent labels)
        predicted_rows = org[batch_indices]  # Get all predicted rows at once
        mask = predicted_rows == -1.0
        predicted_rows[mask] = preds[mask]
        org[batch_indices] = predicted_rows

        chexpert_train_df.iloc[:, -40:] = org
        chexpert_train_df.to_csv(self.chexpert_pseudo_train_df_path, index=False)

        print(f"Chexpert pseudo labels saved to {self.chexpert_pseudo_train_df_path}")
