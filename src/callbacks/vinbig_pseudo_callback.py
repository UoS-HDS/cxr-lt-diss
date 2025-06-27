"""
VinBigData pseudo label writer callback. Adapted from

"""

from typing import Literal
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter

from src.util import flatten


class VinBigWriter(BasePredictionWriter):
    def __init__(
        self,
        vinbig_train_df_path: str,
        vinbig_pseudo_train_df_path: str,
        write_interval: Literal["epoch", "batch", "batch_and_epoch"],
    ):
        super().__init__(write_interval)
        self.vinbig_train_df_path = vinbig_train_df_path
        self.vinbig_pseudo_train_df_path = vinbig_pseudo_train_df_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        preds = predictions.float().squeeze(0).detach().cpu().numpy()

        vinbig_train_df = pd.read_csv(self.vinbig_train_df_path)
        labels = np.array(vinbig_train_df.iloc[:, -40:].values).astype(np.float32)

        # Flatten batch_indices to get the actual row indices
        batch_indices = list(flatten(batch_indices))
        batch_indices = np.array(batch_indices).astype(int)

        # Replace the original labels with the pseudo labels only
        # if labels value is -1 (unmentioned and non-existent labels)
        predicted_rows = labels[batch_indices]  # Get all predicted rows at once
        mask = predicted_rows == -1.0
        predicted_rows[mask] = preds[mask]
        labels[batch_indices] = predicted_rows

        # If both column nodule and mass is 1, then replace the one with lower pred with the pred value
        # mass is column 22 and nodule is column 23
        both_ones_indices = np.where((labels[:, 22] == 1) & (labels[:, 23] == 1))[0]

        for index in both_ones_indices:
            if preds[index, 22] < preds[index, 23]:
                labels[index, 22] = preds[index, 22]
            else:
                labels[index, 23] = preds[index, 23]

        vinbig_train_df.iloc[:, -40:] = labels
        vinbig_train_df.to_csv(self.vinbig_pseudo_train_df_path, index=False)

        print(f"VinBig pseudo labels saved to {self.vinbig_pseudo_train_df_path}")
