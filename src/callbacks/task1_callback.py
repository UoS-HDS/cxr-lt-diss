import os
import zipfile
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class FusionSubmissonWriter(BasePredictionWriter):
    def __init__(self, sample_task1_path, task1_path, task1_zip_path, task1_code_dir, pred_df_path, write_interval):
        super().__init__(write_interval)
        self.sample_task1_path = sample_task1_path
        self.task1_path = task1_path
        self.task1_zip_path = task1_zip_path
        self.task1_code_dir = task1_code_dir
        self.pred_df_path = pred_df_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Add predictions
        predictions = torch.cat(predictions, dim=0)
        torch.save(predictions, "predictions.pt")

        submit_df = pd.read_csv(self.sample_task1_path)
        pred_df = pd.read_csv(self.pred_df_path)
        submit_df['study_id'] = pred_df['study_id']

        temp_df = pd.DataFrame(predictions, columns=submit_df.columns[-27:-1])
        temp_df['study_id'] = list(pred_df.groupby('study_id').groups.keys())
        submit_df = submit_df.merge(temp_df, on='study_id', how='left', suffixes=('_x', ''))

        # Remove _x columns
        submit_df = submit_df.loc[:, ~submit_df.columns.str.endswith('_x')]
        submit_df.drop(columns=['study_id'], inplace=True)

        # Save submission
        submit_df.to_csv(self.task1_path, index=False)
        with zipfile.ZipFile(self.task1_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the folder and its contents to the zip
            for root, _, files in os.walk(self.task1_code_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.join('code',os.path.relpath(file_path, self.task1_code_dir)))

            # Add the file to the zip
            zipf.write(self.task1_path, os.path.basename(self.task1_path))
        
        print(f"Submission saved!")
