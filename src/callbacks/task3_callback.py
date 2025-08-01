import os
from pathlib import Path
import zipfile
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class Task3SubmissionWriter(BasePredictionWriter):
    def __init__(
        self,
        sample_submit_path,
        submit_path,
        submit_zip_path,
        submit_code_dir,
        pred_df_path,
        write_interval,
    ):
        super().__init__(write_interval)
        self.sample_submit_path = Path(sample_submit_path)
        self.submit_path = Path(submit_path)
        self.submit_zip_path = Path(submit_zip_path)
        self.submit_code_dir = Path(submit_code_dir)
        self.pred_df_path = Path(pred_df_path)

        self.submit_path.parent.mkdir(parents=True, exist_ok=True)
        self.submit_zip_path.parent.mkdir(parents=True, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Add predictions
        predictions = torch.cat(predictions, dim=0)  # type: ignore
        predictions_name = (
            "predictions_test" if "test" in self.submit_path.name else "predictions"
        )
        torch.save(predictions, self.submit_path.parent / f"{predictions_name}.pt")

        # submit_df = pd.read_csv(self.sample_submit_path)
        pred_df = pd.read_csv(self.pred_df_path)
        # submit_df['study_id'] = pred_df['study_id']

        preds = predictions.cpu().numpy() >= 0.5

        temp_df = pd.DataFrame(preds.astype(int), columns=pred_df.columns[-5:])
        temp_df["study_id"] = list(pred_df.groupby("study_id").groups.keys())
        temp_df = pred_df[["study_id", "dicom_id"]].merge(
            temp_df,
            on="study_id",
            how="left",
        )
        # temp_df.insert(0, "study_id", pred_df["study_id"].values.tolist())
        # temp_df.insert(1, "dicom_id", pred_df["dicom_id"].values.tolist())
        # submit_df = pred_df.merge(temp_df, on='study_id', how='left', suffixes=('_x', ''))

        # Remove _x columns
        # submit_df = submit_df.loc[:, ~submit_df.columns.str.endswith('_x')]
        # submit_df.drop(columns=['study_id'], inplace=True)

        # Save submission
        # submit_df.to_csv(self.submit_path, index=False)
        temp_df.to_csv(self.submit_path, index=False)
        # with zipfile.ZipFile(self.submit_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        #     # Add the folder and its contents to the zip
        #     for root, _, files in os.walk(self.submit_code_dir):
        #         for file in files:
        #             file_path = os.path.join(root, file)
        #             zipf.write(file_path, os.path.join('code',os.path.relpath(file_path, self.submit_code_dir)))

        #     # Add the file to the zip
        #     zipf.write(self.submit_path, os.path.basename(self.submit_path))

        print(f"Submission saved: {self.submit_path}")
