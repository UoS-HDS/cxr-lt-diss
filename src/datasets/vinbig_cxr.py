"""
VinDr-CXR Dataset. Adapted from
https://github.com/dongkyunk/CheXFusion/blob/main/dataset/vin_dataset.py
"""

from pathlib import Path

import cv2
import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset


class VinBigDataset(Dataset):
    def __init__(self, cfg: dict, df: DataFrame, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index][self.cfg["classes"]].to_numpy().astype(np.float32)
        path = (
            Path(self.cfg["data_dir"])
            / "vinbig-cxr-png/train"
            / f"{self.df.iloc[index]['image_id']}.png"
        )
        resized_path = str(path.relative_to(self.cfg["data_dir"]))
        resized_path = (
            Path(self.cfg["resized_dir"]) / str(self.cfg["size"]) / resized_path
        )
        resized_path.parent.mkdir(parents=True, exist_ok=True)

        if Path(resized_path).exists():
            img = cv2.imread(str(resized_path))
            assert img is not None, f"Image not found at {resized_path}"
            assert img.shape == (self.cfg["size"], self.cfg["size"], 3)
        else:
            img = cv2.imread(str(path))
            assert img is not None, f"Image not found at {path}"
            img = cv2.resize(
                img,
                (self.cfg["size"], self.cfg["size"]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            cv2.imwrite(str(resized_path), img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
            img = np.moveaxis(img, -1, 0)

        return img, label
