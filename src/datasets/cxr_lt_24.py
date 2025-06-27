from pathlib import Path
from typing import Any

import cv2
from pandas import DataFrame
import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose


class CxrDataset(Dataset):
    """Default CXR-LT 24 dataset loader"""

    def __init__(
        self,
        cfg: dict[str, Any],
        df: DataFrame,
        transform: Compose | None = None,
    ):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if all([c in self.df.columns for c in self.cfg["classes"]]):
            label = (
                self.df.iloc[index][self.cfg["classes"]].to_numpy().astype(np.float32)
            )
        else:
            label = np.zeros(len(self.cfg["classes"]))

        path = (
            Path(self.cfg["data_dir"])
            / "mimic-cxr-jpg-2.1.0"
            / Path(self.df.iloc[index]["fpath"])
        )
        resized_path = str(path.relative_to(self.cfg["data_dir"]))
        resized_path = (
            Path(self.cfg["resized_dir"]) / str(self.cfg["size"]) / resized_path
        )
        resized_path.parent.mkdir(parents=True, exist_ok=True)

        if Path(resized_path).exists():
            img = cv2.imread(str(resized_path))
            assert img.shape == (self.cfg["size"], self.cfg["size"], 3)
        else:
            img = cv2.imread(str(path))
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


# class CxrBalancedDataset(Dataset):
#     """Balanced CXR-LT 24 dataset loader"""

#     def __init__(self, cfg: dict[str, Any], df: DataFrame, transform=None):
#         self.cfg = cfg
#         self.df = df
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         class_name = self.cfg["classes"][index % len(self.cfg["classes"])]
#         df = self.df[self.df[class_name] == 1].sample(1).iloc[0]

#         label = df[self.cfg["classes"]].to_numpy().astype(np.float32)

#         path = df["path"]
#         path = os.path.join(self.cfg["data_dir"], path)
#         resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

#         if os.path.exists(resized_path):
#             img = cv2.imread(str(resized_path))
#             assert img.shape == (self.cfg["size"], self.cfg["size"], 3)
#         else:
#             img = cv2.imread(str(path))
#             img = cv2.resize(
#                 img,
#                 (self.cfg["size"], self.cfg["size"]),
#                 interpolation=cv2.INTER_LANCZOS4,
#             )
#             cv2.imwrite(resized_path, img)

#         if self.transform:
#             transformed = self.transform(image=img)
#             img = transformed["image"]
#             img = np.moveaxis(img, -1, 0)

#         return img, label


class CxrStudyIdDataset(Dataset):
    """CXR-LT 24 dataset loader grouped by study_id"""

    def __init__(
        self,
        cfg: dict[str, Any],
        df: DataFrame,
        transform: Compose | None = None,
    ):
        self.cfg = cfg
        self.studies = df.groupby("study_id")
        self.study_ids = list(self.studies.groups.keys())
        self.transform = transform

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        df = self.studies.get_group(self.study_ids[index])
        if len(df) > 4:
            df = df.sample(4)

        if all([c in df.columns for c in self.cfg["classes"]]):
            label = df[self.cfg["classes"]].iloc[0].to_numpy().astype(np.float32)
        else:
            label = np.zeros(len(self.cfg["classes"]))

        study_imgs = []
        for i in range(len(df)):
            path = (
                Path(self.cfg["data_dir"])
                / "mimic-cxr-jpg-2.1.0"
                / Path(df.iloc[i]["fpath"])
            )
            resized_path = str(path.relative_to(self.cfg["data_dir"]))
            resized_path = (
                Path(self.cfg["resized_dir"]) / str(self.cfg["size"]) / resized_path
            )
            resized_path.parent.mkdir(parents=True, exist_ok=True)

            if Path(resized_path).exists():
                img = cv2.imread(str(resized_path))
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
            study_imgs.append(img)

        study_imgs = np.stack(study_imgs, axis=0)
        study_imgs = np.concatenate(
            [
                study_imgs,
                np.zeros((4 - len(df), 3, self.cfg["size"], self.cfg["size"])),
            ],
            axis=0,
        ).astype(np.float32)

        return study_imgs, label
