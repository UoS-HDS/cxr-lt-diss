"""
processes vinbig dataset to convert dicom images
to png format
"""

from pathlib import Path
import asyncio

import pydicom
from pydicom.pixels import apply_voi_lut, apply_modality_lut  # type: ignore
import cv2
import numpy as np
from tqdm.asyncio import tqdm


VINBIG_DICOM_DIR = Path("./data/vinbig-cxr/")
OUTPUT_PATH = Path("./data/vinbig-cxr-png/")
CONCURRENT_TASKS = 2000

if not VINBIG_DICOM_DIR.exists():
    raise FileNotFoundError(f"Directory {VINBIG_DICOM_DIR} does not exist.")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


async def standardise_img(img: np.ndarray, scheme: str) -> np.ndarray:
    """Standardize the image to have zero mean and unit variance."""
    img_std = ((img - img.min()) / img.max()) * 255.0
    if scheme == "MONOCHROME1":
        img_std = 255.0 - img_std

    return img_std.astype(np.uint8)


async def process_dicom_file(dicom_file: Path, sem: asyncio.Semaphore) -> None:
    """process single dicom grayscale image file"""
    async with sem:
        try:
            ds = pydicom.dcmread(dicom_file)

            colour_scheme = ds.PhotometricInterpretation
            img = apply_modality_lut(ds.pixel_array, ds)
            img = apply_voi_lut(img, ds)

            img_std = await standardise_img(img, colour_scheme)

            # replace vinbig-cxr directory with vinbig-cxr-png
            output_file = dicom_file.relative_to(VINBIG_DICOM_DIR)
            output_file = OUTPUT_PATH / output_file.with_suffix(".png")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processed {dicom_file} -> {output_file}")
            success = cv2.imwrite(str(output_file), img_std)
            if not success:
                raise IOError(f"Failed to write image to {output_file}")
        except Exception as e:
            print(f"Error processing {dicom_file}: {e}")


async def main():
    """"""
    SEMAPHORE = asyncio.Semaphore(CONCURRENT_TASKS)

    dicom_files = VINBIG_DICOM_DIR.rglob("*.dicom")
    tasks = [asyncio.create_task(process_dicom_file(f, SEMAPHORE)) for f in dicom_files]
    total_files = len(tasks)
    print(f"Total DICOM files to process: {total_files}")

    for t in tqdm(
        asyncio.as_completed(tasks),
        total=total_files,
        desc="Processing DICOM files",
    ):
        await t


if __name__ == "__main__":
    asyncio.run(main())
