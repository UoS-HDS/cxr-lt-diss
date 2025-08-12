"""
Directory and path management utilities
"""

from pathlib import Path
from typing import Dict


def create_experiment_directories(paths: Dict[str, Path]) -> None:
    """Create all necessary directories for an experiment"""

    directories_to_create = [
        paths["checkpoint_dir"],
        paths["fusion_checkpoint_dir"],
        paths["submission_dir"],
        paths["pseudo_label_dir"] / "chexpert",
        paths["pseudo_label_dir"] / "vinbig",
        paths["pseudo_label_dir"] / "nih",
        paths["configs_dir"],
        paths["scripts_dir"],
        paths["tb_log_dir"],
        Path("logs"),
    ]

    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)

    print("✅ Created experiment directories")
