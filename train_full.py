import subprocess
from pathlib import Path
import time
import re


N_ITER = 3  # number of noisy student iterations
CKPT_DIR = Path("checkpoints/backbones/task1/ckpts/1024/medvit+asl+1024")
FUSION_CKPT_DIR = Path("checkpoints/fusion/task1/ckpts/1024/medvit+asl+1024/")
SCRIPTS_DIR = Path("scripts/task1")

MODEL_PATH = Path("checkpoints/backbones/task1/models/medvit-s-1024-asl.pth")
MODEL_FUSION_PATH = Path("checkpoints/fusion/task1/models/medvit-s-1024-asl.pth")


def get_best_ckpt(ckpt_dir: Path) -> Path | None:
    """
    Get the most recent checkpoint file from the specified directory.
    """

    def best_ap_idx(ckpts: list[Path]) -> int:
        max_ap, max_idx = 0.0, 0
        for i, ckpt in enumerate(ckpts):
            if ckpt.stem.find("last") != -1:
                continue
            val_ap = ckpt.stem.split("=")[-1]
            if float(val_ap) >= max_ap:
                max_ap = float(val_ap)
                max_idx = i

        return max_idx

    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}.")

    best_ckpt = ckpt_files[best_ap_idx(ckpt_files)]

    return best_ckpt


def submit_job(
    script_path: Path,
    ckpt_path: Path | None = None,
    use_pseudo: int = 0,
    predict: bool = False,
    extras: list[str | Path] = [],
) -> str:
    if not ckpt_path:
        cmd = ["sbatch", str(script_path), str(use_pseudo)]
    else:
        if predict and extras:
            cmd = ["sbatch", str(script_path), str(ckpt_path), *extras]
        elif predict:
            cmd = ["sbatch", str(script_path), str(ckpt_path)]
        elif not use_pseudo:
            cmd = ["sbatch", str(script_path), str(ckpt_path)]
        else:
            cmd = ["sbatch", str(script_path), str(use_pseudo), str(ckpt_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise ValueError(
            f"Could not extract job ID from sbatch output: {result.stdout}"
        )
    job_id = match.group(1)
    print(f"Submitted job {job_id}")

    return job_id


def wait_for_job_completion(job_id, poll_interval=60):
    while True:
        # Check if the job is still in the queue
        result = subprocess.run(
            ["squeue", "--job", job_id], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Warning: squeue failed: {result.stderr}")
            time.sleep(poll_interval)
            continue
        lines = result.stdout.strip().split("\n")
        # The first line is the header if job exists in queue
        if len(lines) <= 1:
            print(f"Job {job_id} is no longer in the queue. Assuming it has finished.")
            # Check job status with sacct to verify if it completed successfully
            sacct = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
            )
            if sacct.returncode != 0:
                print(f"Warning: sacct failed: {sacct.stderr}")
            else:
                # Parse the output to get job state
                states = [
                    s.strip() for s in sacct.stdout.strip().split("\n") if s.strip()
                ]
                if states and any(
                    state in ["FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"]
                    for state in states
                ):
                    raise RuntimeError(f"Job {job_id} failed with state: {states[0]}")
                print(
                    f"Job {job_id} completed with state: {states[0] if states else 'UNKNOWN'}"
                )
            break
        else:
            print(f"Job {job_id} is still running...")
            time.sleep(poll_interval)


if __name__ == "__main__":
    train_script = SCRIPTS_DIR / "train_backbone.sh"
    predict_script = SCRIPTS_DIR / "predict_pseudo_labels.sh"
    fusion_script = SCRIPTS_DIR / "train_fusion.sh"
    predict_fusion_script = SCRIPTS_DIR / "predict_final.sh"

    scripts_found = all(
        script.exists()
        for script in [
            train_script,
            predict_script,
            fusion_script,
            predict_fusion_script,
        ]
    )
    if not scripts_found:
        raise FileNotFoundError(f"Required scripts not found in {SCRIPTS_DIR}.")

    for idx in range(N_ITER + 1):
        if idx == 0:
            print("INITIAL TRAINING RUN WITH EXISTING LABELS...")
            job_id = submit_job(train_script)
            wait_for_job_completion(job_id)
            continue

        print(f"\nNOISY STUDENT ITERATION {idx}")
        ckpt = get_best_ckpt(CKPT_DIR)

        print(f"USING CHECKPOINT: {ckpt}")

        job_id = submit_job(predict_script, ckpt, predict=True)
        print(f"Predicting pseudolabels. Job ID: {job_id}...")
        wait_for_job_completion(job_id)

        job_id = submit_job(train_script, ckpt, use_pseudo=1)
        print(f"Training using pseudolabels. Job ID: {job_id}...")
        wait_for_job_completion(job_id)

    ckpt = get_best_ckpt(CKPT_DIR)
    print(f"Best checkpoint after all iterations: {ckpt}")

    res = subprocess.run(
        [
            "uv",
            "run",
            "save_model.py",
            "--ckpt",
            str(ckpt),
            "--save_to",
            MODEL_PATH,
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Failed to save model: {res.stderr}")
    print(res.stdout.strip())

    job_id = submit_job(fusion_script, MODEL_PATH)
    print(f"Training fusion model. Job ID: {job_id}...")
    wait_for_job_completion(job_id)

    fusion_ckpt = get_best_ckpt(FUSION_CKPT_DIR)
    print(f"Best fusion checkpoint: {fusion_ckpt}")

    res = subprocess.run(
        [
            "uv",
            "run",
            "save_model.py",
            "--ckpt",
            str(fusion_ckpt),
            "--save_to",
            MODEL_FUSION_PATH,
            "--type",
            "f",
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Failed to save fusion model: {res.stderr}")
    print(res.stdout.strip())

    job_id = submit_job(
        predict_fusion_script, fusion_ckpt, predict=True, extras=[MODEL_FUSION_PATH]
    )
    print(f"Predicting final labels with fusion model. Job ID: {job_id}...")
    wait_for_job_completion(job_id)
    print("All jobs completed successfully.")
