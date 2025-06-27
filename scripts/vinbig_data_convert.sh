#!/bin/bash
#SBATCH --job-name=vinbig_data_convert
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=bigmem
#SBATCH --time=12:00:00
#SBATCH --mem=16G

project_dir="/sharedscratch/$USER/hds-diss"

cd ${project_dir}

# Execute training script
apptainer exec \
    --home ${project_dir}:$HOME \
    apptainer/pytorch_25.05-py3-uv.sif \
    uv sync -q && \
    uv run vinbig_data_convert.py

# Confirmation message
echo "Job complete"