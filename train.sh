#!/bin/bash
#SBATCH --job-name=apptainer_test
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=long-short
#SBATCH --time=12:00:00
#SBATCH --mem=4G

# # Load and activate Conda environment
# source ~/.bashrc
# conda activate your_environment_name

cd /sharedscratch/$USER/hds-diss/images

# Load Apptainer module
# module load apptainer

# Execute training script
apptainer shell --nv \
    --home /sharedscratch/$USER/hds-diss/:$HOME \
    pytorch_25.05-py3.sif

# Confirmation message
echo "Job complete"