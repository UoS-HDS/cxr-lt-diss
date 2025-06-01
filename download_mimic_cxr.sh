#SBATCH --chdir=/sharedscratch/na200/id5059-group-5/
#SBATCH --job-name=dowload_mimic_cxr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=bigmem
#SBATCH --time=10:00:00
#SBATCH --mem=24G
#SBATCH --output=mimic_cxr.out

export PATH=/sharedscratch/na200/hds-diss/.venv/bin/:$PATH
cd /sharedscratch/na200/hds-diss/
uv run download_mimic_cxr.py

echo "Job complete"