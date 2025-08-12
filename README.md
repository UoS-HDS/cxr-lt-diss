# CXR-LT: LONG-TAILED MULTI-LABEL CXR CLASSIFICATION
This repo contains the code for running the experiments carried out for a Master's Dissertation on Long-Tailed Multi-Label Classification of Chest X-Rays. It is based on the CXR-LT [2023](https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/1.1.0/) and [2024](https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/). The challenge summary is published [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC12306832/).

The thesis extends CheXFusion (repo [here](https://github.com/dongkyunk/CheXFusion) and paper [here](https://ieeexplore.ieee.org/document/10350964))

## PROJECT STRUCTURE
```
data/  
|—— chexpert/CheXpert-v1.0  
|   |——train/  
|   |__valid/  
|   |__...  
|—— cxr-lt-iccv-workshop-cvamd/2.0.0/
|   |—— cxr-lt-2024/
|   |__...
|—— mimic-cxr-jpg-2.1.0
|   |—— files/
|   |    |—— p10/
|   |    |—— ...
|   |    |__ p19/
|   |__...
|—— nih-cxr/
|   |—— images/
|   |__...
|__ vinbig-cxr/
    |—— test/
    |—— train/
    |__...
containers/  # contains Docker and Apptainer container definitions
|—— pytorch_25.05-py3-uv.def  # Apptainer definition file
|__ DockerFile  # Docker definition file
checkpoints/  # directory for model checkpoints
configs/  # sample Pytorch Lightning yaml configs
scripts/  # sample docker and apptainer scripts
figures/  # plots
src/  # source code
submissions/  # results save dir
prelim_analysis.ipynb  # preliminary analysis notebook and prepares csv files for training
pretrain_datasets.ipynb  # pretraining dataset preparation notebook - csv files
main.py  # main entry point for training and evaluation
README.md  # this file
download_khub.py  # downloads NIH and VinBig data from KaggleHub
download_mimic_cxr_gc.py  # downloads MIMIC-CXR-JPG data from Google Cloud Storage
save_model.py  # saves the trained models
train_full_apptainer.py  # training script for Apptainer
train_full_docker.py  # training script for Docker
vinbig_data_convert.py  # converts VinBig data from dicom to jpg/png
pyproject.toml  # Python project configuration file
uv.lock  # Universal Virtual Environment lock file
```

## RUNNING EXPERIMENTS
### PREREQUISITES
Two options for running the experiments: Docker (the experiments were run on DGX system) and Apptainer (with SLURM manager)
- Download the datasets and other files into the `data/` directory (matching the structure above):
  - Challenge files that ties to the MIMIC-CXR-JPG dataset [here](https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/)
  - [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr/2.0.0/)
  - [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
  - [NIH Chest-Xray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)
  - [VinBigData](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data)
- Python `3.12` or higher
- UmlsBERT checkpoint used can be downloaded here: https://github.com/gmichalo/UmlsBERT?tab=readme-ov-file#dowload-pre-trained-umlsbert-model and extracted into the `checkpoint/` directory

### SETUP
- Build docker OR apptainer images:
  - Apptainer SIF image: `apptainer build <IMAGE_PATH> containers/pytorch_25.05-py3-uv.def`
  - Docker: `docker build -t <IMAGE_NAME> -f containers/DockerFile`
- Install `uv` for environment management. Instructions [here](https://docs.astral.sh/uv/getting-started/installation/)
- Clone repo
- Run `uv sync` - installs all packages and generate virtual env
- Execute `prelim_analysis.ipynb` and `pretrain_datasets.ipynb` (in that order) to perform preliminary analysis and prepare CSV files for training
- Run `vinbig_data_convert.py` to convert VinBig data from DICOM to JPG/PNG format can
- **NOTE: CHECK THE NOTEBOOKS AND CONVERSION FILES TO SET NECESSARY PATHS. NOT NEEDED IF YOU FOLLOW THE DIRECTORY STRUCTURE TO THE LETTER**
- Default experiment configuration is in `src/utils/experiment_config.py`. Make changes as required.

### EXECUTION
- To run experiments, use either `train_full_apptainer.py` or `train_full_docker.py` depending on your setup.
  - Apptainer: `uv run train_full_apptainer.py [<OPTIONS>]` or `UV_PROJECT_ENVIRONMENT=<ENV_PATH> uv run train_full_apptainer.py [<OPTIONS>]` if you want to specify an environment (recommended)
  - Docker: `uv run train_full_docker.py [<OPTIONS>]` or `UV_PROJECT_ENVIRONMENT=<ENV_PATH> uv run train_full_docker.py [<OPTIONS>]`
  - **NOTE THAT THE AVAILABLE OPTIONS FOR EACH SETUP IS SLIGHTLY DIFFERENT**
- To view all available options, run `uv run train_full_apptainer.py --help` or `uv run train_full_docker.py --help`
- Tensorboard was used for tracking all experiments. Start it in a separate terminal with the command `uv run tensorboard --logdir=<LOG_DIR>` where `<LOG_DIR>` is the directory containing the Tensorboard logs (default is `.out/tb` which is automatically created)
