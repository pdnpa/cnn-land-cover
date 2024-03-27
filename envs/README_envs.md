# Conda environments
### Contents

The following environment files are included:
- `geo_exactbuilds.txt`: the list of all packages (including dependencies) with their specific builds, used for developing and evaluating the models in _Van der Plas et al., 2023, Remote Sensing_.
- `geo_minimalbuilds.yml`: list of packages as originally used for the paper, installed with conda. Builds not specified. 
- `geo_newbuilds.yml`: list of packages installed with conda, using newest versions of packages (e.g., python 3.10 instead of 3.7). 

### Installation
- For the exact same build, create a new conda environment (on a linux/cuda set up) using: `pip install -r geo_exactbuilds.txt`
- `geo_minimalbuilds.yml` contains all python package versions used for this repo. Create a new virtual environment with all these packages with Anaconda by typing in terminal: `conda env create -f geo.yml` and then `conda activate geo`. Next, install final packages using pip: `pip install geocube==0.1.0`, `pip install patchify`, `pip install segmentation-models-pytorch`, `pip install torchsummary`, `pip install pytorch==1.12.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html`, `pip install torch-tb-profiler`, `pip install -U kaleido`