# Automated classification of land cover using CNNs

### Installation:
- `geo.yml` contains all python package versions used for this repo. Create a new virtual environment with all these packages with Anaconda by typing in terminal: `conda env create -f geo.yml` and then `conda activate geo`. 
- Next, install final packages using pip: `pip install geocube==0.1.0`, `pip install patchify`, `pip install segmentation-models-pytorch`, `pip install torchsummary`, `pip install pytorch==1.12.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html`, `pip install torch-tb-profiler`, `pip install git+https://github.com/geopandas/dask-geopandas.git`
- User-specific file paths are stored in `content/data_paths.json`.
