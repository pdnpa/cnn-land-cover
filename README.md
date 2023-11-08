# Multi-stage semantic segmentation of land cover in the Peak District using high-resolution RGB aerial imagery

This is the code corresponding to the following publication. If you use our code, please cite:

van der Plas, T.L.; Geikie, S.T.; Alexander, D.G.; Simms, D.M. Multi-Stage Semantic Segmentation Quantifies Fragmentation of Small Habitats at a Landscape Scale. _Remote Sensing_ **2023**, 15, 5277. [https://doi.org/10.3390/rs15225277](https://doi.org/10.3390/rs15225277)

### Repository contents:
- All code is in modules in `cnn-land-cover/scripts/`
- Example notebooks of how to use these modules, as well as the figure-generating notebooks, are located in `cnn-land-cover/notebooks/`

### Installation:
- `geo.yml` contains all python package versions used for this repo. Create a new virtual environment with all these packages with Anaconda by typing in terminal: `conda env create -f geo.yml` and then `conda activate geo`. 
- Next, install final packages using pip: `pip install geocube==0.1.0`, `pip install patchify`, `pip install segmentation-models-pytorch`, `pip install torchsummary`, `pip install pytorch==1.12.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html`, `pip install torch-tb-profiler`, `pip install -U kaleido`
- User-specific file paths are stored in `content/data_paths.json`.

### Data:
- The data for training and testing can be found at this [https://cord.cranfield.ac.uk/articles/dataset/Very_high_resolution_aerial_photography_and_annotated_land_cover_data_of_the_Peak_District_National_Park/24221314](CORD repository).
- Additionally, we have created an interpretation key of all land cover classes at [https://reports.peakdistrict.gov.uk/interpretation-key/docs/introduction.html](https://reports.peakdistrict.gov.uk/interpretation-key/docs/introduction.html)
- For more details, please see our [https://www.mdpi.com/2072-4292/15/22/5277](Remote Sensing) publication. 
