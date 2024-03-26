# Multi-stage semantic segmentation of land cover in the Peak District using high-resolution RGB aerial imagery

This is the code corresponding to the following publication. If you use our code, please cite:

van der Plas, T.L.; Geikie, S.T.; Alexander, D.G.; Simms, D.M. Multi-Stage Semantic Segmentation Quantifies Fragmentation of Small Habitats at a Landscape Scale. _Remote Sensing_ **2023**, 15, 5277. [https://doi.org/10.3390/rs15225277](https://doi.org/10.3390/rs15225277)

### Repository contents:
- All code is in organised modules in `cnn-land-cover/scripts/`
- Example notebooks of how to use these modules, as well as the figure-generating notebooks, are located in `cnn-land-cover/notebooks/`

### Installation:
1. To ensure you've got all the necessary packages, follow the instructions in `envs/envs_readme.md` to install a new conda environment with the correct set of packages.
2. Set your user-specific file paths in `content/data_paths.json`. There is "new-username" template that you can use to enter your paths (using your computer username).
3. If you want to train new models using the same data set as [our paper](https://doi.org/10.3390/rs15225277), you can download the images from this [data repository](https://cord.cranfield.ac.uk/articles/dataset/Very_high_resolution_aerial_photography_and_annotated_land_cover_data_of_the_Peak_District_National_Park/24221314)
4. _Download models_. 
5. _Download extra files if possible, or run without_

### Usage
1. _Example notebooks_
2. _Run training script_
3. _Run prediction script_
4. _Figure notebooks_

### Miscellaneous:
- Additionally, we have created an interpretation key of all land cover classes at [https://reports.peakdistrict.gov.uk/interpretation-key/docs/introduction.html](https://reports.peakdistrict.gov.uk/interpretation-key/docs/introduction.html)
- For more details, please see our [Remote Sensing](https://www.mdpi.com/2072-4292/15/22/5277) publication. 
