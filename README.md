# Multi-stage semantic segmentation of land cover in the Peak District using high-resolution RGB aerial imagery

This is the code corresponding to the following publication. If you use our code, please cite:

van der Plas, T.L.; Geikie, S.T.; Alexander, D.G.; Simms, D.M. Multi-Stage Semantic Segmentation Quantifies Fragmentation of Small Habitats at a Landscape Scale. _Remote Sensing_ **2023**, 15, 5277. [https://doi.org/10.3390/rs15225277](https://doi.org/10.3390/rs15225277)

### Installation:
1. To ensure you've got all the necessary packages, follow the instructions in `envs/README_envs.md` to install a new conda environment with the correct set of packages.
2. Set your user-specific file paths in `content/data_paths.json`. There is "new-username" template that you can use to enter your paths (using your computer username). An explanation of what each path is for is given in `content/README_datapaths.md`. 
3. If you want to train new models using the same data set as [our paper](https://doi.org/10.3390/rs15225277), you can download the images from this [data repository](https://cord.cranfield.ac.uk/articles/dataset/Very_high_resolution_aerial_photography_and_annotated_land_cover_data_of_the_Peak_District_National_Park/24221314)
4. If you want to use the CNN models from our paper for predicting new image tiles, you can download these [here](https://drive.google.com/drive/folders/1nEnIWDvWcLVzSE6yViv93I4klY2WzdDo?usp=sharing). 
5. _**TBA** Download extra files if possible, or run without_

### Usage
1. **Example notebook**: Please see `notebooks/Getting started.ipynb` for an example notebook of how to load the data set, models etc. 
2. **Training a LC segmentation model**: There is a script provided in `scripts/train_segmentation_network.py`. See the function call under `if __name__ == '__main__':` for an example of how to call the function. It trains a network using a folder of RGB image patches and a folder of LC annotation mask patches. These can be downloaded from our data repository (see above). 
3. **Predicting LC of new images using an existing model**: There is a script provided in `scripts/prediction_of_trained_network.py`.  See the function call under `if __name__ == '__main__':` for an example of how to call the function. It predicts entire RGB image tiles, which it splits up into patches, predicts LC of the patches, reconstructs the tile and saves. The CNN models from the paper can be downloaded [here](https://drive.google.com/drive/folders/1nEnIWDvWcLVzSE6yViv93I4klY2WzdDo?usp=sharing).
4. Figures for the paper are generated in Jupyter notebooks, see all notebooks in `notebooks/` with a file name starting with `Figure ...`.

### Miscellaneous:
- Additionally, we have created an interpretation key of all land cover classes at [https://reports.peakdistrict.gov.uk/interpretation-key/docs/introduction.html](https://reports.peakdistrict.gov.uk/interpretation-key/docs/introduction.html)
- For more details, please see our [Remote Sensing](https://www.mdpi.com/2072-4292/15/22/5277) publication. 
