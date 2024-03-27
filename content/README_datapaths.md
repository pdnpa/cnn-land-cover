# Explainer data paths

All user-specific data paths used by any of the modules are stored in `content/data_paths.json`. To use this code, please add your profile (username of your computer) and relevant data paths. This is an explainer what each data path is for, and which ones are essential/optional. _In general, these default paths can still be overridden by using function arguments wherever necessary._

## Essential:
- **models**: FOLDER where models 1) will be saved and 2) will be loaded from. 
- **save_folder**: FOLDER where predicted LC tiles will be stored.

## Essential for training a new model:
- **im_patches**: FOLDER that contains the image patches for training/testing. (Can be downloaded from this [data repository](https://cord.cranfield.ac.uk/articles/dataset/Very_high_resolution_aerial_photography_and_annotated_land_cover_data_of_the_Peak_District_National_Park/24221314), using the folder `python_format/images_python_all/`). **NB**: The code assumes that the annoted LC masks for training/testing are in the sibling-folder using the `mask_dir_name` argument in `DataSetPatches` (if you are using the data repository this is already taken care of (by `python_format/masks_python_all/`, and the default `mask_dir_name` argument in `scripts/train_segmentation_network.py` is already set to `masks_python_all`)).

## Essential for predicting new tiles:
- **im_tiles**: FOLDER that contains RGB image tiles to be predicted. 
- **parent_dir_tile_mainpred**: FOLDER than contains tile predictions of main classes. These are used to mask out the relevant classes when predicting subclasses (C, D and E). (These can be created by predicting the main classes of image tiles).

## Optional for predicting new tiles:
- **fgh_layer**: SHP FILE of FGH classes used to overwrite main class predictions. 

## Optional:
- **path_habitat_prio**: SHP FILE of priority rush pasture habitats (can be downloaded from [here](https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::habitat-networks-england-purple-moor-grass-rush-pasture))
- **path_habitat_nonprio**: SHP FILE of non-priority rush pasture habitats (can be downloaded from [here](https://naturalengland-defra.opendata.arcgis.com/datasets/Defra::habitat-networks-england-purple-moor-grass-rush-pasture))
- **peaty_soils_layer**: SHP FILE of peaty soils data used to distinguish between peaty and non-peaty LC classes (can be downloaded from [here](https://naturalengland-defra.opendata.arcgis.com/datasets/1e5a1cdb2ab64b1a94852fb982c42b52_0/about))