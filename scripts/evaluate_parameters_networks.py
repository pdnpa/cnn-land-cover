## File to evaluate trained LCUs

import os, sys
import pickle
sys.path.append('scripts/')
import numpy as np
import rasterio, rasterio.plot
import shapely as shp
import xarray as xr
import rioxarray as rxr
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import loadpaths
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import land_cover_models as lcm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch 
import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
path_dict = loadpaths.loadpaths()

## Parameters:
save_accuracy_results = True
path_dict_results = '/home/tplas/repos/cnn-land-cover/dict_eval_results_LCU_2022-11-30-1205_dissolving-area-sweep_4.pkl'

## Load trained network
LCU = lcm.load_model(filename='LCU_2022-11-30-1205.data')
LCU.eval()

## Evaluate network on evaluation tiles 
dict_eval_stats = {} ## dict to save all accuracy metrics etc. 
# threshold_array = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
threshold_array = [1000]

if save_accuracy_results:
    with open(path_dict_results, 'wb') as f:  # test save 
        pickle.dump(dict_eval_stats, f)

for i, thresh in enumerate(threshold_array):
    print(f'Loop {i}. Evaluating on tiles with area threshold of {thresh} m2')
    dict_eval_stats[thresh] = {}
    tmp_results = lcm.tile_prediction_wrapper(model=LCU, save_shp=True, save_folder=None,
                                dir_im='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/12.5cm Aerial Photo/',
                                dir_mask_eval='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_2022/',
                                dissolve_small_pols=True, area_threshold=thresh,
                                skip_factor=None)
    dict_eval_stats[thresh]['dict_acc_tiles'] = tmp_results[0]
    dict_eval_stats[thresh]['dict_df_stats_tiles'] = tmp_results[1]
    dict_eval_stats[thresh]['dict_conf_mat'] = tmp_results[2]

    ## Save results every loop.
    if save_accuracy_results:
        with open(path_dict_results, 'wb') as f:
            pickle.dump(dict_eval_stats, f)

print('\n\nDone!')