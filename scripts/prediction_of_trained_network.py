## Perform prediction on full tiles using a trained model on a test set

import os, sys, json, pickle
import datetime
import loadpaths
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import land_cover_models as lcm

path_dict = loadpaths.loadpaths()
lca.check_torch_ready(check_gpu=True, assert_versions=True)

##Parameters:
datapath_model = 'LCU_2023-01-23-2018.data'
save_shp_prediction = True
dissolve_small_pols = True
dissolve_threshold = 1000
override_with_fgh_layer = True
skip_factor = 16
padding = 44
parent_save_folder = '/home/tplas/predictions/'
subsample_tiles_for_testing = False

## Load model:
LCU = lcm.load_model(filename=datapath_model)
LCU.eval() 
if dissolve_small_pols:
    dissolved_name = '_dissolved' + str(dissolve_threshold) + 'm2'
else:
    dissolved_name = ''
save_folder = os.path.join(parent_save_folder, 'predictions_' + LCU.model_name + dissolved_name + f'_padding{padding}')

## Predict full tiles of test set:
tmp_results = lcm.tile_prediction_wrapper(model=LCU, save_shp=save_shp_prediction,
                            dir_im='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/12.5cm Aerial Photo/',
                            dir_mask_eval='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_2022/',
                            save_folder=save_folder, dissolve_small_pols=dissolve_small_pols, 
                            area_threshold=dissolve_threshold, skip_factor=skip_factor,
                            padding=padding,
                            subsample_tiles_for_testing=subsample_tiles_for_testing)

## Save results as pickle:
with open(os.path.join(save_folder, 'summary_results.pkl'), 'wb') as f:
    pickle.dump(tmp_results, f)
print('\nResults saved!\n\n')

## Override predictions with manual FGH layer:
if override_with_fgh_layer:
    print('######\n\nOverride predictions with manual FGH layer\n\n######')
    save_folder = lca.override_predictions_with_manual_layer(filepath_manual_layer='/home/tplas/data/gis/tmp_fgh_layer/tmp_fgh_layer.shp', 
                                                             tile_predictions_folder=save_folder, 
                                                             new_tile_predictions_override_folder=None, verbose=1)

## Merge all tiles into one shapefile:
lca.merge_individual_shp_files(dir_indiv_tile_shp=save_folder)
