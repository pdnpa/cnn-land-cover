## Perform prediction on full tiles using a trained model on a test set

import os, sys, json, pickle
import datetime
import loadpaths
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import land_cover_models as lcm

def predict_segmentation_network(datapath_model=None, padding=44, 
                                 dissolve_small_pols = True,
                                dissolve_threshold = 1000,
                                clip_to_main_class=False,
                                main_class_clip_label='D',
                                skip_factor=16,
                                save_shp_prediction = True,
                                parent_save_folder = '/home/tplas/predictions/',
                                override_with_fgh_layer = False,
                                subsample_tiles_for_testing = False,
                                dir_im_pred='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/12.5cm Aerial Photo/',
                                dir_mask_eval='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_main_annotation/',
                                mask_suffix='_lc_2022_main_mask.tif',
                                parent_dir_tile_mainpred = '/home/tplas/predictions/predictions_LCU_2023-01-23-2018_dissolved1000m2_padding44_FGH-override/',
                                tile_outlines_shp_path = '../content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp'
                                ):
    lca.check_torch_ready(check_gpu=True, assert_versions=True)

    ##Parameters:
    if datapath_model is None:
        # datapath_model = 'LCU_2023-01-23-2018.data'
        datapath_model = 'LCU_2023-02-16-2258.data'  

    ## Load model:
    LCU = lcm.load_model(filename=datapath_model)
    LCU.eval() 
    if dissolve_small_pols:
        dissolved_name = '_dissolved' + str(dissolve_threshold) + 'm2'
    else:
        dissolved_name = ''
    if clip_to_main_class:
        dissolved_name = dissolved_name + f'_clipped{main_class_clip_label}'
    identifier = 'predictions_' + LCU.model_name + dissolved_name + f'_padding{padding}'
    save_folder = os.path.join(parent_save_folder, identifier)

    ## Predict full tiles of test set:
    tmp_results = lcm.tile_prediction_wrapper(model=LCU, save_shp=save_shp_prediction,
                                dir_im=dir_im_pred,
                                dir_mask_eval=dir_mask_eval,
                                save_folder=save_folder, dissolve_small_pols=dissolve_small_pols, 
                                area_threshold=dissolve_threshold, skip_factor=skip_factor,
                                padding=padding, mask_suffix=mask_suffix,
                                clip_to_main_class=clip_to_main_class, main_class_clip_label=main_class_clip_label, 
                                parent_dir_tile_mainpred=parent_dir_tile_mainpred, tile_outlines_shp_path=tile_outlines_shp_path,
                                subsample_tiles_for_testing=subsample_tiles_for_testing)

    ## Save results as pickle:
    with open(os.path.join(save_folder, 'summary_results.pkl'), 'wb') as f:
        pickle.dump(tmp_results, f)
    print('\nResults saved!\n\n')

    ## Merge all tiles into one shapefile:
    lca.merge_individual_shp_files(dir_indiv_tile_shp=save_folder, filename=f'merged_tiles_{identifier}.shp')

    ## Override predictions with manual FGH layer:
    if override_with_fgh_layer:
        print('######\n\nOverride predictions with manual FGH layer\n\n######')
        save_folder = lca.override_predictions_with_manual_layer(filepath_manual_layer='/home/tplas/data/gis/tmp_fgh_layer/tmp_fgh_layer.shp', 
                                                                tile_predictions_folder=save_folder, 
                                                                new_tile_predictions_override_folder=None, verbose=1)

        ## Merge all FGH_override tiles into one shapefile:
        lca.merge_individual_shp_files(dir_indiv_tile_shp=save_folder)

if __name__ == '__main__':
    predict_segmentation_network(datapath_model='LCU_2023-03-22-0150.data', 
                                clip_to_main_class=True, 
                                main_class_clip_label='D',
                                dissolve_small_pols=True,
                                dissolve_threshold=20, 
                                dir_mask_eval=None,
                                override_with_fgh_layer=False,
                                parent_dir_tile_mainpred = '/home/tplas/predictions/predictions_LCU_2023-03-22-0407_dissolved20m2_padding44/',
                                subsample_tiles_for_testing=False)