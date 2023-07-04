## Perform prediction on full tiles using a trained model on a test set

import os, sys, json, pickle
import datetime
import loadpaths
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import land_cover_models as lcm

def predict_segmentation_network(datapath_model=None, 
                                 padding=44, 
                                 dissolve_small_pols=True,
                                dissolve_threshold=1000,  # only used if dissolve_small_pols=True AND use_class_dependent_area_thresholds=False
                                use_class_dependent_area_thresholds=False,
                                file_path_class_dependent_area_thresholds=None,
                                clip_to_main_class=False,
                                col_name_class=None,  # name of column in main predictions shapefile that contains the class label (if None, found automatically if only one candidate column exists)
                                main_class_clip_label='D',
                                skip_factor=16,
                                save_shp_prediction=True,
                                parent_save_folder='/home/tplas/predictions/',
                                override_with_fgh_layer=False,
                                subsample_tiles_for_testing=False,
                                dir_im_pred='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/12.5cm Aerial Photo/',
                                dir_mask_eval=None,
                                # dir_mask_eval='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_main_annotation/',
                                mask_suffix='_lc_2022_main_mask.tif',
                                parent_dir_tile_mainpred='/home/tplas/predictions/predictions_LCU_2023-01-23-2018_dissolved1000m2_padding44_FGH-override/',
                                tile_outlines_shp_path='../content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp',  # used BOTH for selecting tiles to predict AND for clipping predictions to tile outlines
                                use_tile_outlines_shp_to_predict_those_tiles_only=False,
                                delete_individual_tile_predictions=False,
                                merge_tiles_into_one_shp=True,
                                ):
    lca.check_torch_ready(check_gpu=True, assert_versions=True)

    ##Parameters:
    if datapath_model is None:
        # datapath_model = 'LCU_2023-01-23-2018.data'
        datapath_model = 'LCU_2023-02-16-2258.data'  

    ## Load model:
    LCU = lcm.load_model(filename=datapath_model)
    LCU.eval() 

    if use_class_dependent_area_thresholds:
        dissolve_threshold = None  # reset if using dict of class-dependent thresholds
        assert file_path_class_dependent_area_thresholds is not None, 'If use_class_dependent_area_thresholds is True, file_path_class_dependent_area_thresholds must be provided'
        class_dependent_area_thresholds, dissolve_threshold = lca.load_area_threshold_json(file_path_class_dependent_area_thresholds)
        if dissolve_threshold is None:  # not defined in json file:
            dissolve_threshold = 0
            print('WARNING: dissolve_threshold not defined in json file, using 0 instead')
        name_combi = file_path_class_dependent_area_thresholds.split('/')[-1].split('.')[0]
        print(f'Using class-dependent area thresholds from {name_combi}')
    else:
        name_combi = None
        class_dependent_area_thresholds = None

    if dissolve_small_pols and (not use_class_dependent_area_thresholds):
        dissolved_name = '_dissolved' + str(dissolve_threshold) + 'm2'
    elif dissolve_small_pols and use_class_dependent_area_thresholds:
        dissolved_name = f'_dissolved-{name_combi}'
    else:
        dissolved_name = '_notdissolved'
    if clip_to_main_class:
        dissolved_name = dissolved_name + f'_clipped{main_class_clip_label}'
    identifier = 'predictions_' + LCU.model_name + dissolved_name + f'_padding{padding}'
    save_folder = os.path.join(parent_save_folder, identifier)

    if use_tile_outlines_shp_to_predict_those_tiles_only:
        df_tile_outlines = lca.load_pols(tile_outlines_shp_path)
        tile_name_col = 'PLAN_NO'
        list_tile_names_to_predict = df_tile_outlines[tile_name_col].unique().tolist()
        print(f'Predicting only the following number of tiles: {len(list_tile_names_to_predict)}')
    else:
        list_tile_names_to_predict = None

    ## Predict full tiles of test set:
    tmp_results = lcm.tile_prediction_wrapper(model=LCU, save_shp=save_shp_prediction,
                                dir_im=dir_im_pred, list_tile_names_to_predict=list_tile_names_to_predict,
                                dir_mask_eval=dir_mask_eval,
                                save_folder=save_folder, dissolve_small_pols=dissolve_small_pols, 
                                area_threshold=dissolve_threshold, 
                                use_class_dependent_area_thresholds=use_class_dependent_area_thresholds,
                                class_dependent_area_thresholds=class_dependent_area_thresholds,
                                name_combi_area_thresholds=name_combi,
                                skip_factor=skip_factor,
                                padding=padding, mask_suffix=mask_suffix,
                                clip_to_main_class=clip_to_main_class, main_class_clip_label=main_class_clip_label, 
                                col_name_class=col_name_class,
                                parent_dir_tile_mainpred=parent_dir_tile_mainpred, tile_outlines_shp_path=tile_outlines_shp_path,
                                subsample_tiles_for_testing=subsample_tiles_for_testing)

    ## Save results as pickle:
    with open(os.path.join(save_folder, 'summary_results.pkl'), 'wb') as f:
        pickle.dump(tmp_results, f)
    print('\nResults saved!\n\n')

    if merge_tiles_into_one_shp:
        ## Merge all tiles into one shapefile:
        lca.merge_individual_shp_files(dir_indiv_tile_shp=save_folder,
                                        delete_individual_shp_files=False)  # set to False because they are needed for FGH override

    ## Override predictions with manual FGH layer:
    if override_with_fgh_layer:
        assert clip_to_main_class is False, 'Expected that FGH override would only happen on main class predictions, but clip_to_main_class is set to True which indicates that these are detailed class predictions'
        print('######\n\nOverride predictions with manual FGH layer\n\n######')
        save_folder_fgh = lca.override_predictions_with_manual_layer(filepath_manual_layer='/home/tplas/data/gis/tmp_fgh_layer/tmp_fgh_layer.shp', 
                                                                tile_predictions_folder=save_folder, 
                                                                new_tile_predictions_override_folder=None, verbose=1)

        if merge_tiles_into_one_shp:
            ## Merge all FGH_override tiles into one shapefile:
            lca.merge_individual_shp_files(dir_indiv_tile_shp=save_folder_fgh, 
                                        delete_individual_shp_files=delete_individual_tile_predictions)

if __name__ == '__main__':

    dict_cnns_best = {  ## determined in `Evaluate trained network.ipynb`
        'main': 'LCU_2023-04-24-1259.data',
        'C': 'LCU_2023-04-21-1335.data',
        'D': 'LCU_2023-04-25-2057.data',
        'E': 'LCU_2023-04-24-1216.data'
    }
    model_use = 'D'
    folder_area_thresholds = '/home/tplas/repos/cnn-land-cover/content/area_threshold_combinations/'
    list_combis = [os.path.join(folder_area_thresholds, x) for x in os.listdir(folder_area_thresholds) if x.endswith('.json')]
    list_combis = list_combis[:4]
    
    for model_use in ['C', 'D', 'E']:
    # for model_use in ['main']:
    # for file_path_class_dependent_area_thresholds in list_combis:
    #     if 'th-combi-1.json' in file_path_class_dependent_area_thresholds:
    #         pass 
    #     else:
    #         print("MOVING ON")
    #         continue
        predict_segmentation_network(datapath_model=dict_cnns_best[model_use], 
                                    clip_to_main_class=False if model_use == 'main' else True, 
                                    col_name_class='lc_label',
                                    main_class_clip_label=model_use, # dict_cnns_clip_to_main_class[model_use],
                                    dissolve_small_pols=False,
                                    dissolve_threshold=10, 
                                    use_class_dependent_area_thresholds=False,
                                    # file_path_class_dependent_area_thresholds=file_path_class_dependent_area_thresholds,
                                    dir_mask_eval=None,
                                    override_with_fgh_layer=True if model_use == 'main' else False,
                                    dir_im_pred='/media/data-hdd/gis_pd/all_pd_tiles/',
                                    # parent_dir_tile_mainpred = '/home/tplas/predictions/predictions_LCU_2023-04-24-1259_notdissolved_padding44_FGH-override/',
                                    parent_dir_tile_mainpred='/media/data-hdd/gis_pd/predictions/all_tiles_pd_notdissolved/predictions_LCU_2023-04-24-1259_notdissolved_padding44_FGH-override/individual_tiles/',
                                    subsample_tiles_for_testing=True,
                                    # tile_outlines_shp_path = '../content/rush_tiles/rush_primaryhabitat_tiles.shp',
                                    # tile_outlines_shp_path='../content/evaluation_sample_50tiles/eval_all_tile_outlines/eval_all_tile_outlines.shp',
                                    tile_outlines_shp_path='../content/landscape_character_grid/Landscape_Character_Grid_tight.shp',
                                    use_tile_outlines_shp_to_predict_those_tiles_only=False,
                                    delete_individual_tile_predictions=False,         
                                    # parent_save_folder = '/home/tplas/predictions/testing_grounds/',
                                    parent_save_folder='/media/data-hdd/gis_pd/predictions/all_tiles_pd_notdissolved/',
                                    merge_tiles_into_one_shp=False                      
                                    )

