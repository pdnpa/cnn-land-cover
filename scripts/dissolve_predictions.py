## Dissolving predictions, per tile and then merging. 

import os, sys, json
# import datetime
# import loadpaths
import land_cover_analysis as lca
# import land_cover_visualisation as lcv
# import land_cover_models as lcm
# import geopandas as gpd
from tqdm import tqdm 

def dissolve_predicted_tiles(
        path_to_predicted_tiles = '/media/data-hdd/gis_pd/predictions/all_tiles_pd_notdissolved/predictions_LCU_detailed-combined/individual_tiles/',
        path_to_dissolved_tiles = '/media/data-hdd/gis_pd/predictions/all_tiles_pd_dissolved/predictions_LCU_detailed-combined/individual_tiles/',
        dissolve_threshold=1000,  # only used if dissolve_small_pols=True AND use_class_dependent_area_thresholds=False
        use_class_dependent_area_thresholds=False,
        file_path_class_dependent_area_thresholds=None,
        label_col='lc_label',
        exclude_no_class_from_large_pols=True,
        test_run=False,
):
    assert exclude_no_class_from_large_pols
    if file_path_class_dependent_area_thresholds is not None and use_class_dependent_area_thresholds is False:
        print('WARNING: file_path_class_dependent_area_thresholds is provided but use_class_dependent_area_thresholds is False, so the file_path_class_dependent_area_thresholds will not be used')

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

    df_mapping = lca.create_df_mapping_labels_2022_to_80s()
    mapping_dict_label_to_class = dict(zip(df_mapping['code_2022'], df_mapping['index_2022']))
    class_col = 'class'

    ## List all individual tiles 
    list_tiles = [x for x in os.listdir(path_to_predicted_tiles) if os.path.isdir(os.path.join(path_to_predicted_tiles, x))]
    print(f'Found {len(list_tiles)} tiles')
    if test_run:
        list_tiles = list_tiles[:5]
        print(f'Running test run with {len(list_tiles)} tiles')
        dict_test_run = {}

    ## Loop through all tiles and dissolve
    with tqdm(enumerate(list_tiles)) as tbar:
        for i_tile, tile_folder in tbar:
            tbar.set_description('Dissolving tiles, iteration %i' % i_tile)
            ## Load tile:
            full_path_tile_folder = os.path.join(path_to_predicted_tiles, tile_folder)
            shp_files = [x for x in os.listdir(full_path_tile_folder) if x.endswith('.shp')]
            assert len(shp_files) == 1, f'Found {len(shp_files)} shp files in {full_path_tile_folder}, expected 1'
            name_file = shp_files[0]
            assert name_file.split('.')[0] == tile_folder, f'Name of shp file {name_file} does not match name of folder {tile_folder}'
            path_file = os.path.join(full_path_tile_folder, name_file)
            gdf = lca.load_pols(path_file)
            assert label_col in gdf.columns, f'Column {label_col} not found in {path_file}'
            unique_classes = gdf[label_col].unique()
            for unique_class in unique_classes:
                assert unique_class in mapping_dict_label_to_class.keys(), f'Class {unique_class} not found in mapping_dict_label_to_class'
            gdf[class_col] = gdf[label_col].map(mapping_dict_label_to_class)

            ## Dissolve:
            gdf_dissolved = lca.filter_small_polygons_from_gdf(gdf=gdf, class_col=class_col, label_col=label_col,
                                            area_threshold=dissolve_threshold, 
                                            use_class_dependent_area_thresholds=use_class_dependent_area_thresholds,
                                            class_dependent_area_thresholds=class_dependent_area_thresholds,
                                            verbose=0, exclude_no_class_from_large_pols=exclude_no_class_from_large_pols)  # if clip is True, then you don't want to exclude no class from large pols because everything that was clipped will be no class
            name_tile_dissolved = tile_folder + '_dissolved'
            save_folder = os.path.join(path_to_dissolved_tiles, name_tile_dissolved)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, name_tile_dissolved + '.shp')
            gdf_dissolved.to_file(save_path)
            
            if test_run:  
                dict_test_run[name_tile_dissolved] = (gdf, gdf_dissolved)
                return gdf, gdf_dissolved
    if test_run:
        return dict_test_run
    else:
        return None

def save_area_threshold_dict(dict_customise={}, default=1000, name_combi='th-combi-unspecified'):
    assert type(dict_customise) == dict, 'dict_customise must be a dict'
    assert default >= 0 
    assert type(name_combi) == str

    dict_area_thresholds = {}
    for k, v in dict_customise.items():
        dict_area_thresholds[k] = v
    
    dict_area_thresholds['default'] = default

    save_path = f'/home/tplas/repos/cnn-land-cover/content/area_threshold_combinations/{name_combi}.json'
    with open(save_path, 'w') as f:
        json.dump(dict_area_thresholds, f)

if __name__ == "__main__":
    _ = dissolve_predicted_tiles()