## Create training data 

import sys
sys.path.append('../scripts/')
# import numpy as np
from tqdm import tqdm
# import pandas as pd
# import geopandas as gpd
import loadpaths
import land_cover_analysis as lca
# import land_cover_models as lcm

path_dict = loadpaths.loadpaths()

## Tile paths:
path_image_tile_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/tiles/'
path_tile_outline_shp = '/home/tplas/repos/cnn-land-cover/content/CDE_training_tiles/CDE_training_tiles.shp'
save_dir_mask_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/tiles/tile_masks/'

## Patch paths:
dir_im_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/images/'  # where to save patches 
dir_mask_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/masks/'
# dir_im_save_patches = '/home/tplas/data/gis/tmp_trial/original/images/'
# dir_mask_patches = '/home/tplas/data/gis/tmp_trial/original/masks/'

## Set parameters:
extract_main_categories_only = False
create_patches = True
assert extract_main_categories_only == False 

## Load landcover polygons:
df_lc_80s, mapping_class_inds = lca.load_landcover(pol_path=path_dict['lc_80s_path'])
# df_lc_80s = lca.load_pols(pol_path='/home/tplas/repos/cnn-land-cover/content/evaluation_polygons/Landscape_Character_80s_2022.shp')
df_lc_80s = lca.test_validity_geometry_column(df=df_lc_80s)
df_lc_80s = lca.add_main_category_column(df_lc=df_lc_80s) 
print('\nLoaded landcover polygons:\n')
print(df_lc_80s.head())

## Load shp files of tiles and intersect with PD LC:
print('\nCreating and exporting tif masks:')
df_tiles_sample = lca.load_pols(path_tile_outline_shp)
dict_intersect_pols_tiles_sample = lca.get_pols_for_tiles(df_pols=df_lc_80s, df_tiles=df_tiles_sample, col_name='PLAN_NO',
                                                          extract_main_categories_only=extract_main_categories_only)
# df_tiles_sample_lc = pd.concat(list(dict_intersect_pols_tiles_sample.values())).reset_index(drop=True)

## Convert all polygons labels to raster and save:
for key_tile, df_tile in tqdm(dict_intersect_pols_tiles_sample.items()):
    if extract_main_categories_only:
        dict_intersect_pols_tiles_sample[key_tile] = lca.add_main_category_index_column(df_tile)
        col_name = 'class_ind'
    else:
        col_name = 'LC_N_80'
    ex_raster = lca.convert_shp_mask_to_raster(df_shp=df_tile, filename=key_tile + '_lc_80s_mask', 
                                maskdir=save_dir_mask_tifs, 
                                col_name=col_name,
                                # ex_tile=ex_raster,
                                # resolution=(-0.125, 0.125),
                                plot_raster=False, # whether to plot
                                save_raster=True, # whether to store on disk
                                verbose=0)

    assert ex_raster[col_name].shape == (8000, 8000), key_tile

print('\nCreating and exporting patches:')
if create_patches:
    list_tiff_files = lca.get_all_tifs_from_dir(dirpath=path_image_tile_tifs)
    list_mask_files = lca.get_all_tifs_from_dir(dirpath=save_dir_mask_tifs)

    print(f'Found {len(list_tiff_files)} images and {len(list_mask_files)} masks')
    print(list_tiff_files[:4])

    lca.create_and_save_patches_from_tiffs(list_tiff_files=list_tiff_files, list_mask_files=list_mask_files,
                                        dir_im_patches=dir_im_save_patches, dir_mask_patches=dir_mask_save_patches,
                                        mask_fn_suffix='_lc_80s_mask.tif',
                                        save_files=True) 

print('Done')