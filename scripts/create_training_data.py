## Create training data 

import sys
from tqdm import tqdm
import loadpaths
import land_cover_analysis as lca

path_dict = loadpaths.loadpaths()

## Tile paths:
# path_image_tile_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/tiles/'
# path_tile_outline_shp = '/home/tplas/repos/cnn-land-cover/content/CDE_training_tiles/CDE_training_tiles.shp'
# save_dir_mask_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/tiles/tile_masks_nfi/'
path_image_tile_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/forest_tiles_2/117915_20230216/12.5cm Aerial Photo/'
path_tile_outline_shp = '/home/tplas/repos/cnn-land-cover/content/forest_tiles/forest_tiles_2/forest_tiles_2.shp'
save_dir_mask_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/forest_tiles_2/tiles/tile_masks_nfi/'

## Patch paths:
# dir_im_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/images/'  # where to save patches 
# dir_mask_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/masks_nfi/'
dir_im_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/forest_tiles_2/images/'  # where to save patches 
dir_mask_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/forest_tiles_2/masks_nfi/'

## Set parameters:
extract_main_categories_only = False
create_patches = True
save_im_patches = True
tif_ims_in_subdirs = True
assert extract_main_categories_only == False, 'deprecated functionality'

# ## For 80s data:
# suffix_name = '_lc_80s_mask'
# col_name_low_level_index = 'LC_N_80'
# col_name_low_level_name = 'LC_D_80'
# path_lc = path_dict['lc_80s_path']
# df_lc_80s, _ = lca.load_landcover(pol_path=path_lc)

## For NFI data:
suffix_name = '_lc_nfi_mask'
col_name_low_level_index = 'Class_lowi'
col_name_low_level_name = 'Class_low'
path_lc = '/home/tplas/repos/cnn-land-cover/content/NFI_data/NFI_pd.shp'
df_lc_80s = lca.load_pols(pol_path=path_lc)

## Load landcover polygons:
df_lc_80s = lca.test_validity_geometry_column(df=df_lc_80s)
if extract_main_categories_only:
    df_lc_80s = lca.add_main_category_column(df_lc=df_lc_80s, col_ind_name=col_name_low_level_index) 
print('\nLoaded landcover polygons:\n')
print(df_lc_80s.head())

## Load shp files of tiles and intersect with PD LC:
print('\nCreating and exporting tif masks:')
df_tiles_sample = lca.load_pols(path_tile_outline_shp)
dict_intersect_pols_tiles_sample = lca.get_pols_for_tiles(df_pols=df_lc_80s, df_tiles=df_tiles_sample, 
                                                          col_name='PLAN_NO',
                                                          extract_main_categories_only=extract_main_categories_only,
                                                           col_ind_name=col_name_low_level_index,
                                                           col_class_name=col_name_low_level_name)
# df_tiles_sample_lc = pd.concat(list(dict_intersect_pols_tiles_sample.values())).reset_index(drop=True)

## Convert all polygons labels to raster and save:
for key_tile, df_tile in tqdm(dict_intersect_pols_tiles_sample.items()):
    if extract_main_categories_only:
        dict_intersect_pols_tiles_sample[key_tile] = lca.add_main_category_index_column(df_tile)
        col_name = 'class_ind'
    else:
        col_name = col_name_low_level_index
    ex_raster = lca.convert_shp_mask_to_raster(df_shp=df_tile, filename=key_tile + suffix_name, 
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
    if tif_ims_in_subdirs:
        list_tiff_files = lca.get_all_tifs_from_subdirs(dirpath=path_image_tile_tifs)
    else:
        list_tiff_files = lca.get_all_tifs_from_dir(dirpath=path_image_tile_tifs)
    list_mask_files = lca.get_all_tifs_from_dir(dirpath=save_dir_mask_tifs)

    print(f'Found {len(list_tiff_files)} images and {len(list_mask_files)} masks')
    print(list_tiff_files[:4])

    lca.create_and_save_patches_from_tiffs(list_tiff_files=list_tiff_files, list_mask_files=list_mask_files,
                                        dir_im_patches=dir_im_save_patches, dir_mask_patches=dir_mask_save_patches,
                                        mask_fn_suffix=suffix_name + '.tif',
                                        save_files=True, save_im=save_im_patches) 

print('Done')