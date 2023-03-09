## Create image + mask patches from polygons, given a directory of tif images 
## This is a more general version of create_training_data.py

import sys, os, datetime
from tqdm import tqdm
import loadpaths
import land_cover_analysis as lca
# import shapely.geometry

path_dict = loadpaths.loadpaths()

def main():
    ## Paths:   
    path_image_tile_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/12.5cm Aerial Photo/'
    path_tile_outline_shp = '/home/tplas/repos/cnn-land-cover/content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp'
    save_dir_mask_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_detailed_annotation/'
    path_lc = '/home/tplas/repos/cnn-land-cover/content/evaluation_polygons/landscape_character_2022_detailed_CFGH-override/landscape_character_2022_detailed_CFGH-override.shp'

    ## Patch paths:
    dir_im_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images/'  # where to save patches 
    dir_mask_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_detailed_annotation/'

    ## Set parameters:
    create_patches = True
    save_im_patches = False
    tif_ims_in_subdirs = True  # True if tif images are in subdirectories of path_image_tile_tifs
    create_metadata_patches = True
    discard_empty_patches = False  # whether to discard patches that do not contain any landcover class (ie only NO CLASS)

    suffix_name = '_lc_2022_detailed_mask'
    col_name_low_level_index = None
    col_name_low_level_name = 'Class_low'

    ## Create directories:
    if not os.path.exists(save_dir_mask_tifs):
        os.makedirs(save_dir_mask_tifs)
    else:
        print(f'Warning: directory {save_dir_mask_tifs} already exists! Overwriting files in this directory!')
    if not os.path.exists(dir_im_save_patches) and save_im_patches:
        os.makedirs(dir_im_save_patches)
    else:
        print(f'Warning: directory {dir_im_save_patches} already exists! Overwriting files in this directory!')
    if not os.path.exists(dir_mask_save_patches):
        os.makedirs(dir_mask_save_patches)
    else:
        print(f'Warning: directory {dir_mask_save_patches} already exists! Overwriting files in this directory!')

    ## Load landcover polygons:
    df_lc = lca.load_pols(pol_path=path_lc)
    df_lc = lca.test_validity_geometry_column(df=df_lc)
    # list_extra_cols = [col_name_low_level_index, col_name_low_level_name]
    print('\nLoaded landcover polygons:\n')
    
    df_lc, col_name_low_level_index = lca.add_detailed_index_column(df_lc=df_lc, col_name_low_level_index=col_name_low_level_index,
                                          col_name_low_level_name=col_name_low_level_name,
                                          exclude_non_mapped_pols=True)  # check to see if exists, else creates one
    
    ## Load shp files of tiles and intersect with PD LC:
    print('\nCreating and exporting tif masks:')
    df_tiles_sample = lca.load_pols(path_tile_outline_shp)
    dict_intersect_pols_tiles_sample, list_empty_tiles = lca.get_pols_for_tiles_general(df_pols=df_lc, df_tiles=df_tiles_sample, 
                                                            col_name='PLAN_NO', list_extra_cols=None,
                                                            fill_empty_space_with_zero=True, 
                                                            use_full_tile_as_zero_background=False)
    if len(list_empty_tiles) > 0:
        assert False, f'Warning: {len(list_empty_tiles)} tiles have no polygons in them!'
    
    ## Convert all polygons labels to raster and save:
    for key_tile, df_tile in tqdm(dict_intersect_pols_tiles_sample.items()):
        ex_raster = lca.convert_shp_mask_to_raster(df_shp=df_tile, filename=key_tile + suffix_name, 
                                    maskdir=save_dir_mask_tifs, 
                                    col_name=col_name_low_level_index,
                                    # ex_tile=ex_raster,
                                    # resolution=(-0.125, 0.125),
                                    plot_raster=False, # whether to plot
                                    save_raster=True, # whether to store on disk
                                    verbose=0)
        assert ex_raster[col_name_low_level_index].shape == (8000, 8000), key_tile

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
                                            mask_fn_suffix=suffix_name + '.tif', discard_empty_patches=discard_empty_patches,
                                            save_files=True, save_im=save_im_patches) 

        if create_metadata_patches:
            ## Create textfile with all datapaths and parameters in parent dir of dir_mask_save_patches:
            dir_text_file = os.path.dirname(dir_mask_save_patches)
            tmp = datetime.datetime.now() 
            date_time = tmp.strftime("%Y-%m-%d_%H-%M")
            with open(os.path.join(dir_text_file, f'metadata_patches_{date_time}.txt'), 'w') as f:
                f.write(f'path_image_tile_tifs: {path_image_tile_tifs}\n')
                f.write(f'path_tile_outline_shp: {path_tile_outline_shp}\n')
                f.write(f'save_dir_mask_tifs: {save_dir_mask_tifs}\n')
                f.write(f'path_lc: {path_lc}\n')
                f.write(f'col_name_low_level_index: {col_name_low_level_index}\n')
                f.write(f'col_name_low_level_name: {col_name_low_level_name}\n')
                f.write(f'dir_im_save_patches: {dir_im_save_patches}\n')
                f.write(f'dir_mask_save_patches: {dir_mask_save_patches}\n')
                f.write(f'suffix_name: {suffix_name}\n')
                f.write(f'discard_empty_patches: {discard_empty_patches}\n')
                f.write(f'save_im_patches: {save_im_patches}\n')
                f.write(f'create_metadata_patches: {create_metadata_patches}\n')
                f.write(f'create_patches: {create_patches}\n')
                f.write(f'Date: {tmp}\n')
            f.close() 

if __name__ == '__main__':
    main()
    