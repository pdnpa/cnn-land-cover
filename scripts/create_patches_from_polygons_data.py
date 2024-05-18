## Create image + mask patches from polygons, given a directory of tif images 
## This is a more general version of create_training_data.py

import sys, os, datetime
from tqdm import tqdm
# import loadpaths
import land_cover_analysis as lca

# path_dict = loadpaths.loadpaths()

def main(
            #path_image_tile_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/12.5cm Aerial Photo/',
            #path_tile_outline_shp = '/home/tplas/repos/cnn-land-cover/content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp',
            #save_dir_mask_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_detailed_annotation/',
            #path_lc = '/home/tplas/repos/cnn-land-cover/content/evaluation_polygons/landscape_character_2022_detailed_CFGH-override/landscape_character_2022_detailed_CFGH-override.shp',
            path_image_tile_tifs = "/home/david/Documents/ADP/4band_12.5cm/",
            path_tile_outline_shp = "/home/david/Documents/evaluation_sample_50tiles/eval_all_tile_outlines.shp",
            save_dir_mask_tifs = "/home/david/Documents/ADP/pd_lc_annotated_patches_data/python_format_4band/",
            path_lc = "../content/evaluation_polygons/landscape_character_2022_detailed_CFGH-override/landscape_character_2022_detailed_CFGH-override_clip.shp",
            df_lc=None,
            description_df_lc_for_metadata=None,
            #dir_im_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images_detailed_annotation/',  # where to save patches 
            #dir_mask_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_detailed_annotation/',
            dir_im_save_patches = "/home/david/Documents/ADP/pd_lc_annotated_patches_data/python_format_4band/images_python_all/",
            dir_mask_save_patches = "/home/david/Documents/ADP/pd_lc_annotated_patches_data/python_format_4band/masks_python_all/",
            create_patches = True,
            create_mask_tiles = True,
            save_im_patches = True,
            tif_ims_in_subdirs = False,  # True if tif images are in subdirectories of path_image_tile_tifs
            create_metadata_patches = True,
            discard_empty_patches = True, # whether to discard patches that do not contain any landcover class (ie only NO CLASS)
            suffix_name = '_lc_2022_detailed_mask',
            col_name_class_index = None,  # if None, will be created
            col_name_class_name = 'Class_low',
            create_high_level_masks = False,
            df_patches_selected=None, 
            df_sel_tile_patch_name_col='tile_patch'
        ):
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
    if df_lc is None:
        df_lc = lca.load_pols(pol_path=path_lc)
        lc_provided = False
    else:
        lc_provided = True
        print('Using provided df_lc')
    df_lc = lca.test_validity_geometry_column(df=df_lc)
    print('\nLoaded landcover polygons:\n')
    
    if create_mask_tiles:
        ## LC class names have to be changed to indices (eg C1 -> 1). Check to see if exists, else create one
        if create_high_level_masks:
            assert False, 'deprecated -- instead, create detailed masks and use dict_mapping to convert to high level masks later'
            df_lc, col_name_class_index = lca.add_main_category_index_column(df_lc=df_lc, col_ind_name=col_name_class_index,
                                            col_code_name=col_name_class_name)  
        else:
            df_lc, col_name_class_index = lca.add_detailed_index_column(df_lc=df_lc, col_name_low_level_index=col_name_class_index,
                                            col_name_low_level_name=col_name_class_name,
                                            exclude_non_mapped_pols=True)  
        
        ## Load shp files of tiles and intersect with PD LC:
        print('\nCreating and exporting tif masks:')
        df_tiles_sample = lca.load_pols(path_tile_outline_shp)
        dict_intersect_pols_tiles_sample, list_empty_tiles = lca.get_pols_for_tiles_general(df_pols=df_lc, df_tiles=df_tiles_sample, 
                                                                col_name='PLAN_NO', list_extra_cols=None,
                                                                fill_empty_space_with_zero=True, 
                                                                use_full_tile_as_zero_background=False)
        if len(list_empty_tiles) > 0:
            ## Remove empty tiles from dict_intersect_pols_tiles_sample:
            for key in list_empty_tiles:
                dict_intersect_pols_tiles_sample.pop(key, None)

        ## Convert all polygons labels to raster and save:
        for key_tile, df_tile in tqdm(dict_intersect_pols_tiles_sample.items()):
            ex_raster = lca.convert_shp_mask_to_raster(df_shp=df_tile, filename=key_tile + suffix_name, 
                                        maskdir=save_dir_mask_tifs, 
                                        col_name=col_name_class_index,
                                        # ex_tile=ex_raster,
                                        # resolution=(-0.125, 0.125),
                                        plot_raster=False, # whether to plot
                                        save_raster=True, # whether to store on disk
                                        verbose=0)
            assert ex_raster[col_name_class_index].shape == (8000, 8000), key_tile  

    if create_patches:
        print('\nCreating and exporting patches:')
        if tif_ims_in_subdirs:
            list_tiff_files = lca.get_all_tifs_from_subdirs(dirpath=path_image_tile_tifs)
        else:
            list_tiff_files = lca.get_all_tifs_from_dir(dirpath=path_image_tile_tifs)
        list_mask_files = lca.get_all_tifs_from_dir(dirpath=save_dir_mask_tifs)

        print(f'Found {len(list_tiff_files)} images and {len(list_mask_files)} masks')
        print(list_tiff_files[:4])
        # return list_tiff_files, list_mask_files
        lca.create_and_save_patches_from_tiffs(list_tiff_files=list_tiff_files, list_mask_files=list_mask_files,
                                            dir_im_patches=dir_im_save_patches, dir_mask_patches=dir_mask_save_patches,
                                            mask_fn_suffix=suffix_name + '.tif', discard_empty_patches=discard_empty_patches,
                                            df_patches_selected=df_patches_selected, df_sel_tile_patch_name_col=df_sel_tile_patch_name_col,
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
                f.write(f'df_lc provided: {lc_provided}\n')
                if lc_provided:
                    f.write(f'description of df_lc: {description_df_lc_for_metadata}\n')
                else:
                    f.write(f'path_lc: {path_lc}\n')
                f.write(f'col_name_class_index: {col_name_class_index}\n')
                f.write(f'col_name_class_name: {col_name_class_name}\n')
                f.write(f'create_high_level_masks: {create_high_level_masks}\n')
                f.write(f'dir_im_save_patches: {dir_im_save_patches}\n')
                f.write(f'dir_mask_save_patches: {dir_mask_save_patches}\n')
                f.write(f'suffix_name: {suffix_name}\n')
                f.write(f'discard_empty_patches: {discard_empty_patches}\n')
                f.write(f'save_im_patches: {save_im_patches}\n')
                f.write(f'create_metadata_patches: {create_metadata_patches}\n')
                f.write(f'create_patches: {create_patches}\n')
                f.write(f'create_mask_tiles: {create_mask_tiles}\n')
                f.write(f'Date: {tmp}\n')
            f.close() 

if __name__ == '__main__':
    # main()
    # df_hab = lca.load_pols('../content/habitat_data_annotations/habitat_data_annotations.shp')
    # df_hab = df_hab[df_hab['SEL_TRAIN'] == 1]
    
    main(
        path_image_tile_tifs = "/home/david/Documents/ADP/4band_12.5cm/",
        path_tile_outline_shp = "/home/david/Documents/evaluation_sample_50tiles/eval_all_tile_outlines.shp",
        save_dir_mask_tifs = "/home/david/Documents/ADP/pd_lc_annotated_patches_data/python_format_4band/",
        path_lc = "../content/evaluation_polygons/landscape_character_2022_detailed_CFGH-override/landscape_character_2022_detailed_CFGH-override_clip.shp",
        dir_im_save_patches = "/home/david/Documents/ADP/pd_lc_annotated_patches_data/python_format_4band/images_python_all/",
        dir_mask_save_patches = "/home/david/Documents/ADP/pd_lc_annotated_patches_data/python_format_4band/masks_python_all/",
        create_patches = True,
        create_mask_tiles = True,
        save_im_patches = True,
        tif_ims_in_subdirs = False,
        create_metadata_patches = True,
        discard_empty_patches = True,
        suffix_name = "_lc_2022_detailed_mask",
        col_name_class_index = None,
        col_name_class_name = "Class_low",
        create_high_level_masks = False,
        df_patches_selected=None, 
        df_sel_tile_patch_name_col='tile_patch'
        '''
        path_image_tile_tifs = '/media/data-hdd/gis_pd/all_pd_tiles/',
        path_tile_outline_shp = '/home/tplas/repos/cnn-land-cover/content/evaluation_sample_50tiles/eval_2_30tiles_outlines.shp',
        save_dir_mask_tifs = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/eval_2_tiles/tile_masks_detailed_annotation/',
        path_lc = '/home/tplas/repos/cnn-land-cover/content/evaluation_polygons/landscape_character_2022_detailed_CFGH-override/landscape_character_2022_detailed_CFGH-override.shp',
        df_lc=None,
        description_df_lc_for_metadata=None,
        dir_im_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/eval_2_tiles/images_detailed_annotation/',  # where to save patches 
        dir_mask_save_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/eval_2_tiles/masks_detailed_annotation/',
        create_patches = True,
        create_mask_tiles = True,
        save_im_patches = True,
        tif_ims_in_subdirs = True,  # True if tif images are in subdirectories of path_image_tile_tifs
        create_metadata_patches = True,
        discard_empty_patches = True, # whether to discard patches that do not contain any landcover class (ie only NO CLASS)
        suffix_name = '_lc_2022_detailed_mask',
        col_name_class_index = None,  # if None, will be created
        col_name_class_name = 'Class_low',
        create_high_level_masks = False,
        '''
    )