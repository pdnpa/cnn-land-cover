import os, sys, copy, datetime, pickle
import numpy as np
from numpy.core.multiarray import square
from numpy.testing import print_assert_equal
import rasterio
import xarray as xr
import rioxarray as rxr
import sklearn.cluster, sklearn.model_selection
from tqdm import tqdm
import shapely as shp
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
import gdal, osr
import loadpaths
import patchify 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp

path_dict = loadpaths.loadpaths()

def create_timestamp():
    dt = datetime.datetime.now()
    timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
    return timestamp

def assert_epsg(epsg, project_epsg=27700):
    '''Check epsg against reference (project) epsg'''
    if type(epsg) != int:
        epsg = int(epsg)
    assert epsg == project_epsg, f'EPSG {epsg} is not project EPSG ({project_epsg})'

def load_tiff(tiff_file_path, datatype='np', verbose=0):
    '''Load tiff file as np or da'''
    with rasterio.open(tiff_file_path) as f:
        if verbose > 0:
            print(f.profile)
        if datatype == 'np':  # handle different file types 
            im = f.read()
            assert type(im) == np.ndarray
        elif datatype == 'da':
            im = rxr.open_rasterio(f)
            assert type(im) == xr.DataArray
        else:
            assert False, 'datatype should be np or da'

    return im 

def get_all_tifs_from_dir(dirpath):
    '''Return list of tifs from dir path'''
    assert type(dirpath) == str
    list_tifs = [os.path.join(dirpath, x) for x in os.listdir(dirpath) if x[-4:] == '.tif']
    return list_tifs 

def get_all_tifs_from_subdirs(dirpath):
    '''Get all tiffs from all subdirs of dir'''
    list_tiffs = []
    list_subdirs = [x for x in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, x))]
    for sd in list_subdirs:
        list_tiff_subdir = get_all_tifs_from_dir(os.path.join(dirpath, sd))
        list_tiffs.append(list_tiff_subdir)

    list_tiffs = sum(list_tiffs, [])
    return list_tiffs

def load_coords_from_geotiff(tiff_file_path):
    '''Create rectangular polygon based on coords of geotiff file'''
    ## from https://stackoverflow.com/questions/50191648/gis-geotiff-gdal-python-how-to-get-coordinates-from-pixel and https://gis.stackexchange.com/questions/126467/determining-if-shapefile-and-raster-overlap-in-python-using-ogr-gdal
    raster = gdal.Open(tiff_file_path)
    raster_epsg = int(osr.SpatialReference(raster.GetProjection()).GetAttrValue('AUTHORITY', 1))
    assert_epsg(raster_epsg)
    x_left, px_width, x_rot, y_top, y_rot, px_height = raster.GetGeoTransform()
    # print(raster.GetGeoTransform())
    n_pix_x = raster.RasterXSize
    n_pix_y = raster.RasterYSize
    x_right = x_left + n_pix_x * px_width
    y_bottom = y_top + n_pix_y * px_height

     # # print(x_left, x_right, y_bottom, y_top)
    # # supposing x and y are your pixel coordinate this is how to get the coordinate in space:
    # pix_x = 1000
    # pix_y= 4000
    # posX = px_width * pix_x + x_rot * pix_y + x_left
    # posY = y_rot * pix_x + px_height * pix_y + y_top
        # # shift to the center of the pixel
    # posX += px_width / 2.0
    # posY += px_height / 2.0
    # # print(posX, posY)
    
    square_coords = shp.geometry.Polygon(zip([x_left, x_right, x_right, x_left], [y_bottom, y_bottom, y_top, y_top]))
    name_tile = tiff_file_path.split('/')[-1].rstrip('.tif')
    return square_coords, name_tile, raster_epsg

def create_df_with_tiff_coords(tiff_paths, verbose=0):
    '''Create df with image tile polygons of all tifs in tiff_paths list'''
    if type(tiff_paths) == str:  # if only one path given as str, turn into list for compatibility
        tiff_paths = [tiff_paths] 

    if verbose > 0:
        print(f'Loading {len(tiff_paths)} tiff files')

    list_coords, list_names = [], []
    prev_epsg = None
    for i_tile, tile_path in enumerate(tiff_paths):
        if verbose > 1:
            print(i_tile, tile_path)
        square_coords, name_tile, raster_epsg = load_coords_from_geotiff(tiff_file_path=tile_path)
        list_coords.append(square_coords)
        list_names.append(name_tile)

        if i_tile > 0:  # ensure epsg stays consistent 
            assert raster_epsg == prev_epsg
        prev_epsg = raster_epsg

    df_tile = gpd.GeoDataFrame(pd.DataFrame({'name': list_names}),
                                             crs=f'epsg:{raster_epsg}', geometry=list_coords)
    
    return df_tile

# def load_geo_tiff(tiff_file_path, datatype='np', verbose=0):
#     ## image
#     im = load_tiff(tiff_file_path=tiff_file_path, datatype=datatype, verbose=verbose)

#     # coords 
#     df_tile = load_coords_from_geotiff(tiff_file_path=tiff_file_path)

#     return im, df_tile

def load_pols(pol_path):
    '''Load shape file'''
    df_pols = gpd.read_file(pol_path)
    assert_epsg(df_pols.crs.to_epsg())
    return df_pols 

def load_landcover(pol_path, col_class_ind='LC_N_80', col_class_names='LC_D_80'):
    '''Load shp file of land cover 80s and 70s (expect that format and attributes)'''
    df_lc = load_pols(pol_path)
    assert col_class_ind in df_lc.columns and col_class_names in df_lc.columns
    inds_classes = np.unique(df_lc[col_class_ind])  # inds corresponding to classes
    dict_classes = {}
    for ind in inds_classes:
        lc_class = np.unique(df_lc[df_lc[col_class_ind] == ind][col_class_names])
        assert len(lc_class) == 1, 'class inds do not uniquely correspond to class names'
        lc_class = lc_class[0]
        assert type(lc_class) == str
        dict_classes[ind] = lc_class  # create mapping 
        
    return df_lc, dict_classes  

def get_lc_mapping_inds_names_dicts(pol_path=path_dict['lc_80s_path'], 
                                    col_class_ind='LC_N_80', col_class_names='LC_D_80'):
    '''Get mapping between LC class inds and names'''
    _, dict_ind_to_name = load_landcover(pol_path=pol_path, col_class_ind=col_class_ind, 
                                         col_class_names=col_class_names)
    dict_ind_to_name[0] = 'NO CLASS'
    dict_name_to_ind = {v: k for k, v in dict_ind_to_name.items()}

    dict_ind_to_name[40] = 'Wood and Forest Land'
    dict_ind_to_name[41] = 'Moor and Heath Land'
    dict_ind_to_name[42] = 'Agro-Pastoral Land'
    dict_ind_to_name[43] = 'Water and Wetland'
    dict_ind_to_name[44] = 'Rock and Coastal Land'
    dict_ind_to_name[45] = 'Developed Land'

    return dict_ind_to_name, dict_name_to_ind

def get_pols_for_tiles(df_pols, df_tiles, col_name='name'):
    '''Extract polygons that are inside a tile, for all tiles in df_tiles. Assuming a df for tiles currently.'''

    n_tiles = len(df_tiles)
    dict_pols = {}
    for i_tile in tqdm(range(n_tiles)):  # loop through tiles, process individually:
        tile = df_tiles.iloc[i_tile]
        pol_tile = tile['geometry']  # polygon of tile 
        name_tile = tile[col_name]

        df_relevant_pols = df_pols[df_pols.geometry.overlaps(pol_tile)]  # find polygons that overlap with tile
        list_pols = []
        list_class_id = []
        list_class_name = []
        for i_pol in range(len(df_relevant_pols)):  # loop through pols
            new_pol = df_relevant_pols.iloc[i_pol]['geometry'].intersection(pol_tile)  # create intersection between pol and tile
            list_pols.append(new_pol)
            list_class_id.append(df_relevant_pols.iloc[i_pol]['LC_N_80'])
            list_class_name.append(df_relevant_pols.iloc[i_pol]['LC_D_80'])
        dict_pols[name_tile] = gpd.GeoDataFrame(geometry=list_pols).assign(LC_N_80=list_class_id, LC_D_80=list_class_name)  # put all new intersections back into a dataframe        # df_relevant_pols

    return dict_pols

def get_area_per_class_df(gdf, col_class_name='LC_D_80', total_area=1e6):
    '''Given a geo df (ie shape file), calculate total area per class present'''
    dict_area = {}
    no_class_name = 'NO CLASS'  # name for area without labels

    if len(gdf) == 0:  # no LC labels available at all, everything is no-class:
        dict_area[no_class_name] = 1
        
    else:
        present_classes = gdf[col_class_name].unique()
        for cl in present_classes:
            tmp_df = gdf[gdf[col_class_name] == cl]  # group potentially multiple polygons with same label
            dict_area[cl] = tmp_df['geometry'].area.sum() / total_area
        total_area_polygons = gdf['geometry'].area.sum()
        if total_area_polygons < total_area:
            dict_area[no_class_name] = 1 - total_area_polygons / total_area  # remainder is no-class
        elif total_area_polygons > total_area:  # I think because of float error or so, it is sometimes marginally larger
            assert (total_area_polygons / total_area) < (1 + 1e-5), (total_area_polygons / total_area)
            for cl in present_classes:
                dict_area[cl] = dict_area[cl] / (total_area_polygons / total_area)
            dict_area[no_class_name] = 0

    return dict_area

def create_df_with_class_distr_per_tile(dict_dfs, all_class_names=[], 
                                        filter_no_class=True, no_class_threshold=1):
    '''Creat DF with LC class distribution per tile (row) per class (column)'''
    assert type(dict_dfs) == dict 
    assert type(all_class_names) == list 
    if 'NO CLASS' not in all_class_names:
        all_class_names.append('NO CLASS')
    n_tiles = len(dict_dfs)
    n_classes = len(all_class_names)

    dict_area_all = {cl: np.zeros(n_tiles) for cl in all_class_names}
    tile_names = list(dict_dfs.keys())
    for i_tile, tilename in tqdm(enumerate(tile_names)):  # get area per class per tile
        dict_classes_tile = get_area_per_class_df(gdf=dict_dfs[tilename])
        for cl_name, area in dict_classes_tile.items():
            dict_area_all[cl_name][i_tile] = area 

    df_distr = pd.DataFrame({**{'tile_name': tile_names}, **dict_area_all})  # put in DF
    assert np.isclose(df_distr.sum(axis=1, numeric_only=True), 1, atol=1e-8).all(), 'Area fraction does not sum to 1'
    assert df_distr['NO CLASS'].min() >= 0, 'negative remainder found'
    if filter_no_class:  # optionally, filter tiles that have too much NO CLASS area
        print(f'{len(df_distr)} tiles analysed')
        df_distr = df_distr[df_distr['NO CLASS'] < no_class_threshold]
        print(f'{len(df_distr)} tiles kept after no-class filter')
    return df_distr

def sample_tiles_by_class_distr_from_df(df_all_tiles_distr, n_samples=100, 
                                        iterations=1000, verbose=1):

    n_tiles = len(df_all_tiles_distr)
    # class_distr_mat = df_all_tiles_distr.select_dtypes(include=np.number).to_numpy() 
    class_distr = df_all_tiles_distr.sum(axis=0, numeric_only=True)
    class_distr = class_distr / class_distr.sum()
    
    for it in range(iterations):
        random_inds = np.random.choice(a=n_tiles, size=n_samples, replace=False)
        df_select = df_all_tiles_distr.iloc[random_inds]
        distr_select = df_select.sum(axis=0, numeric_only=True) / n_samples  # normalise to 1 

        # loss = np.sum(np.power(distr_select - class_distr, 2))
        loss = np.sum(np.abs(distr_select - class_distr))
        if it == 0:
            prev_loss = loss 
            best_selection = random_inds

        else:
            if loss < prev_loss:
                best_selection = random_inds 
                prev_loss = loss 
                if verbose > 0:
                    print(f'At it {it} new loss of {prev_loss}')
    
    return best_selection, df_all_tiles_distr.iloc[best_selection]

def get_shp_all_tiles(shp_all_tiles_path=None):
    if shp_all_tiles_path is None:
        shp_all_tiles_path = path_dict['landscape_character_grid_path']
    
    df_all_tiles = load_pols(shp_all_tiles_path)
    return df_all_tiles

def select_tiles_from_list(list_tile_names=[], shp_all_tiles_path=None, save_new_shp=False, 
                           new_shp_filename=None):
    '''Select tiles by name from shape file, make new shape file'''
    df_all_tiles = get_shp_all_tiles(shp_all_tiles_path=shp_all_tiles_path)
    assert np.isin(list_tile_names, df_all_tiles['PLAN_NO']).all(), f'Not all tiles are in DF: {np.array(list_tile_names)[~np.isin(list_tile_names, df_all_tiles["PLAN_NO"])]}'
    inds_tiles = np.isin(df_all_tiles['PLAN_NO'], list_tile_names)
    df_selection = df_all_tiles[inds_tiles]

    if save_new_shp:
        if new_shp_filename is None:
            timestamp = create_timestamp()
            new_shp_filename = f'selection_tiles_{timestamp}.shp'
        elif new_shp_filename[-4:] != '.shp':
            new_shp_filename = new_shp_filename + '.shp'
        df_selection.to_file(new_shp_filename)        

    return df_selection

def convert_shp_mask_to_raster(df_shp, col_name='LC_N_80',
                                resolution=(-0.125, 0.125),
                                interpolation=None,
                                save_raster=False, filename='mask.tif',
                                maskdir=None, plot_raster=False,
                                verbose=0):
    '''
    Turn gdf of shape file polygon into a raster file. Possibly store & plot.

    interpolation:
        - None: nothing done with missing data (turned into 0)
        - 'nearest': using label of nearest pixels (takes bit of extra time)
    '''
    assert not np.isin(0, np.unique(df_shp[col_name])), '0 is already a class label, so cant be used for fill value'
    assert len(resolution) == 2 and resolution[0] < 0 and resolution[1] > 0, 'resolution has unexpected size/values'
    
    ## Convert shape to raster:
    cube = make_geocube(df_shp, measurements=[col_name],
                        interpolate_na_method=interpolation,
                        # like=ex_tile)  # use resolution of example tiff
                        fill=0,
                        resolution=resolution)

    ## Decrease data size:
    if verbose > 0:
        print(f'Current data size cube is {cube.nbytes / 1e6} MB')
    unique_labels = copy.deepcopy(np.unique(cube[col_name]))  # want to ensure these are not messed up 
    assert np.nanmin(unique_labels) >=0 and np.nanmax(unique_labels) < 256, f'unexpectedly high number of labels. conversion to int8 wont work. Labels: {unique_labels}'
    low_size_raster = cube[col_name].astype('uint8')  # goes from 0 to & incl 255
    cube[col_name] = low_size_raster
    new_unique_labels = np.unique(cube[col_name])
    assert (unique_labels == new_unique_labels).all(), f'Old labels: {unique_labels}, new labels: {new_unique_labels}, comaprison: {(unique_labels == new_unique_labels)}'  # unique labels are sorted by default so this works as sanity check
    if verbose > 0:
        print(f'New cube data size is {cube.nbytes / 1e6} MB')

    if save_raster:
        assert type(filename) == str, 'filename must be string'
        if filename[-4:] != '.tif':
            filename = filename + '.tif'
        if maskdir is None:  # use default path for mask files 
            maskdir = path_dict['mask_path']
        filepath = os.path.join(maskdir, filename)
        cube[col_name].rio.to_raster(filepath)
        if verbose > 0:
            print(f'Saved to {filepath}')

    if plot_raster:
        cube[col_name].plot()

    return cube 

def create_image_mask_patches(image, mask=None, patch_size=512):
    '''Given a loaded image (as DataArray) and mask (as np array), create patches (ie sub images/masks)'''
    assert type(image) == xr.DataArray, 'expecting image to be a xr.DataArray'
    assert image.ndim == 3, 'expecting band by x by y dimensions'
    assert patch_size < len(image.x) and patch_size < len(image.y)
    assert len(image.x) == len(image.y)

    if mask is not None:
        assert type(mask) == np.ndarray 
        assert mask.shape == (1, len(image.x), len(image.y))
        mask = np.squeeze(mask)  # get rid of extra dim 

    n_exp_patches = int(np.floor(len(image.x) / patch_size))

    ## Create patches of patch_size x patch_size (x n_bands)
    patches_img = patchify.patchify(image.to_numpy(), (3, patch_size, patch_size), step=patch_size)
    assert patches_img.shape == (1, n_exp_patches, n_exp_patches, 3, patch_size, patch_size)
    assert type(patches_img) == np.ndarray 
    
    if mask is not None:
        patches_mask = patchify.patchify(mask, (patch_size, patch_size), step=patch_size)
        assert patches_mask.shape == (n_exp_patches, n_exp_patches, patch_size, patch_size)
        assert type(patches_mask) == np.ndarray
    else:
        patches_mask = None

    ## Reshape to get array of patches:
    patches_img = np.reshape(np.squeeze(patches_img), (n_exp_patches ** 2, 3, patch_size, patch_size), order='C')
    if mask is not None:
        patches_mask = np.reshape(patches_mask, (n_exp_patches ** 2, patch_size, patch_size), order='C')

        assert patches_img.shape[0] == patches_mask.shape[0]
    
    return patches_img, patches_mask

def create_all_patches_from_dir(dir_im=path_dict['image_path'], 
                                dir_mask=path_dict['mask_path'], 
                                mask_fn_suffix='_lc_80s_mask.tif',
                                patch_size=512, search_subdir_im=False):
    '''Create patches from all images & masks in given dirs.'''
    if search_subdir_im:
        im_paths = get_all_tifs_from_subdirs(dir_im)
    else:
        im_paths = get_all_tifs_from_dir(dir_im)
    mask_paths = get_all_tifs_from_dir(dir_mask)
    # assert len(im_paths) == len(mask_paths), 'different number of masks and images'
    assert type(mask_fn_suffix) == str 
    for ii, image_fn in enumerate(im_paths):
        ## Find mask that matches image by filename:
        im_name = image_fn.split('/')[-1].rstrip('.tif')
        mask_name = im_name + mask_fn_suffix
        mask_fn = os.path.join(dir_mask, mask_name)
        assert mask_fn in mask_paths, f'Mask for {im_name} not found in {dir_mask} under name of {mask_fn}'
        
        image_tile = load_tiff(tiff_file_path=image_fn, datatype='da')
        mask_tif = load_tiff(tiff_file_path=mask_fn, datatype='np')

        patches_img, patches_mask = create_image_mask_patches(image=image_tile, mask=mask_tif, 
                                                              patch_size=patch_size)
        if ii == 0:  # first iteration, create object:
            all_patches_img = patches_img 
            all_patches_mask = patches_mask
        else:
            all_patches_img = np.concatenate((all_patches_img, patches_img), axis=0)
            all_patches_mask = np.concatenate((all_patches_mask, patches_mask), axis=0)

    return all_patches_img, all_patches_mask

def create_and_save_patches_from_tiffs(list_tiff_files=[], list_mask_files=[], 
                                       mask_fn_suffix='_lc_80s_mask.tif', patch_size=512,
                                       dir_im_patches='', dir_mask_patches='', save_files=False):
    '''Function that loads an image tiff and creates patches of im and masks and saves these'''    
    assert mask_fn_suffix[-4:] == '.tif'
    print(f'WARNING: this will save approximately {len(list_tiff_files) / 5 * 1.3}GB of data')

    for i_tile, tilepath in tqdm(enumerate(list_tiff_files)):
        tile_name = tilepath.split('/')[-1].rstrip('.tif')
        maskpath = list_mask_files[np.where(np.array([x.split('/')[-1] for x in list_mask_files]) == tile_name + mask_fn_suffix)[0][0]]
 
        assert tile_name in tilepath and tile_name in maskpath
 
        image_tile = load_tiff(tiff_file_path=tilepath, datatype='da')
        mask_tif = load_tiff(tiff_file_path=maskpath, datatype='np')
        patches_img, patches_mask = create_image_mask_patches(image=image_tile, mask=mask_tif, 
                                                              patch_size=patch_size)
        n_patches = patches_mask.shape[0]
        assert n_patches < 1000, 'if more than 1e3 patches, change zfill in lines below '
        for i_patch in range(n_patches):
            patch_name = tile_name + f'_patch{str(i_patch).zfill(3)}'
            
            im_patch_name = patch_name + '.npy'
            mask_patch_name = patch_name + mask_fn_suffix.rstrip('.tif') + '.npy'

            im_patch_path = os.path.join(dir_im_patches, im_patch_name)
            mask_patch_path = os.path.join(dir_mask_patches, mask_patch_name)
            
            if save_files:
                np.save(im_patch_path, patches_img[i_patch, :, :, :])
                np.save(mask_patch_path, patches_mask[i_patch, :, :])

            if i_tile == 0 and i_patch == 0:
                assert type(patches_img[i_patch]) == np.ndarray
                assert type(patches_mask[i_patch]) == np.ndarray
                assert patches_img[i_patch].shape[1:] == patches_mask[i_patch].shape

def augment_patches(all_patches_img, all_patches_mask):
    '''Augment patches by rotating etc.
    See existing torch functions (eg https://www.kaggle.com/code/haphamtv/pytorch-smp-unet/notebook)'''
    ## Assert data sizes:
    pass
    ## Rotate
    pass
    ## Mirror 
    pass
    return all_patches_img, all_patches_mask

def change_data_to_tensor(*args, tensor_dtype='int'):
    '''Change data to torch tensor type.'''
    assert tensor_dtype in ['int', 'float'], f'tensor dtype {tensor_dtype} not recognised'
    print('WARNING: data not yet normalised!!')
    new_ds = []
    for ds in args:
        ds = torch.tensor(ds)
        if tensor_dtype == 'int':
            # ds = ds.int()
            ds = ds.type(torch.LongTensor)
        elif tensor_dtype == 'float':
            ds = ds.float()
        new_ds.append(ds)
    return tuple(new_ds)

def apply_zscore_preprocess_images(im_ds, f_preprocess, verbose=0):
    '''Apply preprocessing function to image data set.
    Assuming a torch preprocessing function here that essentially z-scores and only works on RGB tensor of shape (3,)'''
    assert type(im_ds) == torch.Tensor, 'expected tensor here'
    assert im_ds.ndim == 4 and im_ds.shape[1] == 3, 'unexpected shape'

    dtype = im_ds.dtype

    rgb_means = f_preprocess.keywords['mean']
    rgb_std = f_preprocess.keywords['std']

    rgb_means = torch.tensor(np.array(rgb_means)[None, :, None, None])  # get into right dimensions
    rgb_std = torch.tensor(np.array(rgb_std)[None, :, None, None])  # get into right dimensions

    ## Change to consistent dtype:
    rgb_means = rgb_means.type(dtype)
    rgb_std = rgb_std.type(dtype)

    if verbose > 0:
        print('Changing range')
    if (im_ds > 1).any():
        im_ds = im_ds / 255 

    if verbose > 0:
        print('Z scoring data')
    im_ds = (im_ds - rgb_means) / rgb_std

    assert im_ds.dtype == torch.float32, f'Expected image to have dtype float32 but instead it has {im_ds.dtype}'
    return im_ds

def apply_zscore_preprocess_single_image(im, f_preprocess):
    '''Apply preprocessing function to a single image.
    Assuming a torch preprocessing function here that essentially z-scores and only works on RGB tensor of shape (3,)'''
    assert type(im) == torch.Tensor, 'expected tensor here'
    assert im.ndim == 3 and im.shape[0] == 3, 'unexpected shape'

    dtype = im.dtype

    if f_preprocess is not None:
        rgb_means = f_preprocess.keywords['mean']
        rgb_std = f_preprocess.keywords['std']

        rgb_means = torch.tensor(np.array(rgb_means)[:, None, None])  # get into right dimensions
        rgb_std = torch.tensor(np.array(rgb_std)[:, None, None])  # get into right dimensions

        ## Change to consistent dtype:
        rgb_means = rgb_means.type(dtype)
        rgb_std = rgb_std.type(dtype)

        if (im > 1).any():
            im = im / 255 

        im = (im - rgb_means) / rgb_std

    assert im.dtype == torch.float32, f'Expected image to have dtype float32 but instead it has {im.dtype}'
    return im

def undo_zscore_single_image(im_ds, f_preprocess):
    '''Undo the z scoring of f_preprocess'''
    dtype = im_ds.dtype

    rgb_means = f_preprocess.keywords['mean']
    rgb_std = f_preprocess.keywords['std']

    rgb_means = torch.tensor(np.array(rgb_means)[None, :, None, None])  # get into right dimensions
    rgb_std = torch.tensor(np.array(rgb_std)[None, :, None, None])  # get into right dimensions

    ## Change to consistent dtype:
    rgb_means = rgb_means.type(dtype)
    rgb_std = rgb_std.type(dtype)

    im_ds = im_ds * rgb_std + rgb_means  # not multiplying with 255 because image plotting function from rasterio doesnt like that
    
    assert im_ds.dtype == torch.float32, f'Expected image to have dtype float32 but instead it has {im_ds.dtype}'
    return im_ds

def create_empty_label_mapping_dict():
    '''Create empty dict with right format for label mapping'''

    dict_ind_to_name, _ = get_lc_mapping_inds_names_dicts()  # get labels of PD

    ## Add labels to dict that don't exist PD (for sake of completeness):
    dict_ind_to_name[10] = 'Unenclosed Lowland Rough Grassland'
    dict_ind_to_name[11] = 'Unenclosed Lowland Heath'
    dict_ind_to_name[17] = 'Coastal Heath'
    dict_ind_to_name[21] = 'Open Water, Coastal'
    dict_ind_to_name[23] = 'Wetland, Peat Bog'
    dict_ind_to_name[24] = 'Wetland, Freshwater Marsh'
    dict_ind_to_name[25] = 'Wetland, Saltmarsh'
    dict_ind_to_name[27] = 'Coastal Bare Rock'
    dict_ind_to_name[28] = 'Coastal Dunes'
    dict_ind_to_name[29] = 'Coastal Sand Beach'
    dict_ind_to_name[30] = 'Coastal Shingle Beach'
    dict_ind_to_name[31] = 'Coastal Mudflats'

    n_classes = 39  # hard code to ensure asserts return expected behaviour
    assert len(dict_ind_to_name) == n_classes
    assert len(np.unique(list(dict_ind_to_name.values()))) == n_classes
    assert (np.sort(np.array(list(dict_ind_to_name.keys()))) == np.arange(n_classes)).all()

    ## Set up format of label mapping dict with a trivial identity transformation:
    dict_label_mapping = {x: x for x in range(len(dict_ind_to_name))}  # dummy dict (with identity transformation)
    dict_name_mapping = {v: v for v in dict_ind_to_name.values()}
    old_names = {x: dict_ind_to_name[x] for x in range(n_classes)}
    new_names = {x: dict_ind_to_name[x] for x in range(n_classes)}

    dict_mapping = {'dict_label_mapping': dict_label_mapping,
                    'dict_name_mapping': dict_name_mapping,
                    'dict_old_names': old_names, 
                    'dict_new_names': new_names}

    return dict_mapping

def change_lc_label_in_dict(dict_mapping, dict_new_names,
                            old_ind_list=[0], new_ind=0, new_name=''):
    for ind in old_ind_list:  # woodlands
        ## dict_old_names stays as is
        dict_mapping['dict_label_mapping'][ind] = new_ind
        dict_mapping['dict_name_mapping'][dict_mapping['dict_old_names'][ind]] = new_name
    dict_new_names[new_ind] = new_name

    return dict_mapping, dict_new_names

def create_new_label_mapping_dict(mapping_type='identity', save_folder='/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/',
                                  save_mapping=False):

    dict_mapping = create_empty_label_mapping_dict()

    if mapping_type == 'identity':
        pass 
    elif mapping_type == 'main_categories':
        dict_new_names = {}
        
        list_old_inds_new_name = [  
                                    ([0, 38], 'NO CLASS'),
                                    ([1, 2, 3, 4, 5], 'Wood and Forest Land'),
                                    ([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 'Moor and Heath Land'),
                                    ([18, 19, 20], 'Agro-Pastoral Land'),
                                    ([21, 22, 23, 24, 25], 'Water and Wetland'),
                                    ([26, 27, 28, 29, 30, 31], 'Rock and Coastal Land'),
                                    ([32, 33, 34, 35, 36, 37], 'Developed Land')
                                 ]

        for new_ind, (old_ind_list, new_name) in enumerate(list_old_inds_new_name):
            
            dict_mapping, dict_new_names = change_lc_label_in_dict(dict_mapping=dict_mapping, dict_new_names=dict_new_names,
                                                            old_ind_list=old_ind_list, new_ind=new_ind, new_name=new_name)
        ## finish:
        dict_mapping['dict_new_names'] = dict_new_names

    if save_mapping:
        timestamp = create_timestamp()
        fn = f'label_mapping_dict__{mapping_type}__{timestamp}.pkl'
        filepath = os.path.join(save_folder, fn)
        with open(filepath, 'wb') as f:
            pickle.dump(dict_mapping, f)

    return dict_mapping

def change_labels_to_consecutive_numbers(mask_patches, unique_labels_array=None, 
                                         use_all_pd_classes=False, verbose=0):
    '''Map labels to consecutive numbers (eg [0, 2, 5] to [0, 1, 2])'''
    assert type(mask_patches) == np.ndarray, 'expected np array here'

    if use_all_pd_classes is False:
        if unique_labels_array is None:
            if verbose > 0:
                print('Using classes of provided masks for relabelling')
            ## Warning: this might take some time to compute if there are many patches
            unique_labels_array = np.unique(mask_patches)
        else:
            if verbose > 0:
                print('Using provided unique labels for relabelling')
            assert type(unique_labels_array) == np.array 

        mapping_label_to_new_dict = {label: ind for ind, label in enumerate(unique_labels_array)}

    dict_ind_to_name, dict_name_to_ind =  get_lc_mapping_inds_names_dicts()

    if use_all_pd_classes:
        if verbose > 0:
            print('Using all PD classes for relabelling')
        unique_labels_array = np.unique(np.array(list(dict_ind_to_name.keys())))  # unique sorts too
        mapping_label_to_new_dict = {label: ind for ind, label in enumerate(unique_labels_array)}
    class_name_list = [dict_ind_to_name[label] for label in unique_labels_array]

    new_mask = np.zeros_like(mask_patches)  # takes up more RAM (instead of reassigning mask_patches.. But want to make sure there are no errors when changing labels). Although maybe it's okay because with labels >= 0 you're always changing down so no chance of getting doubles I think.
    for ind, label in enumerate(unique_labels_array):
        new_mask[mask_patches == label] = mapping_label_to_new_dict[label]

    # print(type(new_mask), np.shape(new_mask))
    print('WARNING: changing the labels messes with the associated label names later on')
    return new_mask, (unique_labels_array, mapping_label_to_new_dict, class_name_list)

def create_data_loaders(x_train, x_test, y_train, y_test, batch_size=100):
    '''Create torch data loaders'''
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)  # could specify num_workers?? 

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl

def print_info_ds(ds):
    '''Print basic info of data set'''
    dtype = ds.dtype 
    shape = ds.shape 
    print(f'{dtype} of shape {shape}')

def get_distr_classes_from_patches(patches_mask):
    '''Count for each class label the number of occurences'''
    class_inds, frequency = np.unique(patches_mask, return_counts=True)
    return (class_inds, frequency)

def split_patches_in_train_test(all_patches_img, all_patches_mask, 
                                fraction_test=0.2, split_method='random',
                                augment_data=False):
    '''Split image and mask patches into a train and test set'''
    assert type(all_patches_img) == np.ndarray and type(all_patches_mask) == np.ndarray
    assert all_patches_img.ndim == 4 and all_patches_mask.ndim == 3
    assert all_patches_img[:, 0, :, :].shape == all_patches_mask.shape 

    if split_method == 'random':
        im_train, im_test, mask_train, mask_test = sklearn.model_selection.train_test_split(all_patches_img, all_patches_mask,
                                                                                        test_size=fraction_test)
    else:
        assert False, f'Split method {split_method} not implemented'

    if augment_data:  # augment the train/test sets separately so augmented images are always in the same set 
        im_train, mask_train = augment_patches(all_patches_img=im_train, all_patches_mask=mask_train)
        im_test, mask_test = augment_patches(all_patches_img=im_test, all_patches_mask=mask_test)

    return im_train, im_test, mask_train, mask_test
    
def check_torch_ready(verbose=1, check_gpu=True, assert_versions=False):
    '''Check if pytorch, cuda, gpu, etc are ready to be used'''
    if check_gpu:
        assert torch.cuda.is_available()
    if verbose > 0:  # possibly also insert assert versions
        print(f'Pytorch version is {torch.__version__}') 
        # print(f'Torchvision version is {torchvision.__version__}')  # not using torchvision at the moment though .. 
        print(f'Segmentation-models-pytorch version is {smp.__version__}')
    if assert_versions:
        assert torch.__version__ == '1.12.1+cu102'
        # assert torchvision.__version__ == '0.13.1+cu102'
        assert smp.__version__ == '0.3.0'

def change_tensor_to_max_class_prediction(pred, expected_square_size=512):
    '''CNN typically outputs a prediction for each class. This function finds the max/most likely
    predicted class and gets rid of dimenion.'''

    assert type(pred) == torch.Tensor, 'expected tensor'
    assert pred.ndim == 4, 'expected 4D (batch x class x width x height'
    assert pred.shape[2] == pred.shape[3] and pred.shape[2] == expected_square_size
    ## assuming dim 0 is batch size and dim 1 is class size. 

    pred = torch.argmax(pred, dim=1)

    return pred 

def concat_list_of_batches(batches):
    '''Concatenate list of batch of output masks etc'''
    assert type(batches) == list 
    for b in batches:
        assert type(b) == torch.Tensor, type(b)

    return torch.cat(batches, dim=0)