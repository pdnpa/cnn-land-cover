import os, sys, copy
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
from patchify import patchify 
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp

path_dict = loadpaths.loadpaths()

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

    return dict_ind_to_name, dict_name_to_ind

def get_pols_for_tiles(df_pols, df_tiles):
    '''Extract polygons that are inside a tile, for all tiles in df_tiles. Assuming a df for tiles currently.'''

    n_tiles = len(df_tiles)
    dict_pols = {}
    for i_tile in range(n_tiles):  # loop through tiles, process individually:
        tile = df_tiles.iloc[i_tile]
        pol_tile = tile['geometry']  # polygon of tile 
        name_tile = tile['name']

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

def create_image_mask_patches(image, mask, patch_size=500):
    '''Given a loaded image (as DataArray) and mask (as np array), create patches (ie sub images/masks)'''
    assert type(image) == xr.DataArray, 'expecting image to be a xr.DataArray'
    assert image.ndim == 3, 'expecting band by x by y dimensions'
    assert patch_size < len(image.x) and patch_size < len(image.y)
    assert len(image.x) == len(image.y)

    assert type(mask) == np.ndarray 
    assert mask.shape == (1, len(image.x), len(image.y))
    mask = np.squeeze(mask)  # get rid of extra dim 

    n_exp_patches = int(np.floor(len(image.x) / patch_size))

    ## Create patches of patch_size x patch_size (x n_bands)
    patches_img = patchify(image.to_numpy(), (3, patch_size, patch_size), step=patch_size)
    patches_mask = patchify(mask, (patch_size, patch_size), step=patch_size)
    assert patches_img.shape == (1, n_exp_patches, n_exp_patches, 3, patch_size, patch_size)
    assert patches_mask.shape == (n_exp_patches, n_exp_patches, patch_size, patch_size)
    assert type(patches_img) == np.ndarray and type(patches_mask) == np.ndarray

    ## Reshape to get array of patches:
    patches_img = np.reshape(np.squeeze(patches_img), (n_exp_patches ** 2, 3, patch_size, patch_size), order='C')
    patches_mask = np.reshape(patches_mask, (n_exp_patches ** 2, patch_size, patch_size), order='C')

    assert patches_img.shape[0] == patches_mask.shape[0]
    return patches_img, patches_mask

def create_all_patches_from_dir(dir_im=path_dict['image_path'], 
                                dir_mask=path_dict['mask_path'], 
                                mask_fn_suffix='_lc_80s_mask.tif',
                                patch_size=500):
    '''Create patches from all images & masks in given dirs.'''
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
        n_patches = patches_mask.shape[0]
        if ii == 0:  # first iteration, create object:
            all_patches_img = patches_img 
            all_patches_mask = patches_mask
        else:
            all_patches_img = np.concatenate((all_patches_img, patches_img), axis=0)
            all_patches_mask = np.concatenate((all_patches_mask, patches_mask), axis=0)

    return all_patches_img, all_patches_mask

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

# def change_data_to_tensor(x_train, y_train, x_test, y_test):
#     '''Change data to torch tensor type. Could be tidier with args'''
#     x_train, y_train, x_test, y_test = map(
#         torch.tensor, (x_train, y_train, x_test, y_test))  # create tensors
#     x_train, y_train, x_test, y_test = x_train.int(), y_train.int(), x_test.int(), y_test.int()  # need to be float type (instead of 'double', which is somewhat silly)
#     return x_train, y_train, x_test, y_test

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

def create_data_loaders(x_train, x_test, y_train, y_test, batch_size=100):
    '''Create torch data loaders'''
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size)  # could specify num_workers?? 

    test_ds = TensorDataset(x_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl

def print_info_ds(ds):
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
        print(f'Torchvision version is {torchvision.__version__}')
        print(f'Segmentation-models-pytorch version is {smp.__version__}')
    if assert_versions:
        assert torch.__version__ == '1.12.1+cu102'
        assert torchvision.__version__ == '0.13.1+cu102'
        assert smp.__version__ == '0.3.0'