import os, sys, copy, datetime, pickle
import time, datetime
import numpy as np
from numpy.core.multiarray import square
from numpy.testing import print_assert_equal
import rasterio
import xarray as xr
import rioxarray as rxr
import rtree
import scipy.spatial
import sklearn.cluster, sklearn.model_selection
from tqdm import tqdm
import shapely as shp
import shapely.validation
from rasterio.features import shapes
from shapely.geometry import shape
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
import gdal, osr
import libpysal
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
    if df_pols.crs is not None:
        assert_epsg(df_pols.crs.to_epsg())
    else:
        print('WARNING: no crs found in shape file. Continuing without crs check.')
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

def get_mapping_class_names_to_shortcut():
    mapping_dict_to_full = {'C': 'Wood and Forest Land',
                    'D': 'Moor and Heath Land',
                    'E': 'Agro-Pastoral Land',
                    'F': 'Water and Wetland',
                    'G': 'Rock and Coastal Land',
                    'H': 'Developed Land',
                    'I': 'Unclassified Land',
                    '0': 'NO CLASS'}

    mapping_dict_to_shortcut = {v: k for k, v in mapping_dict_to_full.items()}
    return mapping_dict_to_full, mapping_dict_to_shortcut

def add_main_category_column(df_lc, col_ind_name='LC_N_80', col_code_name='Class_Code'):
    '''Add main category class code'''
    mapping_dict = {**{x: 'C' for x in range(1, 6)},
                    **{x: 'D' for x in range(6, 18)},
                    **{x: 'E' for x in range(18, 21)},
                    **{x: 'F' for x in range(21, 26)},
                    **{x: 'G' for x in range(26, 32)}, 
                    **{x: 'H' for x in range(32, 38)},
                    **{38: 'I'}}

    class_col = np.zeros(len(df_lc[col_ind_name]), dtype=str)
    for ind, label in mapping_dict.items():
        class_col[df_lc[col_ind_name] == ind] = label 
    
    df_lc[col_code_name] = class_col

    return df_lc

def add_main_category_name_column(df_lc, col_code_name='Class_Code', col_label_name='Class name'):
    '''Add column with main category names, based on class codes'''
    mapping_dict = {'C': 'Wood and Forest Land',
                    'D': 'Moor and Heath Land',
                    'E': 'Agro-Pastoral Land',
                    'F': 'Water and Wetland',
                    'G': 'Rock and Coastal Land',
                    'H': 'Developed Land',
                    'I': 'Unclassified Land'}

    class_col = list(np.zeros(len(df_lc[col_code_name]), dtype=str))
    for code, label in mapping_dict.items():
        inds_code = np.where(df_lc[col_code_name] == code)[0]
        for ii in inds_code:
            class_col[ii] = label 
    
    df_lc[col_label_name] = class_col

    return df_lc

def add_main_category_index_column(df_lc, col_code_name='Class_Code', col_ind_name='class_ind'):
    '''Add column with main category indices, based on class codes'''
    assert col_code_name in df_lc.columns, df_lc.columns
    mapping_dict = {'C': 1,
                    'D': 2,
                    'E': 3,
                    'F': 4,
                    'G': 5,
                    'H': 6,
                    'I': 0}

    class_ind_col = np.zeros(len(df_lc[col_code_name]), dtype=int)
    for code, new_ind in mapping_dict.items():
        inds_code = np.where(df_lc[col_code_name].apply(lambda x: x[0]) == code)[0]  # compare first letter
        class_ind_col[inds_code] = new_ind 
    
    df_lc[col_ind_name] = class_ind_col

    return df_lc

def test_validity_geometry_column(df):
    '''Test if all polygons in geometry column of df are valid. If not, try to fix.'''
    arr_valid = np.array([shapely.validation.explain_validity(df['geometry'].iloc[x]) for x in range(len(df))])
    unique_vals = np.unique(arr_valid)
    if len(unique_vals) == 1 and unique_vals[0] == 'Valid Geometry':
        return df 
    else:
        for val in unique_vals:
            if val != 'Valid Geometry':
                inds_val = np.where(arr_valid == val)[0] 
                print(f'Geometry {val} for inds {inds_val}')
                print('Attempting to make valid')
                for ind in inds_val:
                    new_geom = shapely.validation.make_valid(df['geometry'].iloc[ind])
                    df['geometry'].iloc[ind] = new_geom
                print('Done')
        return df


def get_pols_for_tiles(df_pols, df_tiles, col_name='name', extract_main_categories_only=False):
    '''Extract polygons that are inside a tile, for all tiles in df_tiles. Assuming a df for tiles currently.'''

    n_tiles = len(df_tiles)
    dict_pols = {}
    for i_tile in tqdm(range(n_tiles)):  # loop through tiles, process individually:
        tile = df_tiles.iloc[i_tile]
        pol_tile = tile['geometry']  # polygon of tile 
        name_tile = tile[col_name]
        df_relevant_pols = df_pols[df_pols.geometry.intersects(pol_tile)]  # find polygons that overlap with tile
        list_pols = []
        list_class_id = []
        list_class_name = []
        list_class_code = []
        for i_pol in range(len(df_relevant_pols)):  # loop through pols
            new_pol = df_relevant_pols.iloc[i_pol]['geometry'].intersection(pol_tile)  # create intersection between pol and tile
            list_pols.append(new_pol)
            if extract_main_categories_only:
                list_class_code.append(df_relevant_pols.iloc[i_pol]['Class_Code'])
                if df_relevant_pols.iloc[i_pol]['Class_Code'] is None: 
                    print(f'{name_tile} contains a polygon with missing Class_Code label')
            else:
                list_class_id.append(df_relevant_pols.iloc[i_pol]['LC_N_80'])
                list_class_name.append(df_relevant_pols.iloc[i_pol]['LC_D_80'])
        if extract_main_categories_only:  # kind of a silly way to do this, but wasnt sure how to soft code these? look into it again if more columns are (potentially ) needed
            dict_pols[name_tile] = gpd.GeoDataFrame(geometry=list_pols).assign(Class_Code=list_class_code)  # put all new intersections back into a dataframe
        else:
            dict_pols[name_tile] = gpd.GeoDataFrame(geometry=list_pols).assign(LC_N_80=list_class_id, LC_D_80=list_class_name)  # put all new intersections back into a dataframe

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
        n_tiles_before = len(df_distr)
        df_distr = df_distr[df_distr['NO CLASS'] < no_class_threshold]
        n_tiles_after = len(df_distr)
        if n_tiles_after != n_tiles_before:
            print(f'{n_tiles_after}/{n_tiles_before} tiles kept after no-class filter')
    return df_distr

def sample_tiles_by_class_distr_from_df(df_all_tiles_distr, n_samples=100, 
                                        class_distr = None,
                                        iterations=1000, verbose=1):

    n_tiles = len(df_all_tiles_distr)
    if class_distr is None:  # use distr of given df
        class_distr = df_all_tiles_distr.sum(axis=0, numeric_only=True)
        class_distr = class_distr / class_distr.sum()
    else:
        assert len(class_distr) == 27, f'expected 27 classes but received {len(class_distr)}'
        assert type(class_distr) == np.array or type(class_distr) == pd.core.series.Series, type(class_distr)
        assert np.sum(class_distr) == 1 or np.isclose(np.sum(class_distr), 1, atol=1e-8)
        print('Using predefined class distribution')

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

def save_tile_names_to_list(list_tile_names, text_filename):
    assert type(list_tile_names) == list 
    assert type(text_filename) == str and text_filename[-4:] == '.txt'
    print(f'Saving {len(list_tile_names)} tile names to {text_filename}')

    with open(text_filename, 'w') as f:
        for line in list_tile_names:
            f.write(f"{line}\n")

def convert_shp_mask_to_raster(df_shp, col_name='LC_N_80',
                                resolution=(-0.125, 0.125),
                                interpolation=None, 
                                save_raster=False, filename='mask.tif',
                                maskdir=None, plot_raster=False,
                                verbose=0):
    '''
    Turn gdf of shape file polygon into a raster file. Possibly store & plot.
    Assumes col_name is a numeric column with class labels.

    interpolation:
        - None: nothing done with missing data (turned into 0)
        - 'nearest': using label of nearest pixels (takes bit of extra time)
    '''
    # assert not np.isin(0, np.unique(df_shp[col_name])), '0 is already a class label, so cant be used for fill value'
    assert len(resolution) == 2 and resolution[0] < 0 and resolution[1] > 0, 'resolution has unexpected size/values'
    
    ## Convert shape to raster:
    cube = make_geocube(df_shp, measurements=[col_name],
                        interpolate_na_method=interpolation,
                        # like=ex_tile,  # use resolution of example tiff
                        resolution=resolution,
                        fill=0)
    shape_cube = cube[col_name].shape  # somehow sometimes an extra row or of NO CLASS is added... 
    if shape_cube[0]  == 8001:
        if len(np.unique(cube[col_name][0, :])) > 1:
            print(f'WARNING: {filename} has shape {shape_cube} but first y-row contains following classes: {np.unique(cube[col_name][:, 0])}. Still proceeding..')    
        # assert np.unique(cube[col_name][0, :]) == np.array([0])
        cube = cube.isel(y=np.arange(1, 8001))  #discard first one that is just no classes 
    if shape_cube[1] == 8001:
        if len(np.unique(cube[col_name][:, 0])) > 1:
            print(f'WARNING: {filename} has shape {shape_cube} but first x-col contains following classes: {np.unique(cube[col_name][:, 0])}. Still proceeding..')    
        # assert np.unique(cube[col_name][:, 0]) == np.array([0])
        cube = cube.isel(x=np.arange(1, 8001))  #discard first one that is just no classes 

    assert cube[col_name].shape == (8000, 8000), f'Cube of {filename} is not expected shape, but {cube[col_name].shape}'

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
        # print(maskdir, filename)
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
        assert mask.shape == (1, len(image.x), len(image.y)), mask.shape
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
    print(f'WARNING: this will save approximately {np.round(len(list_tiff_files) / 5 * 1.3)}GB of data')

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

def change_data_to_tensor(*args, tensor_dtype='int', verbose=1):
    '''Change data to torch tensor type.'''
    assert tensor_dtype in ['int', 'float'], f'tensor dtype {tensor_dtype} not recognised'
    if verbose > 0:
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

def compute_confusion_mat_from_two_masks(mask_true, mask_pred, lc_class_name_list, 
                                         unique_labels_array, skip_factor=None):
    '''Compute confusion matrix given two np or da masks/matrices.'''
    if type(mask_true) == xr.DataArray:
        mask_true = mask_true.to_numpy()
    if type(mask_pred) == xr.DataArray:
        mask_pred = mask_pred.to_numpy()
    mask_true = np.squeeze(mask_true)
    mask_pred = np.squeeze(mask_pred)

    assert mask_pred.shape == mask_true.shape, f'{mask_pred.shape}, {mask_true.shape}'
    assert len(lc_class_name_list) == len(unique_labels_array)
    n_classes = len(lc_class_name_list)
    conf_mat = np.zeros((n_classes, n_classes))

    if skip_factor is not None and skip_factor > 0:
        assert type(skip_factor) == int
        assert skip_factor > 0 and skip_factor < mask_pred.shape[0] and skip_factor < mask_pred.shape[1]
        mask_pred = mask_pred[::skip_factor, ::skip_factor]
        mask_true = mask_true[::skip_factor, ::skip_factor]
    for ic_true in range(n_classes):
        for ic_pred in range(n_classes):
            n_match = int((mask_pred[mask_true == ic_true] == ic_pred).sum())
            conf_mat[ic_true, ic_pred] += n_match  # just add to existing matrix; so it can be done in batches

    return conf_mat

def compute_stats_from_confusion_mat(model=None, conf_mat=None, class_name_list=None,
                                     dim_truth=0, normalise_hm=True):
    '''Given a confusion matrix, compute precision/sensitivity/accuracy etc stats'''
    if model is not None:
        if conf_mat is not None:
            print('WARNING: using models confusion matrix even though conf_mat was given')
        conf_mat = model.test_confusion_mat 
        class_name_list = model.dict_training_details['class_name_list']
        n_classes = model.dict_training_details['n_classes']
    else:
        n_classes = conf_mat.shape[0]
    assert conf_mat.ndim == 2 and conf_mat.shape[0] == conf_mat.shape[1]
    assert len(class_name_list) == conf_mat.shape[0], len(class_name_list) == n_classes
    assert (conf_mat >= 0).all()
    assert dim_truth == 0, 'if true labels are on the other axis, code below doesnt work. Add transpose here..?'

    _, dict_name_to_shortcut = get_mapping_class_names_to_shortcut()
    shortcuts = ''.join([dict_name_to_shortcut[x] for x in class_name_list])        
    assert len(shortcuts) == n_classes

    if normalise_hm:
        conf_mat_norm = conf_mat / conf_mat.sum() 
    else:
        conf_mat_norm = conf_mat / (64 * 1e6)  # convert to square km

    sens_arr = np.zeros(n_classes)  #true positive rate
    # spec_arr = np.zeros(n_classes)  # true negative rate
    prec_arr = np.zeros(n_classes)  # positive predictive value (= 1 - false discovery rate)
    dens_true_arr = np.zeros(n_classes)   # density of true class
    dens_pred_arr = np.zeros(n_classes)

    for i_c in range(n_classes):
        dens_true_arr[i_c] = conf_mat_norm[i_c, :].sum()  # either density (when normalised) or total area
        dens_pred_arr[i_c] = conf_mat_norm[:, i_c].sum()
        if dens_true_arr[i_c] > 0:
            sens_arr[i_c] = conf_mat_norm[i_c, i_c] / dens_true_arr[i_c]  # sum of true pos + false neg
        else:
            sens_arr[i_c] = np.nan 
        if dens_pred_arr[i_c] > 0: 
            prec_arr[i_c] = conf_mat_norm[i_c, i_c] / dens_pred_arr[i_c]  # sum of true pos + false pos
        else:
            prec_arr[i_c] = np.nan

    df_stats_per_class = pd.DataFrame({'class name': class_name_list, 'class shortcut': [x for x in shortcuts],
                                      'sensitivity': sens_arr, 
                                      'precision': prec_arr, 'true density': dens_true_arr,
                                      'predicted density': dens_pred_arr})

    overall_accuracy = conf_mat.diagonal().sum() / conf_mat.sum() 
    sub_mat = conf_mat[1:4, :][:, 1:4]
    sub_accuracy = sub_mat.diagonal().sum() / sub_mat.sum()

    return df_stats_per_class, overall_accuracy, sub_accuracy, conf_mat_norm, shortcuts, n_classes

def compute_confusion_mat_from_dirs(dir_mask_true,  
                                    lc_class_name_list, unique_labels_array,
                                    dir_mask_pred_shp=None, dir_mask_pred_tif=None,
                                    path_mapping_pred_dict=None,
                                    col_name_shp_file='class', shape_predicted_tile=(7680, 7680),
                                    skip_factor=None, mask_suffix='_lc_2022_mask', verbose=1):
    '''Compute confusion mat & stats per tile, for a dir of tiles. 
    Assuming that dir_mask_true is a dir containing TIFs. 
    Assuming that if dir_mask_pred_shp is given, this is a dir of shp files'''
    list_mask_true = get_all_tifs_from_dir(dir_mask_true)
    if dir_mask_pred_shp is not None and dir_mask_pred_tif is None:
        print('Loading predicted mask shp files')
        list_names_masks_pred = [x for x in os.listdir(dir_mask_pred_shp) if x != 'merged_tiles']
        print(f'Found {len(list_names_masks_pred)} predicted mask shp files')
        list_files_masks_pred = [os.path.join(dir_mask_pred_shp, xx, f'{xx}.shp') for xx in list_names_masks_pred]
        load_pred_as_shp = True 
    elif dir_mask_pred_tif is not None and dir_mask_pred_shp is None:
        print('Loading predicted mask tif files')
        list_files_masks_pred = get_all_tifs_from_dir(dir_mask_pred_tif)
        load_pred_as_shp = False
    else:
        raise ValueError('Either dir_mask_pred_shp or dir_mask_pred_tif must be specified')

    if path_mapping_pred_dict is not None:
        if verbose > 0:
            print('Loading dict mapping predicted labels to original labels')
        dict_mapping_pred = pickle.load(open(path_mapping_pred_dict, 'rb'))
        dict_mapping_pred = dict_mapping_pred['dict_label_mapping']
        unique_original_labels = np.array(list(dict_mapping_pred.keys()))
        remap_pred_labels = True 
    else:
        remap_pred_labels = False

    dict_acc = {} 
    dict_conf_mat = {}
    dict_df_stats = {}

    for i_tile, tilepath in tqdm(enumerate(list_mask_true)):
        tilename = tilepath.split('/')[-1].rstrip('.tif')[:6]#.rstrip(mask_suffix)
        mask_tile_true = np.squeeze(load_tiff(tilepath, datatype='np'))

        corresponding_shp_path = [xx for xx in list_files_masks_pred if tilename in xx]
        assert len(corresponding_shp_path) == 1, f'Tile {tilename}, Length: {len(corresponding_shp_path)}, {corresponding_shp_path}'
        corresponding_shp_path = corresponding_shp_path[0]
        
        if load_pred_as_shp:
            mask_pred_shp = load_pols(corresponding_shp_path)
            ds_pred_tile = convert_shp_mask_to_raster(df_shp=mask_pred_shp, col_name=col_name_shp_file)
            np_pred_tile = ds_pred_tile[col_name_shp_file].to_numpy()
        else:
            np_pred_tile = np.squeeze(load_tiff(corresponding_shp_path, datatype='np'))
            if remap_pred_labels:
                ## Remap if needed:
                    
                new_mask = np.zeros_like(np_pred_tile)  # takes up more RAM (instead of reassigning mask_patches.. But want to make sure there are no errors when changing labels). Although maybe it's okay because with labels >= 0 you're always changing down so no chance of getting doubles I think.
                for label in unique_original_labels:
                    new_mask[np_pred_tile == label] = dict_mapping_pred[label]
                np_pred_tile = new_mask

        ## Cut off no-class edge
        assert mask_tile_true.shape == np_pred_tile.shape, f'Predicted mask shape {np_pred_tile.shape} does not match true mask shape {mask_tile_true.shape}'
        np_pred_tile = np_pred_tile[:shape_predicted_tile[0], :shape_predicted_tile[1]]
        mask_tile_true = mask_tile_true[:shape_predicted_tile[0], :shape_predicted_tile[1]]

        ## Compute confusion matrix:
        conf_mat = compute_confusion_mat_from_two_masks(mask_true=mask_tile_true, mask_pred=np_pred_tile, 
                                                    lc_class_name_list=lc_class_name_list, 
                                                    unique_labels_array=unique_labels_array, skip_factor=skip_factor)
        tmp = compute_stats_from_confusion_mat(conf_mat=conf_mat, class_name_list=lc_class_name_list, 
                                               normalise_hm=True)

        dict_df_stats[tilename] = tmp[0] 
        dict_acc[tilename] = tmp[1]
        dict_conf_mat[tilename] = conf_mat

    return dict_acc, dict_df_stats, dict_conf_mat

def filter_small_polygons_from_gdf(gdf, area_threshold=1e1, class_col='class', verbose=1, max_it=5, ignore_index=0):
    '''Filter small polygons by changing all polygons with area < area_threshold to label of neighbour'''
    assert type(gdf) == gpd.GeoDataFrame
    n_pols_start = len(gdf)
    gdf = copy.deepcopy(gdf)
       
    ## Each iteration of the loop will dissolve polygons that are smaller than area_threshold and adjacent to a large polygon
    ## But if a small polygon is inside another small polygon, it will not be dissolved. Hence the while loop to iterate until no more small polygons are left.
    current_it = 0
    continue_dissolving = True
    sort_ascending = True
    while continue_dissolving:
        if verbose > 0:
            print(f'Current iteration: {current_it}/{max_it}')
        area_array = gdf['geometry'].area
        inds_pols_greater_th = np.where(np.logical_and(area_array >= area_threshold, gdf[class_col] != ignore_index))[0]  # don't take into account no-class (index by ignore_index) for large pols
        inds_pols_lower_th = np.where(area_array < area_threshold)[0]
        n_pols_start_loop = len(gdf)
        if verbose > 0 and current_it == 0:
            print(f'Number of pols smaller than {area_threshold}: {len(inds_pols_lower_th)}/{n_pols_start}')
        other_cols = [x for x in gdf.columns if x not in ['geometry', class_col]]

        if len(inds_pols_greater_th) == 0:
            ## No pols greater than area threshold; convert all small pols to no-class. Do this manually to speed up. 
            if verbose > 0:
                print('No pols greater than area threshold. Converting all small pols to no-class')
            bounds_tile = tuple(gdf.total_bounds)
            pol_tile = shapely.geometry.box(*bounds_tile)
            gdf_new = gpd.GeoDataFrame(geometry=[pol_tile], crs=gdf.crs)
            gdf_new[class_col] = ignore_index
            no_class_inds_original_gdf = np.where(gdf[class_col] == ignore_index)[0]
            if len(no_class_inds_original_gdf) > 0:
                for col_name in other_cols:
                    gdf_new[col_name] = gdf.iloc[no_class_inds_original_gdf[0]][col_name]
            else:
                for col_name in other_cols:
                    gdf_new[col_name] = np.nan
            return gdf_new 

        elif len(inds_pols_greater_th) == 1:
            ## Only 1 pol greater than are; convert all small pols to this class. Do this manually to speed up. 
            if verbose > 0:
                print('Only 1 pol greater than area threshold. Converting all small pols to this class')
            ind_large_pol = inds_pols_greater_th[0]
            bounds_tile = tuple(gdf.total_bounds)
            pol_tile = shapely.geometry.box(*bounds_tile)
            gdf_new = gpd.GeoDataFrame(geometry=[pol_tile], crs=gdf.crs)
            gdf_new[class_col] = gdf.iloc[ind_large_pol][class_col]
            for col_name in other_cols:
                gdf_new[col_name] = gdf.iloc[ind_large_pol][col_name]
            gdf_new = gdf_new.assign(area=gdf_new['geometry'].area)
            return gdf_new

        ## Sort large pols by area for faster NN search
        gdf_l = gdf.iloc[inds_pols_greater_th].copy()
        gdf_l = gdf_l.assign(area=gdf_l['geometry'].area)
        gdf_l = gdf_l.sort_values(by='area', ascending=sort_ascending) 
        
        ## Create an rtree of large pols for fast NN search. 
        idx = gdf_l.sindex

        ## For each small pol, find the nearest large pol and take over class. 
        for i_pol, ind_pol in tqdm(enumerate(inds_pols_lower_th)):
            # Rtree works based on bounds, so it typically selects too many. Using rtree, find selection of large pols:
            pol = gdf.iloc[ind_pol]['geometry']
            list_selection_nearby_large_pols = np.sort(list(idx.intersection(pol.bounds)))  # sort to maintain order of area
            n_sel = len(list_selection_nearby_large_pols)
            gdf_selection = gdf_l.iloc[list_selection_nearby_large_pols]
            convert_pol = False 

            ## Loop through selection of large pols based on bounds to find exact boundary pol using touches():
            for i_large_pol, large_pol in enumerate(gdf_selection['geometry']):
                if large_pol.touches(pol):
                    ind_nearest_pol = list_selection_nearby_large_pols[i_large_pol]
                    convert_pol = True
                    break
                if i_large_pol == n_sel - 2:  # this is the second last large pol, so it must be the last one. Because they are sorted by area, the last one will take most time (especially when there is 1 huge polygon at the end)
                    ind_nearest_pol = list_selection_nearby_large_pols[-1]
                    convert_pol = True
                    break
        
            if convert_pol:  # just to make sure ind_near_pol is defined.
                ## Assign class and other cols of large pol to small pol:
                gdf.at[ind_pol, class_col] = gdf_l.iloc[ind_nearest_pol][class_col]
                for col_name in other_cols:
                    gdf.at[ind_pol, col_name] = gdf_l.iloc[ind_nearest_pol][col_name]
            
        ## Dissolve all polygons with same class and explode multipols:
        gdf = gdf.dissolve(by=class_col, as_index=False)  # this takes most time.
        gdf = gdf.explode().reset_index(drop=True)
        gdf = gdf.assign(area=gdf['geometry'].area)

        n_pols_end_loop = len(gdf)
        if verbose > 0:
            print(f'Number of new polygons: {n_pols_end_loop}, number of old polygons: {n_pols_start_loop}')
        gdf['polygon_id_in_tile'] = gdf.index

         ## Stop conditions: 1) no more small pols, 2) no more changes, 3) max number of iterations
        if len(gdf[gdf['area'] < area_threshold]) == 0:
            if verbose > 0:
                print(f'SUCCESS: No more polygons smaller than {area_threshold}. Stopping.')
            continue_dissolving = False
        current_it += 1
        if current_it >= max_it:
            if verbose > 0:
                print(f'Warning: maximum number of iterations ({max_it}) reached. Stopping.')
            continue_dissolving = False
        if sort_ascending is False:
            if verbose > 0:
                print(f'Have tried sorting other way once. Stopping.')
            continue_dissolving = False
        if n_pols_end_loop == n_pols_start_loop:
            if verbose > 0:
                print(f'No more changes. Trying to sort other way around.')
            sort_ascending = False  # if no more changes, but still small pols left, then sort by descending area to dissolve into the largest polygons first, to resolve cases where they touch by a single pixel

    return gdf

def override_predictions_with_manual_layer(filepath_manual_layer='/home/tplas/data/gis/tmp_fgh_layer/tmp_fgh_layer.shp', 
                                           tile_predictions_folder='/home/tplas/predictions/predictions_LCU_2022-11-30-1205_dissolved1000m2/', 
                                           new_tile_predictions_override_folder=None, verbose=0):
    ## Load FGH layer & get list of tile paths, and new directory for tile predictions with FGH override
    df_fgh = gpd.read_file(filepath_manual_layer)
    date_fgh_modified = str(datetime.datetime.strptime(time.ctime(os.path.getmtime(filepath_manual_layer)), '%a %b %d %H:%M:%S %Y').date())
    df_fgh['source'] = f'OS NGD retrieved {date_fgh_modified}'  # set source OS NGD 

    if new_tile_predictions_override_folder is None:
        new_tile_predictions_override_folder = tile_predictions_folder.rstrip('/') + '_FGH-override/'
    if not os.path.exists(new_tile_predictions_override_folder):
        os.mkdir(new_tile_predictions_override_folder)

    ## Collect all filepaths:
    subdirs_tiles = [os.path.join(tile_predictions_folder, x) for x in os.listdir(tile_predictions_folder) if os.path.isdir(os.path.join(tile_predictions_folder, x))]
    dict_shp_files = {} 
    dict_new_shp_files = {}
    for tile_dir in subdirs_tiles:
        ## Get tile name
        tilename = tile_dir.split('/')[-1].split('_')[2]
        assert len(tilename) == 6

        ## Get path to shp file
        tmp_list_shp_files = [os.path.join(tile_dir, x) for x in os.listdir(tile_dir) if x[-4:] == '.shp']
        assert len(tmp_list_shp_files) == 1 
        dict_shp_files[tilename] = tmp_list_shp_files[0]

        ## Set path for new shp file to be saved
        new_dir = os.path.join(new_tile_predictions_override_folder, tile_dir.split('/')[-1] + '_FGH-override')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        ## Load shp file to see how many pols there are (takes some extra time)
        dict_new_shp_files[tilename] = os.path.join(new_dir, new_dir.split('/')[-1] + '.shp')
        tmp_pols = load_pols(dict_shp_files[tilename])
        if len(tmp_pols) > 20:
            print(f'Loaded {len(tmp_pols)} polygons for tile {tilename}')

    ## Loop over tiles and apply FGH override
    mapping_dict = {'Wood and Forest Land': 'C', 'Moor and Heath Land': 'D', 
                    'Agro-Pastoral Land': 'E', 'NO CLASS': 'I'}
    for tilename, tile_pred_path in tqdm(dict_shp_files.items()):
        if verbose > 0:
            print(tilename)
        df_pred = gpd.read_file(tile_pred_path)
        df_pred = df_pred.copy()
        ## Get rid of columns that are not needed:
        df_pred = df_pred.drop(['class', 'area'], axis=1)
        if len(df_pred) == 1:  # just to verify that what's happening is what I think is happening
            assert 'polygon_id' not in df_pred.columns
        else:
            df_pred = df_pred.drop(['polygon_id'], axis=1)
        df_pred['lc_label'] = df_pred['Class name'].map(mapping_dict)
        df_pred = df_pred.drop(['Class name'], axis=1)
        df_pred['source'] = 'model prediction'  # set source 
    
        ## Merge with FGH layer:
        df_diff = gpd.overlay(df_pred, df_fgh, how='difference')  # Get df pred polygons that are not in df fgh 
        df_diff = df_diff.explode().reset_index(drop=True)
        df_intersect = gpd.overlay(df_pred, df_fgh, how='intersection')  # Get overlap between df pred and df fgh
        df_intersect['lc_label'] = df_intersect['lc_label_2']  #  FGH layer has priority (and was 2nd arg in overlay() above)
        df_intersect['source'] = df_intersect['source_2']
        df_intersect = df_intersect.drop(['lc_label_1', 'lc_label_2'], axis=1)
        df_intersect = df_intersect.drop(['source_1', 'source_2'], axis=1)
        df_intersect = df_intersect.explode().reset_index(drop=True)  # in case multiple polygons are created by intersection
        df_new = gpd.GeoDataFrame(pd.concat([df_diff, df_intersect], ignore_index=True))  # Concatenate all polygons
        df_new = add_main_category_index_column(df_lc=df_new, col_code_name='lc_label',
                                                    col_ind_name='class')  # add numeric main label column
        df_new.crs = df_pred.crs  # set crs

        ## Dissolve again because intersections can break up FGH polygons, which then need to be dissolved again
        df_new = df_new.dissolve(by='lc_label', as_index=False)  
        df_new = df_new.explode().reset_index(drop=True)
        
        ## Save:
        new_tile_path = dict_new_shp_files[tilename]
        df_new.to_file(new_tile_path)
    
    print('\n#####\n\nDone with FGH override\n\n#####\n')
    return new_tile_predictions_override_folder

def merge_individual_shp_files(dir_indiv_tile_shp, save_merged_shp_file=True):

    subdirs_tiles = [os.path.join(dir_indiv_tile_shp, x) for x in os.listdir(dir_indiv_tile_shp) if os.path.isdir(os.path.join(dir_indiv_tile_shp, x))]
    for i_tile, pred_dir in tqdm(enumerate(subdirs_tiles)):
        if i_tile == 0:
            df_all = load_pols(pred_dir)
        else:
            df_tmp = load_pols(pred_dir)
            df_all = pd.concat([df_all, df_tmp], ignore_index=True)
    
    if save_merged_shp_file:
        dir_path_merged = os.path.join(dir_indiv_tile_shp, 'merged_tiles')
        if not os.path.exists(dir_path_merged):
            os.mkdir(dir_path_merged)
        df_all.to_file(os.path.join(dir_path_merged, 'merged_tiles.shp'))

    return df_all