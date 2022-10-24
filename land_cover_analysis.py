import os, sys, copy
import numpy as np
import rasterio
import xarray as xr
import rioxarray as rxr
import sklearn.cluster
from tqdm import tqdm
import shapely as shp
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
import gdal, osr
import loadpaths


def assert_epsg(epsg, project_epsg=27700):
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

def load_coords_from_geotiff(tiff_file_path):
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
    
    square_coords = shp.geometry.Polygon(zip([x_left, x_right, x_right, x_left], [y_bottom, y_bottom, y_top, y_top]))
    df_tile = gpd.GeoDataFrame(crs=f'epsg:{raster_epsg}', geometry=[square_coords])
    df_tile = df_tile.assign(name=tiff_file_path.split('/')[-1].rstrip('.tif'))  # could assign index with index=[0] 
    
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

    return df_tile

def load_geo_tiff(tiff_file_path, datatype='np', verbose=0):
    ## image
    im = load_tiff(tiff_file_path=tiff_file_path, datatype=datatype, verbose=verbose)

    # coords 
    df_tile = load_coords_from_geotiff(tiff_file_path=tiff_file_path)

    return im, df_tile

def load_pols(pol_path):
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

def get_pols_for_tiles(df_pols, df_tiles):
    '''Assuming a df for tiles currently.. Could just use Polygon geometry instead'''

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
                                resolution=(0.125, -0.125),
                                save_raster=False, filename='mask.tif',
                                plot_raster=False):

    ## Convert shape to raster:
    cube = make_geocube(df_shp, measurements=[col_name],
                    # like=ex_tile)  # use resolution of example tiff
                    resolution=resolution)

    if save_raster:
        filepath = filename
        cube[col_name].rio.to_raster(filepath)
        print(f'Saved to {filepath}')

    if plot_raster:
        cube[col_name].plot()

    return cube 
