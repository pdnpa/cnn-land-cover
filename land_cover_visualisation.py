import os, sys
import random, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors 
import matplotlib.patches as mpatches
from cycler import cycler
import seaborn as sns
import rasterio, rasterio.plot
import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import land_cover_analysis as lca

## Set default settings.
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)
color_dict_stand[10] = '#994F00'
color_dict_stand[11] = '#4B0092'

## Retrieve LC class specific colour mappings:
with open('lc_colour_mapping.json', 'r') as f:
    lc_colour_mapping_inds = json.load(f, object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})  # mapping from class ind to colour hex

dict_ind_to_name, dict_name_to_ind = lca.get_lc_mapping_inds_names_dicts()
lc_colour_mapping_names = {dict_ind_to_name[k]: v for k, v in lc_colour_mapping_inds.items() if k in dict_ind_to_name.keys()}

def generate_list_random_colours(n):
    list_colours =  ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
    """
    ## Load or save a dict of colours by:
    tmp = lcv.generate_list_random_colours(40)
    color_dict = {x: tmp[x] for x in range(40)}
    with open('lc_color_mapping.json', 'w') as f:
        json.dump(color_dict, f, indent=4)
    """
    return list_colours

## Some generic plotting utility functions:
def despine(ax):
    '''Remove top and right spine'''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def naked(ax):
    '''Remove all spines, ticks and labels'''
    for ax_name in ['top', 'bottom', 'right', 'left']:
        ax.spines[ax_name].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

def set_fontsize(font_size=12):
    '''Set font size everywhere in mpl'''
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.autolimit_mode'] = 'data' # default: 'data'
    params = {'legend.fontsize': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size}
    plt.rcParams.update(params)
    print(f'Font size is set to {font_size}')

def equal_xy_lims(ax, start_zero=False):
    '''Set x-axis lims equal to y-axis lims'''
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    max_outer_lim = np.maximum(xlims[1], ylims[1])
    min_inner_lim = np.minimum(xlims[0], ylims[0])

    if start_zero:
        ax.set_xlim([0, max_outer_lim])
        ax.set_ylim([0, max_outer_lim])
    else:
        ax.set_xlim([min_inner_lim, max_outer_lim])
        ax.set_ylim([min_inner_lim, max_outer_lim])

def equal_lims_two_axs(ax1, ax2):
    '''Set limits of two ax elements equal'''
    xlim_1 = ax1.get_xlim()
    xlim_2 = ax2.get_xlim()
    ylim_1 = ax1.get_ylim()
    ylim_2 = ax2.get_ylim()
     
    new_x_min = np.minimum(xlim_1[0], xlim_2[0])
    new_x_max = np.maximum(xlim_1[1], xlim_2[1])
    new_y_min = np.minimum(ylim_1[0], ylim_2[0])
    new_y_max = np.maximum(ylim_1[1], ylim_2[1])

    ax1.set_xlim([new_x_min, new_x_max])
    ax2.set_xlim([new_x_min, new_x_max])
    ax1.set_ylim([new_y_min, new_y_max])
    ax2.set_ylim([new_y_min, new_y_max])

def remove_xticklabels(ax):  # remove labels but keep ticks
    ax.set_xticklabels(['' for x in ax.get_xticklabels()])

def remove_yticklabels(ax):  # remove labels but keep ticks
    ax.set_yticklabels(['' for x in ax.get_yticklabels()])

def remove_both_ticklabels(ax):  # remove labels but keep ticks
    remove_xticklabels(ax)
    remove_yticklabels(ax)

## Plotting images:
def plot_image_simple(im, ax=None, name_file=None):
    '''Plot image (as np array or xr DataArray)'''
    if ax is None:
        ax = plt.subplot(111)
    if type(im) == xr.DataArray:
        plot_im = im.to_numpy()
    else:
        plot_im = im
    rasterio.plot.show(plot_im, ax=ax, cmap='viridis')
    naked(ax)
    if name_file is None:
        pass 
    else:
        name_tile = name_file.split('/')[-1].rstrip('.tif')
        ax.set_title(name_tile)

def plot_lc_from_gdf_dict(df_pols_tiles, tile_name='SK0066', 
                          col_name='LC_D_80', ax=None, leg_box=(-.1, 1.05)):
    '''Plot LC polygons'''
    if ax is None:
        ax = plt.subplot(111)

    df_tile = df_pols_tiles[tile_name]
    list_colours_tile = [lc_colour_mapping_names[name] for name in df_tile['LC_D_80']]
    
    ax = df_tile.plot(legend=True, linewidth=0.4, ax=ax,
                      color=list_colours_tile, edgecolor='k')

    ## Create legend:
    list_patches = []
    for i_class, class_name in enumerate(df_tile[col_name].unique()):
        leg_patch = mpatches.Patch(facecolor=lc_colour_mapping_names[class_name],
                                edgecolor='k',linewidth=0.4,
                                label=class_name)
        list_patches.append(leg_patch)

    ax.legend(handles=list_patches,  
        title="Legend", bbox_to_anchor=leg_box,
        fontsize=10, frameon=False, ncol=1)

    naked(ax)
    ax.set_title(f'Land cover of tile {tile_name}')
    return ax
        
def plot_comparison_class_balance_train_test(train_patches_mask, test_patches_mask, ax=None):

    ## Get counts: (can take some time)
    class_ind_train, freq_train = lca.get_distr_classes_from_patches(patches_mask=train_patches_mask)
    class_ind_test, freq_test = lca.get_distr_classes_from_patches(patches_mask=test_patches_mask)

    # print(class_ind_train, class_ind_test)
    # print(freq_train, freq_test)
    for c_train in class_ind_train:
        if c_train in class_ind_test:
            continue 
        else:
            class_ind_test = np.concatenate((class_ind_test, [c_train]), axis=0)
            freq_test = np.concatenate((freq_test, [0]), axis=0)

    for c_test in class_ind_test:
        if c_test in class_ind_train:
            continue 
        else:
            class_ind_train = np.concatenate((class_ind_train, [c_test]), axis=0)
            freq_train = np.concatenate((freq_train, [0]), axis=0)

    arg_train = np.argsort(class_ind_train)
    arg_test = np.argsort(class_ind_test)
    class_ind_test = class_ind_test[arg_test]
    freq_test = freq_test[arg_test]
    class_ind_train = class_ind_train[arg_train]
    freq_train = freq_train[arg_train]

    # print(class_ind_train, class_ind_test)
    # print(freq_train, freq_test)
    assert (class_ind_train == class_ind_test).all(), 'there is a unique class in either one of the splits => build in a way to accomodate this by adding a 0 count'
    inds_classes = class_ind_test  # assuming they are equal given the assert bove
    names_classes = [dict_ind_to_name[x] for x in inds_classes]

    if ax is None:
        ax = plt.subplot(111)
    
    bar_locs = np.arange(len(inds_classes))
    dens_train = freq_train / freq_train.sum() 
    dens_test = freq_test / freq_test.sum()

    ax.bar(x=bar_locs - 0.2, height=dens_train, 
           width=0.4, label='Train')
    ax.bar(x=bar_locs + 0.2, height=dens_test, 
           width=0.4, label='Test')

    ax.legend(frameon=False)
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(names_classes, rotation=90)
    ax.set_xlabel('LC classes')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of LC classes in train and test set', fontdict={'weight': 'bold'})
    despine(ax)



