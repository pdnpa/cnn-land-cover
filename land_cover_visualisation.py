import os, sys, copy
import random, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors 
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
import seaborn as sns
import rasterio, rasterio.plot
import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import torch
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

def create_lc_cmap(lc_class_name_list, unique_labels_array):
    '''Create custom colormap of LC classes, based on list of names given.'''
    lc_colours_list = [lc_colour_mapping_names[xx] for xx in lc_class_name_list]  # get list of colours based on class names
    lc_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('LC classes', colors=lc_colours_list, 
                                                                  N=len(lc_colours_list))  # create qualitative cmap of colour lists
    formatter = plt.FuncFormatter(lambda val, loc: f'{val} ({unique_labels_array[val]}): {dict_ind_to_name[unique_labels_array[val]]}')  # create formatter for ticks/ticklabels of cbar
    ticks = np.arange(len(lc_colours_list))

    return lc_cmap, formatter, ticks 

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

def plot_landcover_image(im, lc_class_name_list=[], unique_labels_array=None, ax=None, 
                         plot_colorbar=True, cax=None):
    '''Plot LC as raster. Give lc_class_name_list and unique_labels_array to create 
    color legend of classes'''
    if ax is None:
        ax = plt.subplot(111)

    # unique_labels = np.unique(im).astype('int')
    # present_class_names = [lc_class_name_list[lab] for lab in unique_labels]
    # lc_cmap, formatter, cbar_ticks = create_lc_cmap(lc_class_name_list=present_class_names)
    lc_cmap, formatter, cbar_ticks = create_lc_cmap(lc_class_name_list=lc_class_name_list, 
                                                    unique_labels_array=unique_labels_array)  # get cbar specifics for LC classes

    im_plot = ax.imshow(im, cmap=lc_cmap, vmin=- 0.5, vmax=len(lc_class_name_list) - 0.5)  # set min and max to absolute number of classes. Hence this ONLY works with adjacent classes.
    if plot_colorbar:
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.04)
        plt.colorbar(im_plot, format=formatter, ticks=cbar_ticks, cax=cax)
    naked(ax)

def plot_image_mask_pred(image, mask, pred, lc_class_name_list=[], unique_labels_array=None, 
                         ax_list=None, plot_colorbar=True, cax=None):
    '''Plot image/mask/prediction next to each other & cbar if given'''
    ## check that ax_list is of correct format:
    create_new_ax = True
    if ax_list is not None:
        if (type(ax_list) == list or type(ax_list) == np.ndarray) and (len(ax_list) == 3 or len(ax_list) == 4):
            create_new_ax = False 
        else:
            print(f'ax_list is type {type(ax_list)} of len {len(ax_list)}')

    if create_new_ax:
        fig, ax_list = plt.subplot(1, 3, figsize=(6, 20))

    ## Use specific functions for each:
    plot_image_simple(im=image, ax=ax_list[0])
    plot_landcover_image(im=mask, ax=ax_list[1], lc_class_name_list=lc_class_name_list, 
                         unique_labels_array=unique_labels_array, plot_colorbar=False)
    plot_landcover_image(im=pred, ax=ax_list[2], lc_class_name_list=lc_class_name_list, 
                        unique_labels_array=unique_labels_array, 
                        plot_colorbar=plot_colorbar, cax=cax)
    
    return ax_list 

def plot_image_mask_pred_from_all(all_ims, all_masks, all_preds, preprocessing_fun=None, ind_list=[0],
                                  lc_class_name_list=[], unique_labels_array=None, save_fig=False, 
                                  filename_prefix='example_predictions'):
    '''Plot rows of image/mask/prediction + legend.'''
    assert type(ind_list) == list
    ind_list = np.sort(np.array(ind_list))
    assert all_ims.ndim == 4 and all_masks.ndim == 3, 'images and masks dont have expected shape'
    assert all_preds.ndim == 3, 'predicted masks dont have expected shape. Maybe they are not yet argmaxed?'
    assert all_ims.shape[0] == all_masks.shape[0] and all_ims.shape[0] == all_preds.shape[0]
    assert all_ims.shape[-2:] == all_masks.shape[-2:] and all_ims.shape[-2:] == all_preds.shape[-2:]

    ## Select images to be plotted:
    ims_plot = all_ims[ind_list, :, :, :]
    masks_plot = all_masks[ind_list, :, :]
    preds_plot = all_preds[ind_list, :, :]
    assert ims_plot.ndim == 4 and masks_plot.ndim == 3 and preds_plot.ndim == 3
    if preprocessing_fun is None:
        print('WARNING: no preprocessing (eg z-scoring) can be undone because no preprocessing function passed on')
    else:
        ## undo preprocessing of image so the true RGB image is shown again:
        ims_plot = lca.undo_zscore_single_image(im_ds=ims_plot, f_preprocess=preprocessing_fun)

    if type(ims_plot) == torch.Tensor:
        ims_plot = ims_plot.detach().numpy()
    if type(masks_plot) == torch.Tensor:
        masks_plot = masks_plot.detach().numpy()
    if type(preds_plot) == torch.Tensor:
        preds_plot = preds_plot.detach().numpy()

    ## Create figure and ax handles:
    n_pics = len(ind_list)
    fig = plt.figure(constrained_layout=False, figsize=(7, n_pics * 2))
    gs_ims = fig.add_gridspec(ncols=3, nrows=n_pics, bottom=0.02, top=0.95, 
                              left=0.02, right=0.8, wspace=0.15, hspace=0.15)
    figsize = fig.get_size_inches()
    ideal_legend_height_inch = len(lc_class_name_list) * 0.4
    ideal_legend_height_fraction = ideal_legend_height_inch / figsize[1]
    legend_height = np.minimum(ideal_legend_height_fraction + 0.02, 0.95)
    gs_cbar = fig.add_gridspec(ncols=1, nrows=1, top=legend_height, bottom=0.02, 
                               left=0.82, right=0.835)
    ax_ims = {}
    ax_cbar = fig.add_subplot(gs_cbar[0])

    ## Plot using specific function, for each row:
    for i_ind in range(n_pics):
        ax_ims[i_ind] = [fig.add_subplot(gs_ims[i_ind, xx]) for xx in range(3)]
        plot_image_mask_pred(image=ims_plot[i_ind, :, :, :], mask=masks_plot[i_ind, :, :],
                             pred=preds_plot[i_ind, :, :], ax_list=ax_ims[i_ind],
                             lc_class_name_list=lc_class_name_list, unique_labels_array=unique_labels_array,
                             plot_colorbar=(i_ind == 0), cax=ax_cbar)

        if i_ind == 0:
            ax_ims[i_ind][0].set_title('Image')
            ax_ims[i_ind][1].set_title('Land cover 80s')
            ax_ims[i_ind][2].set_title('Model prediction')

    if save_fig:
        str_list_inds = '-'.join([str(x) for x in ind_list])
        filename = f'figures/lc_predictions/{filename_prefix}_{str_list_inds}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')


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
        
def plot_comparison_class_balance_train_test(train_patches_mask, test_patches_mask, ax=None, names_classes=None):
    '''Get & plot distribution of LC classes for both train and test patches (pixel wise). '''
    ## Get counts: (can take some time)
    class_ind_train, freq_train = lca.get_distr_classes_from_patches(patches_mask=train_patches_mask)
    class_ind_test, freq_test = lca.get_distr_classes_from_patches(patches_mask=test_patches_mask)

    ## In case one data set contains classes the other doesn't, add 0-count to the other:
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
    ## Sort again by class ind:
    arg_train = np.argsort(class_ind_train)
    arg_test = np.argsort(class_ind_test)
    class_ind_test = class_ind_test[arg_test]
    freq_test = freq_test[arg_test]
    class_ind_train = class_ind_train[arg_train]
    freq_train = freq_train[arg_train]

    ## Get inds, names and density:
    assert (class_ind_train == class_ind_test).all(), 'there is a unique class in either one of the splits => build in a way to accomodate this by adding a 0 count'
    inds_classes = class_ind_test  # assuming they are equal given the assert bove
    if names_classes is None:
        names_classes = [dict_ind_to_name[x] for x in inds_classes]
    n_classes = len(inds_classes)

    bar_locs = np.arange(len(inds_classes))
    dens_train = freq_train / freq_train.sum() 
    dens_test = freq_test / freq_test.sum()

    ## Plot bar hist of density:
    if ax is None:
        ax = plt.subplot(111)
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

    return (class_ind_train, freq_train), (class_ind_test, freq_test), (inds_classes, names_classes, n_classes)

def plot_distr_classes_from_shape(df_lc, ax=None):

    if ax is None:
        ax = plt.subplot(111)

    class_label_col = 'LC_D_80'
    unique_classes = df_lc[class_label_col].unique() 
    area_classes = np.zeros(len(unique_classes))

    for i_c, c_n in enumerate(unique_classes):
        tmp_df = df_lc[df_lc[class_label_col] == c_n]
        area_classes[i_c] = tmp_df['AREA'].sum()

    sort_classes = np.argsort(area_classes)
    area_classes = area_classes[sort_classes]
    unique_classes = unique_classes[sort_classes]

    area_classes = area_classes / np.sum(area_classes)

    bar_locs = np.arange(len(unique_classes))
    ax.bar(x=bar_locs, width=0.8, height=area_classes, facecolor='k')
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(unique_classes, rotation=90)
    ax.set_ylabel('Area (fraction)')
    despine(ax)
    return ax, (area_classes, unique_classes)