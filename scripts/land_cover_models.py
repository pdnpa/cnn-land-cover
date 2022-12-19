from json import encoder
import os, sys, copy, shutil
import numpy as np
from tqdm import tqdm
import datetime
import pickle
import pandas as pd
import loadpaths
import pandas as pd 
import geopandas as gpd
import rasterio
import rasterio.features
import shapely.geometry
import patchify 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import custom_losses as cl


path_dict = loadpaths.loadpaths()

class DataSetPatches(torch.utils.data.Dataset):
    '''Data set for images & masks. Saves file paths, but only loads into memory during __getitem__.
    
    Used for training etc - __getitem__ has expected output (input, output) for PL models.
    '''
    def __init__(self, im_dir, mask_dir, mask_suffix='_lc_80s_mask.npy', list_tile_names=None,
                 preprocessing_func=None, unique_labels_arr=None, shuffle_order_patches=True,
                 subsample_patches=False, frac_subsample=1, relabel_masks=True,
                 path_mapping_dict='/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__main_categories__2022-11-17-1512.pkl'):
        super(DataSetPatches, self).__init__()
        self.im_dir = im_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.preprocessing_func = preprocessing_func
        self.path_mapping_dict = path_mapping_dict
        self.shuffle_order_patches = shuffle_order_patches
        self.frac_subsample = frac_subsample
        self.relabel_masks = relabel_masks
        self.subsample_patches = subsample_patches
        self.unique_labels_arr = unique_labels_arr
        self.list_tile_names = list_tile_names

        if self.preprocessing_func is not None:  # prep preprocess transformation
            rgb_means = self.preprocessing_func.keywords['mean']
            rgb_std = self.preprocessing_func.keywords['std']

            rgb_means = torch.tensor(np.array(rgb_means)[:, None, None])  # get into right dimensions
            rgb_std = torch.tensor(np.array(rgb_std)[:, None, None])  # get into right dimensions

            dtype = torch.float32

            ## Change to consistent dtype:
            self.rgb_means = rgb_means.type(dtype)
            self.rgb_std = rgb_std.type(dtype)

            self.preprocess_image = self.zscore_image 
        else:
            print('WARNING: no normalisation will be applied when loading images')
            self.preprocess_image = self.pass_image

        ## create data frame or something of all files 
        if type(self.im_dir) == list:
            self.multiple_im_dirs = True
            self.list_im_npys = []
            print(f'Multiple ({len(self.im_dir)}) image directories provided. Will concatenate all patches together.')
        elif type(self.im_dir) == str:
            self.multiple_im_dirs = False

        if list_tile_names is None:
            if self.multiple_im_dirs is False:
                self.list_im_npys = [os.path.join(im_dir, x) for x in os.listdir(im_dir)]
            else:
                for curr_dir in im_dir:
                    self.list_im_npys += [os.path.join(curr_dir, x) for x in os.listdir(curr_dir)]
        else:
            assert type(list_tile_names) == list
            print(f'Only using patches that are in tile list (of length {len(list_tile_names)}).')
            if self.multiple_im_dirs is False:
                self.list_im_npys = [os.path.join(im_dir, x) for x in os.listdir(im_dir) if x[:6] in list_tile_names]
            else:
                for curr_dir in im_dir:
                    self.list_im_npys += [os.path.join(curr_dir, x) for x in os.listdir(curr_dir) if x[:6] in list_tile_names]  
                self.list_im_npys = [os.path.join(im_dir, x) for x in os.listdir(im_dir) if x[:6] in list_tile_names]
        self.list_patch_names = [x.split('/')[-1].rstrip('.npy') for x in self.list_im_npys]
        if mask_dir is None:
            print('No mask directory provided. Will use image parent directory instead.')
            self.list_mask_npys = [x.replace('/images/', '/masks/').replace('.npy', mask_suffix) for x in self.list_im_npys]
        else:
            self.list_mask_npys = [os.path.join(mask_dir, x.split('/')[-1].rstrip('.npy') + mask_suffix) for x in self.list_im_npys]

        self.create_df_patches()
        self.organise_df_patches()
        print(f'Loaded {len(self.df_patches)} patches')
        self.create_label_mapping()        

    def __getitem__(self, index):
        '''Function that gets data items by index. I have added timings in case this should be sped up.'''
        patch_row = self.df_patches.iloc[index]
        im = np.load(patch_row['im_filepath'])  #0.25ms
        mask = np.load(patch_row['mask_filepath'])  #0.25ms
        im = torch.tensor(im).float()  # 0.1ms
        im = self.preprocess_image(im)  # 0.25 ms
        if self.relabel_masks:
            mask = self.remap_labels(mask)  # 2.5ms
        mask = torch.tensor(mask).type(torch.LongTensor)  #0.1ms
        return im, mask 

    def __repr__(self):
        return f'DataLoaderPatches class'

    def __len__(self):
        return len(self.df_patches)

    def create_df_patches(self):
        '''Create dataframe with all patch locations'''
        self.df_patches = pd.DataFrame({'patch_name': self.list_patch_names,
                                        'im_filepath': self.list_im_npys, 
                                        'mask_filepath': self.list_mask_npys})

    def organise_df_patches(self):
        '''Subsample & sort/shuffle patches DF'''
        if self.subsample_patches:
            assert self.frac_subsample <= 1 and self.frac_subsample > 0
            n_subsample = int(len(self.df_patches) * self.frac_subsample)
            print(f'Subsampling {n_subsample} patches')
            self.df_patches = self.df_patches[:n_subsample]
        
        if self.shuffle_order_patches:
            print('Patches ordered randomly')
            self.df_patches = self.df_patches.sample(frac=1, replace=False)
        else:
            print('Patches sorted by tile/patch order')
            self.df_patches = self.df_patches.sort_values('patch_name')
        self.df_patches = self.df_patches.reset_index(drop=True)
            
    def create_label_mapping(self):
        '''Prep the transformation of class inds'''
        if self.path_mapping_dict is None:
            print('WARNING: no label mapping given - so using all labels individually')
            dict_ind_to_name, dict_name_to_ind = lca.get_lc_mapping_inds_names_dicts() 
            if self.unique_labels_arr == None:  # if no array given, presume full array:
                self.unique_labels_arr = np.unique(np.array(list(dict_ind_to_name.keys())))  # unique sorts too 
            else:
                self.unique_labels_arr = np.unique(self.unique_labels_arr)
            self.mapping_label_to_new_dict = {label: ind for ind, label in enumerate(self.unique_labels_arr)}
            self.class_name_list = [dict_ind_to_name[label] for label in self.unique_labels_arr]
            self.n_classes = len(self.class_name_list)
        else:
            self.dict_mapping = pickle.load(open(self.path_mapping_dict, 'rb'))           
            if self.relabel_masks:  # normal loading of dict_mapping:
                print(f'Loaded {self.path_mapping_dict.split("/")[-1]} to map labels')
                self.mapping_label_to_new_dict = self.dict_mapping['dict_label_mapping']
                self.unique_labels_arr = np.array(list(self.dict_mapping['dict_label_mapping'].keys()))
            else:  # don't remap, but just changing these two objects for consistency.
                ## Because there is no remapping, it is just identity transformation. 
                ## Assume the given path_mapping_dict output is thus already input (hence no remapping required)
                print(f'Loaded {self.path_mapping_dict.split("/")[-1]} just for meta data. Will not remap labels.')
                self.unique_labels_arr = np.unique(list(self.dict_mapping['dict_label_mapping'].values())) 
                self.mapping_label_to_new_dict = {k: k for k in self.unique_labels_arr}
            self.class_name_list = list(self.dict_mapping['dict_new_names'].values())
            self.n_classes = len(self.class_name_list)

    def zscore_image(self, im):
        '''Apply preprocessing function to a single image. 
        Adapted from lca.apply_zscore_preprocess_images() but more specific/faster.
        
        What would be much faster, would be to store the data already pre-processed, but
        what function to use depends on the Network.'''
        im = im / 255 
        im = (im - self.rgb_means) / self.rgb_std
        return im
 
    def pass_image(self, im):
        return im

    def remap_labels(self, mask):
        '''Remapping labels to consecutive labels. Is quite slow, so would be better to store mask as 
        remapped already.. But can depend on how much data is used; although for the entire data set
        it is known of course. '''
        new_mask = np.zeros_like(mask)  # takes up more RAM (instead of reassigning mask_patches.. But want to make sure there are no errors when changing labels). Although maybe it's okay because with labels >= 0 you're always changing down so no chance of getting doubles I think.
        for label in self.unique_labels_arr:
            new_mask[mask == label] = self.mapping_label_to_new_dict[label]
        return new_mask

class DataSetPatchesTwoMasks(DataSetPatches):
    '''DS class that holds another set of masks (of the same images).
    Useful for plotting both mask sets; but generally not compatible with PL trainer API.
    '''
    def __init__(self, mask_dir_2, mask_suffix_2='_lc_2022_mask.npy',
                 mask_1_name='LC 80s', mask_2_name='LC 2020', 
                 relabel_masks_2=False, **kwargs):
        self.mask_dir_2 = mask_dir_2
        self.mask_suffix_2 = mask_suffix_2
        self.relabel_masks_2 = relabel_masks_2
        self.mask_1_name = mask_1_name
        self.mask_2_name = mask_2_name
        
        super().__init__(**kwargs)
        
        assert 'mask_2_filepath' in self.df_patches.columns  # just double checking that create_df_patches() has been correctly overridden.
        print(f'Relabelling mask 1: {self.relabel_masks}\nRelabelling mask 2: {self.relabel_masks_2}')

    def __getitem__(self, index):
        '''Function that gets data items by index.
        Override to include mask_2'''
        patch_row = self.df_patches.iloc[index]
        im = np.load(patch_row['im_filepath'])  
        mask = np.load(patch_row['mask_filepath'])  
        mask_2 = np.load(patch_row['mask_2_filepath'])
        im = torch.tensor(im).float() 
        im = self.preprocess_image(im)  
        if self.relabel_masks:
            mask = self.remap_labels(mask) 
        if self.relabel_masks_2:
            mask_2 = self.remap_labels(mask_2)
        
        mask = torch.tensor(mask).type(torch.LongTensor)
        mask_2 = torch.tensor(mask_2).type(torch.LongTensor)
        return im, mask, mask_2

    def create_df_patches(self):
        '''Override to include mask_2_filepath'''
        self.list_mask_2_npys = [os.path.join(self.mask_dir_2, x.split('/')[-1].rstrip('.npy') + self.mask_suffix_2) for x in self.list_im_npys]
        self.df_patches = pd.DataFrame({'patch_name': self.list_patch_names,
                                        'im_filepath': self.list_im_npys, 
                                        'mask_filepath': self.list_mask_npys,
                                        'mask_2_filepath': self.list_mask_2_npys})


class LandCoverUNet(pl.LightningModule):
    '''
    UNet for semantic segmentation. Build using API of pytorch lightning
    (see: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
    
    
    '''
    def __init__(self, n_classes=10, encoder_name='resnet50', pretrained='imagenet',
                 lr=1e-3, loss_function='cross_entropy', skip_factor_eval=1):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.skip_factor_eval = skip_factor_eval
        # pl.seed_everything(7)

        ## Use SMP Unet as base model. This PL class essentially just wraps around that:
        ## https://readthedocs.org/projects/segmentation-modelspytorch/downloads/pdf/latest/
        self.base = smp.Unet(encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights=pretrained,     # use `imagenet` pre-trained weights for encoder initialization or None
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            classes=n_classes,                      # model output channels (number of classes in your dataset)
                            activation='softmax')  # activation function to apply after final convolution; One of [sigmoid, softmax, logsoftmax, identity, callable, None]

        ## Define the preprocessing function that the data needs to be applied to
        self.preprocessing_func = smp.encoders.get_preprocessing_fn(encoder_name, pretrained=pretrained)

        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
        # self.focal_loss = cl.FocalLoss(gamma=0.75)
        self.focal_loss = cl.FocalLoss_2(gamma=0.75, reduction='mean', ignore_index=0)
        self.iou_loss = cl.mIoULoss(n_classes=n_classes)
        self.dice_loss = torchmetrics.Dice(num_classes=n_classes, ignore_index=0, requires_grad=True)#, average='macro')

        ## Define loss used for training:
        if loss_function == 'dummy':
            self.loss = self.dummy_loss
        elif loss_function == 'cross_entropy':
            self.loss = self.ce_loss  # reduction: 'none' (returns full-sized tensor), 'mean', 'sum'. Can also insert class weights and ignore indices
        elif loss_function == 'focal_loss':
            self.loss = self.focal_loss
        elif loss_function == 'iou_loss':
            self.loss = self.iou_loss
        elif loss_function == 'dice_loss':
            self.loss = self.dice_loss
        elif loss_function == 'focal_and_dice_loss':
            self.loss = lambda x, y: self.focal_loss(x, y) + self.dice_loss(x, y)
        else:
            assert False, f'Loss function {loss_function} not recognised.'
        print(f'{loss_function} loss is used.')
        self.calculate_test_confusion_mat = True
        self.test_confusion_mat = np.zeros((7, 7))
        # self.seg_val_metric = pl.metrics.Accuracy()  # https://devblog.pytorchlightning.ai/torchmetrics-pytorch-metrics-built-to-scale-7091b1bec919

        self.model_name = 'LCU (not saved)'
        self.description = 'LandCoverUNet class using FocalClass2'
        self.filename = None
        self.filepath = None

        ## Add info dict with some info: epochs, PL version, .. 
        self.dict_training_details = {}  # can be added post hoc once train dataset is defined

    def __str__(self):
        """Define name"""
        if hasattr(self, 'model_name'):
            return self.model_name
        else:
            return 'LCU model'

    def __repr__(self):
        if hasattr(self, 'model_name'):
            return f'Instance {self.model_name} of LandCoverUNet class'
        else:
            return f'Instance of LandCoverUNet class'

    def change_description(self, new_description='', add=False):
        '''Just used for keeping notes etc.'''
        if add:
            self.description = self.description + '\n' + new_description
        else:
            self.description = new_description

    def dummy_loss(self, y, output):
        '''Dummy function for loss'''
        assert y.shape == output.shape, f'y has shape {y.shape} but output has shape {output.shape}'
        return output

    def forward(self, x):
        '''By default, the predict_step() method runs the forward() method. In order to customize this behaviour, simply override the predict_step() method.'''
        assert type(x) == torch.Tensor, f'This is type {type(x)} with len {len(x)}. Type and shape of first {type(x[0])}, {x[0].shape} and 2nd: {type(x[1])}, {x[1].shape}'
        return self.base(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.base(x)
        loss = self.loss(output, y)
        self.log('train_loss', loss, on_epoch=True)
        # return {"loss": loss}
        return loss

    def training_step_end(self, training_step_outputs):
        '''Only adjust if you need outputs of different training steps'''
        return training_step_outputs

    # def training_epoch_end(self, outputs) -> None:
    #     '''Only adjust if you need outputs of different training steps'''
    #     torch.stack([x["loss"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):  # add dataloader id for separating test losss per test ds? 
        '''Done after training finished.'''
        x, y = batch
        output = self.base(x)
        if self.skip_factor_eval is not None:
            assert y.ndim == 3 and output.ndim == 4
            y = y[:, ::self.skip_factor_eval, ::self.skip_factor_eval]
            output = output[:, :, ::self.skip_factor_eval, ::self.skip_factor_eval]
        loss = self.loss(output, y)
        self.log('test_loss', loss)  # to be the same metric that it was trained on.. Maybe redundant? 
    
        self.log('test_ce_loss', self.ce_loss(output, y))
        self.log('test_focal_loss', self.focal_loss(output, y))
        self.log('test_iou_loss', self.iou_loss(output, y))
        ## TODO: accuracy per class, iou per clas, ... ? 

        if self.calculate_test_confusion_mat:
            det_output = lca.change_tensor_to_max_class_prediction(pred=output, expected_square_size=512 / self.skip_factor_eval)  # change soft maxed output to arg max
            assert det_output.shape == y.shape
            assert output.ndim == 4
            n_classes = output.shape[1]
            for ic_true in range(n_classes):
                for ic_pred in range(n_classes):
                    n_match = int((det_output[y == ic_true] == ic_pred).sum()) 
                    self.test_confusion_mat[ic_true, ic_pred] += n_match  # just add to existing matrix; so it can be done in batches
            overall_accuracy = self.test_confusion_mat.diagonal().sum() / self.test_confusion_mat.sum() 
            self.log('test_overall_accuracy', overall_accuracy)


    def validation_step(self, batch, batch_idx):
        '''Done during training (with unseen data), eg after each epoch.
        This is commonly a small portion of the train data. (eg 20%). '''
        x, y = batch
        output = self.base(x)
        loss = self.loss(output, y)

        self.log('val_loss', loss, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        '''Takes batches of (images, masks), like training etc. Then call forward and only give output.
        Output will be tensor of output masks.
        
        When calling this function with a DataLoader (trainer.predict(LCU, test_dl), then the output will be
        a list of batches'''
        if len(batch) == 1:
            x = batch[0]
        elif len(batch) == 2:
            x, y = batch
        output = self.base(x)
        
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # return {'optimizer': optimizer,
        #         'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
        #         'monitor': 'val_loss'}
        
    def save_model(self, folder='', verbose=1):
        '''Save model'''
        timestamp = lca.create_timestamp()
        self.filename = f'LCU_{timestamp}.data'
        self.model_name = f'LCU_{timestamp}'
        self.filepath = os.path.join(folder, self.filename)

        file_handle = open(self.filepath, 'wb')
        pickle.dump(self, file_handle)
        if verbose > 0:
            print(f'LCU model saved as {self.filename} at {self.filepath}')

def load_model(folder='', filename='', verbose=1):
    '''Load previously saved (pickled) LCU model'''
    with open(os.path.join(folder, filename), 'rb') as f:
        LCU = pickle.load(f)

    if verbose > 0:  # print some info
        print(f'Loaded {LCU}')
        for info_name in ['loss_function', 'n_max_epochs']:
            if info_name in LCU.dict_training_details.keys():
                print(f'{info_name} is {LCU.dict_training_details[info_name]}')
        if hasattr(LCU, 'description'):
            print(LCU.description)

    return LCU 

def get_batch_from_ds(ds, batch_size=5, start_ind=0):
    '''Given DS, retrieve a batch of data (for plotting etc)'''
    tmp_items = []
    names_patches = []
    assert type(batch_size) == int and type(start_ind) == int
    for ii in range(start_ind, start_ind + batch_size):
        tmp_items.append(ds[ii])
        names_patches.append(ds.df_patches.iloc[ii]['patch_name'])
        if len(ds[ii]) == 2:
            n_outputs = 2
        elif len(ds[ii]) == 3:
            n_outputs = 3
        else:
            assert False, f'DS has {len(ds[ii])} outputs. Expected 2 or 3'
    list_inputs = [torch.Tensor(x[0])[None, :, :, :] for x in tmp_items]
    list_outputs = [torch.Tensor(x[1])[None, :, :] for x in tmp_items]
    list_inputs = lca.concat_list_of_batches(list_inputs)
    list_outputs = lca.concat_list_of_batches(list_outputs)
    if n_outputs == 2:
        return [list_inputs, list_outputs], names_patches
    elif n_outputs == 3:
        list_outputs_2 = [torch.Tensor(x[2])[None, :, :] for x in tmp_items]
        list_outputs_2 = lca.concat_list_of_batches(list_outputs_2)
        return [list_inputs, list_outputs, list_outputs_2], names_patches

def predict_single_batch_from_testdl_or_batch(model, test_dl=None, batch=None, names_patches=None,
                                              plot_prediction=True, preprocessing_fun=None,
                                              lc_class_name_list=None, unique_labels_array=None):
    '''Predict LC of a single batch, and plot if wanted'''
    if batch is None and test_dl is not None:
        batch = next(iter(test_dl))
    elif batch is not None and test_dl is None:
        pass  # use batch 
    assert (len(batch) == 2 or len(batch) == 3) and type(batch) == list, 'batch of unexpected format.. Expected [batch_images, batch_masks]'
    predicted_labels = model.forward(batch[0])
    predicted_labels = lca.change_tensor_to_max_class_prediction(pred=predicted_labels)
    if len(batch) == 3:
        mask_2 = batch[2]
    else:
        mask_2 = None
    if preprocessing_fun is None:
        preprocessing_fun = model.preprocessing_fun
    if plot_prediction:
        lcv.plot_image_mask_pred_wrapper(ims_plot=batch[0], masks_plot=batch[1], masks_2_plot=mask_2,
                                         preds_plot=predicted_labels, preprocessing_fun=preprocessing_fun,
                                         lc_class_name_list=lc_class_name_list, names_patches=names_patches,
                                         unique_labels_array=unique_labels_array)
    if len(batch) == 3:
        return (batch[0], batch[1], batch[2], predicted_labels)    
    elif len(batch) == 2:
        return (batch[0], batch[1], predicted_labels)

def prediction_one_tile(model, trainer=None, tilepath='', patch_size=512,
                        batch_size=10, save_raster=False, save_shp=False,
                        create_shp=False, model_name=None, verbose=1,
                        dissolve_small_pols=False, area_threshold=100,
                        save_folder='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_predicted/predictions_LCU_2022-11-30-1205'):
    if trainer is None:
        trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, enable_progress_bar=False)  # run on GPU; and set max_epochs.
    if save_shp or dissolve_small_pols:
        create_shp = True  # is needed to save or dissolve
    ## Load tile
    im_tile = lca.load_tiff(tiff_file_path=tilepath, datatype='da')
    im_tile = im_tile.assign_coords({'ind_x': ('x', np.arange(len(im_tile.x))),
                                     'ind_y': ('y', np.arange(len(im_tile.y)))})
    ## TODO: catch geo coords / ref
    ## Copy of full tile
    mask_tile = copy.deepcopy(im_tile.sel(band=1, drop=True))
    mask_tile[:, :] = 0  # set everything to no class

    ## Split up tile in main + right side + bottom
    assert len(im_tile.x) == len(im_tile.y)
    n_pix = len(im_tile.x)
    n_patches_per_side = int(np.floor(n_pix / patch_size))
    n_pix_fit = n_patches_per_side * patch_size
    assert n_pix_fit % batch_size == 0
    
    im_main = im_tile.where(im_tile.ind_x < n_pix_fit, drop=True)
    im_main = im_main.where(im_tile.ind_y < n_pix_fit, drop=True)

    ## Cut off top & right side. Patch mirrored matrix and just do first row? 
    if verbose > 0:
        print('Divided tile')
    
    ## Create patches
    patches_im, _ = lca.create_image_mask_patches(image=im_main, mask=None, patch_size=patch_size)
    patches_im = lca.change_data_to_tensor(patches_im, tensor_dtype='float', verbose=0)
    patches_im = lca.apply_zscore_preprocess_images(im_ds=patches_im[0], f_preprocess=model.preprocessing_func)
    assert patches_im.shape[0] == n_patches_per_side ** 2

    ## Create DL with patches (just use standard DL as in notebook)
    predict_ds = TensorDataset(patches_im)
    predict_dl = DataLoader(predict_ds, batch_size=batch_size, num_workers=8)

    if verbose > 0:
        print('Predicting patches:')
    ## Predict from DL
    pred_masks = trainer.predict(model, predict_dl)
    pred_masks = lca.concat_list_of_batches(pred_masks)
    pred_masks = lca.change_tensor_to_max_class_prediction(pred=pred_masks)

    ## set dtype to something appropriate (uint8?) 

    ## [Handle edges, by patching from the other side; or using next tile]
    ## For no, shape_predicted_tile_part is returned to indicate the shape of the predicted part of the tile

    ## Reconstruct full tile
    assert pred_masks.shape[0] == patches_im.shape[0] and pred_masks.shape[0] == n_patches_per_side ** 2
    assert pred_masks.shape[-2:] == patches_im.shape[-2:] and pred_masks.shape[-2] == patch_size
    temp_shape = (n_patches_per_side, n_patches_per_side, patch_size, patch_size)  # need for unpatchifying below:
    reconstructed_tile_mask = patchify.unpatchify(pred_masks.detach().numpy().reshape(temp_shape), im_main.shape[-2:]) # won't work if im_tile has a remainder
    assert reconstructed_tile_mask.ndim == 2
    shape_predicted_tile_part = reconstructed_tile_mask.shape
    ## Add back geo coord:
    mask_tile[:shape_predicted_tile_part[0], :shape_predicted_tile_part[1]] = reconstructed_tile_mask
    
    ## Save & return
    assert model_name is not None, 'add model name to LCU upon saving.. '
    tile_name = tilepath.split('/')[-1].rstrip('.tif')
    if create_shp:
        if verbose > 0:
            print('Now creating polygons of prediction')
        shape_gen = ((shapely.geometry.shape(s), v) for s, v in rasterio.features.shapes(mask_tile.to_numpy(), transform=mask_tile.rio.transform()))  # create generator with shapes
        gdf = gpd.GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=mask_tile.rio.crs)
        gdf['Class name'] = 'A'
        for ii, lab in enumerate(model.dict_training_details['class_name_list']):
           gdf['Class name'].iloc[gdf['class'] == ii] = lab 
        if dissolve_small_pols:
            gdf = lca.filter_small_polygons_from_gdf(gdf=gdf, area_threshold=area_threshold, class_col='class',
                                                     verbose=verbose)
            ## Then convert back to raster so they are consistent: 
            ds_dissolved_tile = lca.convert_shp_mask_to_raster(df_shp=gdf, col_name='class')
            assert ds_dissolved_tile['class'].shape == mask_tile.shape
            assert (ds_dissolved_tile['class'].x == mask_tile.x).all()
            assert (ds_dissolved_tile['class'].y == mask_tile.y).all()
            mask_tile[:, :] = ds_dissolved_tile['class'][:, :]
        if save_shp:
            if dissolve_small_pols:
                name_file = f'{model_name}_{tile_name}_LC-prediction_dissolved_{area_threshold}m2'
            else:
                name_file = f'{model_name}_{tile_name}_LC-prediction'
            save_path = os.path.join(save_folder, name_file)
            gdf.to_file(save_path)
            if verbose > 0:
                print(f'Saved {name_file} with {len(gdf)} polygons to {save_path}')
            ## zip:
            ## shutil.make_archive(output_filename, 'zip', dir_name)
    else:
        gdf = None

    if save_raster:
        assert False, 'raster saving not yet implemented'
    
    return mask_tile, gdf, shape_predicted_tile_part

def tile_prediction_wrapper(model, trainer=None, dir_im='', dir_mask_eval=None, mask_suffix='_lc_2022_mask.tif',
                             patch_size=512, batch_size=10, save_shp=False, save_raster=False, save_folder=None,
                             dissolve_small_pols=False, area_threshold=100, skip_factor=None):
    '''Wrapper function that predicts & reconstructs full tile.'''
    ## Get list of all image tiles to predict
    list_tiff_tiles = lca.get_all_tifs_from_subdirs(dir_im)
    print(f'Loaded {len(list_tiff_tiles)} tiffs from subdirs of {dir_im}')
    unique_labels_array = np.arange(7)  # hard coded because dict_treaining_details needs to be fixed

    if dir_mask_eval is not None:
        print('Evaluating vs true masks')
        dict_acc = {} 
        dict_conf_mat = {}
        dict_df_stats = {}
    else:
        print('No true masks given')
        dict_acc = None 
        dict_df_stats = None
        dict_conf_mat = None 

    if area_threshold == 0 and dissolve_small_pols is True:
        dissolve_small_pols = False
        print('WARNING: area_threshold is 0, so no polygons will be dissolved.')
    if area_threshold > 0 and dissolve_small_pols is False:
        print('Warning: area_threshold is > 0, but dissolve_small_pols is False, so NOT dissolving small polygons.')
        assert False
    if dissolve_small_pols:
        print('Dissolving small polygons. WARNING: this takes considerable overhead computinsave_rasterg time')

    if save_shp:
        if save_folder is None:
            save_folder = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_predicted/predictions_LCU_2022-11-30-1205_dissolved_1000m2'
            print(f'No save folder given, so saving to {save_folder}')
                    
    ## Loop across tiles:

    for i_tile, tilepath in tqdm(enumerate(list_tiff_tiles)):
        mask_tile, mask_shp, shape_predicted_tile = prediction_one_tile(model=model, tilepath=tilepath, trainer=trainer, verbose=0,
                                                      save_shp=save_shp, save_raster=save_raster, save_folder=save_folder,
                                                      model_name='LCU_2022-11-30-1205',
                                                      create_shp=True,
                                                      dissolve_small_pols=dissolve_small_pols, area_threshold=area_threshold)

        if dir_mask_eval is not None:
            tilename = tilepath.split('/')[-1].rstrip('.tif')
            tile_path_mask = os.path.join(dir_mask_eval, tilename + mask_suffix)
            mask_tile_true = np.squeeze(lca.load_tiff(tile_path_mask, datatype='np'))
            mask_tile = mask_tile.to_numpy()

            ## Cut off no-class edge
            assert mask_tile_true.shape == mask_tile.shape, f'Predicted mask shape {mask_tile.shape} does not match true mask shape {mask_tile_true.shape}'
            mask_tile = mask_tile[:shape_predicted_tile[0], :shape_predicted_tile[1]]
            mask_tile_true = mask_tile_true[:shape_predicted_tile[0], :shape_predicted_tile[1]]

            ## Compute confusion matrix:
            conf_mat = lca.compute_confusion_mat_from_two_masks(mask_true=mask_tile_true, mask_pred=mask_tile, 
                                                        lc_class_name_list=model.dict_training_details['class_name_list'], 
                                                        unique_labels_array=unique_labels_array, skip_factor=skip_factor)
            tmp = lcv.plot_confusion_summary(conf_mat=conf_mat, class_name_list=model.dict_training_details['class_name_list'], 
                                             plot_results=False, normalise_hm=True)
            dict_df_stats[tilename], dict_acc[tilename], _, __ = tmp 
            dict_conf_mat[tilename] = conf_mat
    
        # if i_tile == 1:
        #     break

    return dict_acc, dict_df_stats, dict_conf_mat

def save_details_trainds_to_model(model, train_ds):
    '''Save details of train data set to model'''
    assert type(model) == LandCoverUNet, f'{type(model)} not recognised'
    assert type(train_ds) == DataSetPatches, f'{type(train_ds)} not recognised'
    assert type(model.dict_training_details) == dict 

    assert len(model.dict_training_details) == 0, f'training dictionary is not empty but contains {model.dict_training_details.keys()}. Consider a new function that adds info of second training procedure?'

    list_names_attrs = ['df_patches', 'im_dir', 'mask_dir', 'path_mapping_dict', 
                        'preprocessing_func', 'rgb_means', 'rgb_std', 'shuffle_order_patches', 
                        'frac_subsample', 'unique_labels_arr', 'mapping_label_to_new_dict', 
                        'class_name_list', 'n_classes']

    for name_attr in list_names_attrs:  # add to model one by one:
        model.dict_training_details[name_attr] = getattr(train_ds, name_attr)

    print(f'Details of training data set {train_ds} have been added to {model}')