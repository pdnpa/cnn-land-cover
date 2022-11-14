from json import encoder
import os, sys, copy
import numpy as np
# from numpy.core.multiarray import square
# from numpy.testing import print_assert_equal
# import rasterio
# import xarray as xr
# import rioxarray as rxr
import sklearn.model_selection
from tqdm import tqdm
# import shapely as shp
import pandas as pd
# import geopandas as gpd
# from geocube.api.core import make_geocube
# import gdal, osr
import loadpaths
# from patchify import patchify 
import torch, torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import land_cover_analysis as lca

path_dict = loadpaths.loadpaths()

class DataLoaderPatches(torch.utils.data.Dataset):
    def __init__(self, im_dir, mask_dir, mask_suffix='_lc_80s_mask.npy', 
                 preprocessing_func=None, unique_labels_arr=None):
        super(DataLoaderPatches, self).__init__()
        self.im_dir = im_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.preprocessing_func = preprocessing_func

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
        self.list_im_npys = [os.path.join(im_dir, x) for x in os.listdir(im_dir)]
        self.list_patch_names = [x.split('/')[-1].rstrip('.npy') for x in self.list_im_npys]
        self.list_mask_npys = [os.path.join(mask_dir, x.split('/')[-1].rstrip('.npy') + mask_suffix) for x in self.list_im_npys]

        self.df_patches = pd.DataFrame({'patch_name': self.list_patch_names,
                                        'im_filepath': self.list_im_npys, 
                                        'mask_filepath': self.list_mask_npys})

        ## Prep the transformation of class inds:
        dict_ind_to_name, dict_name_to_ind = lca.get_lc_mapping_inds_names_dicts() 
        if unique_labels_arr == None:  # if no array given, presume full array:
            self.unique_labels_arr = np.unique(np.array(list(dict_ind_to_name.keys())))  # unique sorts too 
        else:
            self.unique_labels_arr = np.unique(unique_labels_arr)
        self.mapping_label_to_new_dict = {label: ind for ind, label in enumerate(self.unique_labels_arr)}
        self.class_name_list = [dict_ind_to_name[label] for label in self.unique_labels_arr]
        
    def __getitem__(self, index):
        '''Function that gets data items by index'''
        patch_row = self.df_patches.iloc[index]
        im = np.load(patch_row['im_filepath'])
        mask = np.load(patch_row['mask_filepath'])
        im = torch.tensor(im).float()
        im = self.preprocess_image(im)
        mask = torch.tensor(mask).type(torch.LongTensor)
        mask = self.remap_labels(mask)
        return im, mask 

    def __repr__(self):
        return f'DataLoaderPatches class'

    def __len__(self):
        return len(self.df_patches)
    
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

class LandCoverUNet(pl.LightningModule):
    ## I think it's best to use the Lightning Module. See here:
    ## https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    ## https://colab.research.google.com/drive/1eRgcdQvNWzcEed2eTj8paDnnQ0qplXAh?usp=sharing#scrollTo=V7ELesz1kVQo
    def __init__(self, n_classes=10, encoder_name='resnet50', pretrained='imagenet',
                 lr=1e-3):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
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

        ## Define loss used for training:
        # self.loss = self.dummy_loss
        self.loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)  # reduction: 'none' (returns full-sized tensor), 'mean', 'sum'. Can also insert class weights and ignore indices
        # self.seg_val_metric = pl.metrics.Accuracy()

        # self.log(prog_bar=True)
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.base(x)
        loss = self.loss(output, y)

        self.log('test_loss', loss)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.base(x)
        loss = self.loss(output, y)

        self.log('val_loss', loss, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        '''Takes batches of (images, masks), like training etc. Then call forward and only give output.
        Output will be tensor of output masks.
        
        When calling this function with a DataLoader (trainer.predict(LCU, test_dl), then the output will be
        a list of batches'''
        x, y = batch
        output = self.base(x)
        
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # momentum=0.9
        return optimizer

