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
