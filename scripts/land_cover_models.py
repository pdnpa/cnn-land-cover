from json import encoder
import os, sys, copy
import numpy as np
# from numpy.core.multiarray import square
# from numpy.testing import print_assert_equal
# import rasterio
# import xarray as xr
# import rioxarray as rxr
# import sklearn.model_selection
from tqdm import tqdm
import datetime
import pickle
# import shapely as shp
import pandas as pd
# import geopandas as gpd
# from geocube.api.core import make_geocube
# import gdal, osr
import loadpaths
import patchify 
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import land_cover_analysis as lca
import land_cover_visualisation as lcv


path_dict = loadpaths.loadpaths()

class DataSetPatches(torch.utils.data.Dataset):
    def __init__(self, im_dir, mask_dir, mask_suffix='_lc_80s_mask.npy', 
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

        if subsample_patches:
            assert frac_subsample <= 1 and frac_subsample > 0
            n_subsample = int(len(self.df_patches) * frac_subsample)
            print(f'Subsampling {n_subsample} patches')
            self.df_patches = self.df_patches[:n_subsample]
        
        if self.shuffle_order_patches:
            print('Patches ordered randomly')
            self.df_patches = self.df_patches.sample(frac=1, replace=False)
        else:
            print('Patches sorted by tile/patch order')
            self.df_patches = self.df_patches.sort_values('patch_name')
        self.df_patches = self.df_patches.reset_index(drop=True)
            
        # ## Prep the transformation of class inds:
        if self.path_mapping_dict is None:
            print('WARNING: no label mapping given - so using all labels individually')
            dict_ind_to_name, dict_name_to_ind = lca.get_lc_mapping_inds_names_dicts() 
            if unique_labels_arr == None:  # if no array given, presume full array:
                self.unique_labels_arr = np.unique(np.array(list(dict_ind_to_name.keys())))  # unique sorts too 
            else:
                self.unique_labels_arr = np.unique(unique_labels_arr)
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

    def __getitem__(self, index):
        '''Function that gets data items by index'''
        patch_row = self.df_patches.iloc[index]
        im = np.load(patch_row['im_filepath'])  #0.25ms
        mask = np.load(patch_row['mask_filepath'])  #0.25ms
        im = torch.tensor(im).float()  # 0.1ms
        im = self.preprocess_image(im)  # 0.25 ms
        mask = torch.tensor(mask).type(torch.LongTensor)  #0.1ms
        if self.relabel_masks:
            mask = self.remap_labels(mask)  # 2.5ms
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

        self.filename = None
        self.filepath = None

        ## Add info dict with some info: epochs, PL version, .. 
        self.dict_training_details = {}  # can be added post hoc once train dataset is defined

    def __repr__(self):
        return f'LandCoverUNet class'

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

    def save_model(self, folder='', verbose=1):
        '''Save model'''
        timestamp = lca.create_timestamp()
        self.filename = f'LCU_{timestamp}.data'
        self.filepath = os.path.join(folder, self.filename)

        file_handle = open(self.filepath, 'wb')
        pickle.dump(self, file_handle)
        if verbose > 0:
            print(f'LCU model saved as {self.filename} at {self.filepath}')

def load_model(folder='', filename=''):
    with open(os.path.join(folder, filename), 'rb') as f:
        LCU = pickle.load(f)
    return LCU 

def get_batch_from_ds(ds, batch_size=5, start_ind=0):
    tmp_items = []
    assert type(batch_size) == int and type(start_ind) == int
    for ii in range(start_ind, start_ind + batch_size):
        tmp_items.append(ds[ii])
    list_inputs = [torch.Tensor(x[0])[None, :, :, :] for x in tmp_items]
    list_outputs = [torch.Tensor(x[1])[None, :, :] for x in tmp_items]
    list_inputs = lca.concat_list_of_batches(list_inputs)
    list_outputs = lca.concat_list_of_batches(list_outputs)
    return [list_inputs, list_outputs]

def predict_single_batch_from_testdl_or_batch(model, test_dl=None, batch=None, 
                                              plot_prediction=True, preprocessing_fun=None,
                                              lc_class_name_list=None, unique_labels_array=None):
    if batch is None and test_dl is not None:
        batch = next(iter(test_dl))
    elif batch is not None and test_dl is None:
        pass  # use batch 
    assert len(batch) == 2 and type(batch) == list, 'batch of unexpected format.. Expected [batch_images, batch_masks]'
    predicted_labels = model.forward(batch[0])
    predicted_labels = lca.change_tensor_to_max_class_prediction(pred=predicted_labels)
    if preprocessing_fun is None:
        preprocessing_fun = model.preprocessing_fun
    if plot_prediction:
        lcv.plot_image_mask_pred_wrapper(ims_plot=batch[0], masks_plot=batch[1],
                                         preds_plot=predicted_labels, preprocessing_fun=preprocessing_fun,
                                         lc_class_name_list=lc_class_name_list,
                                         unique_labels_array=unique_labels_array)
    return (batch[0], batch[1], predicted_labels)

def tile_prediction_wrapper(model, trainer, dir_im='', patch_size=512,
                            batch_size=10):
    ## Get list of all image tiles to predict
    list_tiff_tiles = lca.get_all_tifs_from_dir(dir_im)

    ## Loop across tiles:
    for i_tile, tilepath in tqdm(enumerate(list_tiff_tiles)):

        ## Load tile
        im_tile = lca.load_tiff(tiff_file_path=tilepath, datatype='da')

        ## Create patches
        ## TODO: cut off im_tile here so exactly factorized by patch_size. 
        ## Cut off top & right side. Patch mirrored matrix and just do first row? 

        patches_im, _ = lca.create_image_mask_patches(image=im_tile, mask=None, patch_size=patch_size)
        patches_im = lca.change_data_to_tensor(patches_im, tensor_dtype='float')
        patches_im = lca.apply_zscore_preprocess_images(im_ds=patches_im, f_preprocess=model.preprocessing_func)

        ## Create DL with patches (just use standard DL as in notebook)
        predict_ds = TensorDataset(patches_im)
        predict_dl = DataLoader(predict_ds, batch_size=batch_size)

        ## Predict from DL
        pred_masks = trainer.predict(model, predict_dl)
        pred_masks = lca.concat_list_of_batches(pred_masks)
        pred_masks = lca.change_tensor_to_max_class_prediction(pred=pred_masks)
        ## set dtype to something appropriate (uint8?) 

        ## [Handle edges, by patching from the other side; or using next tile]


        ## Reconstruct full tile
        assert pred_masks.shape == patches_im.shape 
        reconstructed_tile_mask = patchify.unpatchify(pred_masks, im_tile.shape) # won't work if im_tile has a remainder

        ## Save

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