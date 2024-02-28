from json import encoder
import os, sys, copy, shutil
import numpy as np
from tqdm import tqdm
import datetime
import random
import pickle
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
from torchvision import transforms
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import custom_losses as cl

class DataSetPatches(torch.utils.data.Dataset):
    '''Data set for images & masks. Saves file paths, but only loads into memory during __getitem__.
    
    Used for training etc - __getitem__ has expected output (input, output) for PL models.
    '''
    def __init__(self, im_dir, mask_dir, mask_suffix='_lc_80s_mask.npy', mask_dir_name='masks', 
                 list_tile_names=None, list_tile_patches_use=None,
                 preprocessing_func=None, shuffle_order_patches=True,
                 subsample_patches=False, frac_subsample=1, relabel_masks=True, random_transform_data=False,
                 path_mapping_dict='../content/label_mapping_dicts/label_mapping_dict__main_categories__2022-11-17-1512.pkl'):
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
        self.list_tile_names = list_tile_names
        self.list_tile_patches_use = list_tile_patches_use
        self.random_transform_data = random_transform_data

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
        self.mask_dir_name = mask_dir_name.rstrip('/').lstrip('/')
            
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
            print(f'No mask directory provided. Will use {self.mask_dir_name}/ in image parent directory instead.')
            if self.multiple_im_dirs:
                tmp = list(set([x.split('/')[-2] for x in self.im_dir]))
                assert len(tmp) == 1
                im_dir_name = tmp[0]
            else:
                im_dir_name = self.im_dir.split('/')[-2]
            self.list_mask_npys = [x.replace(f'/{im_dir_name}/', f'/{self.mask_dir_name}/').replace('.npy', mask_suffix) for x in self.list_im_npys]
        else:
            assert self.multiple_im_dirs is False, 'Cannot use multiple image directories if mask directory is provided.'
            self.list_mask_npys = [os.path.join(mask_dir, x.split('/')[-1].rstrip('.npy') + mask_suffix) for x in self.list_im_npys]

        self.create_df_patches(list_tile_patches_use=self.list_tile_patches_use)
        self.organise_df_patches()
        print(f'Loaded {len(self.df_patches)} patches')
        self.create_label_mapping()    
        assert hasattr(self, 'n_classes'), 'n_classes not defined (should be defined in create_label_mapping())'
        
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
        if self.random_transform_data:
            im, mask = self.transform_data(im, mask)
        return im, mask 

    def __repr__(self):
        return f'DataLoaderPatches class'

    def __len__(self):
        return len(self.df_patches)

    def create_df_patches(self, list_tile_patches_use=None):
        '''Create dataframe with all patch locations'''
        self.df_patches = pd.DataFrame({'patch_name': self.list_patch_names,
                                        'tile_name': [x[:6] for x in self.list_patch_names],
                                        'im_filepath': self.list_im_npys, 
                                        'mask_filepath': self.list_mask_npys})
        if list_tile_patches_use is not None:
            print(f'Only using patches that are in tile_patches list (of length {len(list_tile_patches_use)}).')
            # print(f'Loaded {len(self.df_patches)} patches, of {len(self.list_patch_names)} total patches.)')
            self.df_patches = self.df_patches[self.df_patches['patch_name'].isin(list_tile_patches_use)]
            # print(f'Loaded {len(self.df_patches)} patches, of {len(self.list_patch_names)} total patches.)')
            # assert len(self.df_patches) == len(list_tile_patches_use), f'Not all patches in list_tile_patches_use are in the image/mask directories: {len(self.df_patches)} vs {len(list_tile_patches_use)}. These are the missing patches: {list(set(list_tile_patches_use) - set(self.df_patches["patch_name"]))}'

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
            
    def remove_no_class_patches(self, check_unique_classes=True):
        ## Loop through df_patches, load patches.
        ## Check if any class present (that isn't no class). If not, remove from df_patches
        n_patches = len(self.df_patches)
        list_patches_stay = []
        if check_unique_classes:
            list_unique_classes = []
        for ind in tqdm(range(n_patches)):
            _, mask = self.__getitem__(ind)
            if mask.sum() == 0:
                pass 
            else:
                list_patches_stay.append(ind)
                if check_unique_classes:
                    list_unique_classes.append(np.unique(mask))
        if check_unique_classes:
            list_unique_classes = np.unique(np.concatenate(list_unique_classes))
            bool_classes_complete = (list_unique_classes == np.arange(len(list_unique_classes))).all()
            if bool_classes_complete is False:
                print(f'Unique classes are not 0 to n_classes-1: {list_unique_classes}')
            self.list_unique_classes = list_unique_classes
        else:
            self.list_unique_classes = None
        self.df_patches = self.df_patches.iloc[np.array(list_patches_stay)]
        self.df_patches = self.df_patches.reset_index(drop=True)
        print(f'Removed {n_patches - len(self.df_patches)} patches with no class')

    def create_label_mapping(self):
        '''Prep the transformation of class inds'''
        if self.path_mapping_dict is None:
            assert False, 'WARNING: no label mapping given'
            ##  using all labels individually
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

    def transform_data(self, im, mask):
        '''https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7'''
        # Random horizontal flipping
        if random.random() > 0.5:
            im = TF.hflip(im)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            im = TF.vflip(im)
            mask = TF.vflip(mask)

        return im, mask

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

    def create_df_patches(self, list_tile_patches_use=None):
        '''Override to include mask_2_filepath'''
        assert list_tile_patches_use is None, 'Not implemented yet'
        
        self.list_mask_2_npys = [os.path.join(self.mask_dir_2, x.split('/')[-1].rstrip('.npy') + self.mask_suffix_2) for x in self.list_im_npys]
        self.df_patches = pd.DataFrame({'patch_name': self.list_patch_names,
                                        'tile_name': [x[:6] for x in self.list_patch_names],
                                        'im_filepath': self.list_im_npys, 
                                        'mask_filepath': self.list_mask_npys,
                                        'mask_2_filepath': self.list_mask_2_npys})


class LandCoverUNet(pl.LightningModule):
    '''
    UNet for semantic segmentation. Build using API of pytorch lightning
    (see: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)
    
    
    '''
    def __init__(self, n_classes=10, encoder_name='resnet50', pretrained='imagenet',
                 lr=1e-3, loss_function='cross_entropy', skip_factor_eval=1,
                 first_class_is_no_class=False, ignore_index=0):
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
        self.encoder_name = encoder_name
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        self.focal_loss = cl.FocalLoss_2(gamma=0.75, reduction='mean', ignore_index=ignore_index)
        # self.iou_loss = cl.mIoULoss(n_classes=n_classes)  # has no ignore-index
        # self.dice_loss = torchmetrics.Dice(num_classes=n_classes, ignore_index=ignore_index, requires_grad=True)#, average='macro')
        # self.focal_and_dice_loss = lambda x, y: self.focal_loss(x, y) + self.dice_loss(x, y)
        self.n_classes = n_classes
        self.first_class_is_no_class = first_class_is_no_class

        ## Define loss used for training:
        if loss_function == 'dummy':
            self.loss = self.dummy_loss
        elif loss_function == 'cross_entropy':
            self.loss = self.ce_loss  # reduction: 'none' (returns full-sized tensor), 'mean', 'sum'. Can also insert class weights and ignore indices
        elif loss_function == 'focal_loss':
            self.loss = self.focal_loss
        # elif loss_function == 'iou_loss':
        #     self.loss = self.iou_loss
        # elif loss_function == 'dice_loss':
        #     self.loss = self.dice_loss
        # elif loss_function == 'focal_and_dice_loss':
        #     self.loss = self.focal_and_dice_loss
        # elif loss_function == 'weighted_cross_entropy':
        #     self.loss_weights = torch.zeros(n_classes)
        #     self.loss_weights[1] = 1.0  # D1 
        #     self.loss_weights[2] = 0.7  # D2b
        #     self.loss_weights[13] = 0.5  # F3a
        #     self.loss_weights[14] = 6.0 # F3d
        #     self.loss = nn.CrossEntropyLoss(weight=self.loss_weights, reduction='mean', ignore_index=ignore_index)
        else:
            assert False, f'Loss function {loss_function} not recognised.'
        print(f'{loss_function} loss is used.')
        self.calculate_test_confusion_mat = True
        self.reset_test_confusion_mat()
        
        self.model_name = 'LCU (not saved)'
        self.description = f'LandCoverUNet class using {self.loss}'
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
    
    def reset_test_confusion_mat(self):
        self.test_confusion_mat = np.zeros((self.n_classes, self.n_classes))

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
    
        if hasattr(self, 'ce_loss'):
            self.log('test_ce_loss', self.ce_loss(output, y))
        if hasattr(self, 'focal_loss'):
            self.log('test_focal_loss', self.focal_loss(output, y))
        if hasattr(self, 'iou_loss'):
            self.log('test_iou_loss', self.iou_loss(output, y))
        if hasattr(self, 'dice_loss'):
            self.log('test_dice_loss', self.dice_loss(output, y))
        if hasattr(self, 'focal_and_dice_loss'):
            self.log('test_focal_and_dice_loss', self.focal_and_dice_loss(output, y))
        
        if self.calculate_test_confusion_mat:
            if self.skip_factor_eval is None:
                det_output = lca.change_tensor_to_max_class_prediction(pred=output, expected_square_size=512)  # change soft maxed output to arg max
            else:   
                det_output = lca.change_tensor_to_max_class_prediction(pred=output, expected_square_size=512 / self.skip_factor_eval)  # change soft maxed output to arg max
            assert det_output.shape == y.shape
            assert output.ndim == 4
            n_classes = output.shape[1]
            for ic_true in range(n_classes):
                for ic_pred in range(n_classes):
                    n_match = int((det_output[y == ic_true] == ic_pred).sum()) 
                    self.test_confusion_mat[ic_true, ic_pred] += n_match  # just add to existing matrix; so it can be done in batches
            if self.first_class_is_no_class:
                conf_mat_use = self.test_confusion_mat[1:, 1:]
            else:
                conf_mat_use = self.test_confusion_mat
            overall_accuracy = conf_mat_use.diagonal().sum() / conf_mat_use.sum() 
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
    
    def tidy_up_metrics(self):
        if self.metrics is None:
            self.metrics_float = None
            return
        
        self.n_epochs_converged = len(self.metrics) 
        self.metrics_float = []
        self.set_metric_names = set()
        for ii in range(len(self.metrics)):
            self.metrics_float.append({})
            for key in self.metrics[ii].keys():
                self.set_metric_names.add(key)
                value = self.metrics[ii][key]
                if type(value) == float:
                    continue 
                if type(value) == torch.Tensor:
                    self.metrics_float[ii][key] = value.detach().cpu().numpy()
                assert type(self.metrics_float[ii][key]) == np.ndarray, type(self.metrics_float[ii][key])
                assert self.metrics_float[ii][key].shape == (), self.metrics_float[ii][key]
                self.metrics_float[ii][key] = float(self.metrics_float[ii][key])

        self.metric_arrays = {}
        for key in self.set_metric_names:
            self.metric_arrays[key] = np.zeros(self.n_epochs_converged) + np.nan
            for ii in range(self.n_epochs_converged):
                if key in self.metrics_float[ii]:
                    self.metric_arrays[key][ii] = self.metrics_float[ii][key]
         
        return 

    def save_model(self, folder='/home/david/models/', verbose=1, metrics=None):
        '''Save model'''
        ## Save v_num that is used for tensorboard
        self.v_num = self.logger.version
        ## Save logging directory that is used for tensorboard
        self.log_dir = self.logger.log_dir
        self.metrics = metrics
        self.tidy_up_metrics()
        
        timestamp = lca.create_timestamp()
        self.filename = f'LCU_{timestamp}.data'
        self.model_name = f'LCU_{timestamp}'
        self.filepath = os.path.join(folder, self.filename)

        file_handle = open(self.filepath, 'wb')
        pickle.dump(self, file_handle)

        if verbose > 0:
            print(f'LCU model saved as {self.filename} at {self.filepath}')
        return self.filepath

def load_model(folder='/home/tplas/models', filename='', verbose=1):
    '''Load previously saved (pickled) LCU model'''
    with open(os.path.join(folder, filename), 'rb') as f:
        LCU = pickle.load(f)

    if verbose > 0:  # print some info
        print(f'Loaded {LCU}')
        for info_name in ['loss_function', 'n_max_epochs']:
            if info_name in LCU.dict_training_details.keys() and verbose > 0:
                print(f'{info_name} is {LCU.dict_training_details[info_name]}')
        if hasattr(LCU, 'description') and verbose > 0:
            print(LCU.description)

    return LCU 

def get_batch_from_ds(ds, batch_size=5, start_ind=0):
    '''Given DS, retrieve a batch of data (for plotting etc)'''
    tmp_items = []
    names_patches = []
    assert type(batch_size) == int and type(start_ind) == int, f'batch_size {batch_size} and start_ind {start_ind} should be int, not {type(batch_size)} and {type(start_ind)}'
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

def get_any_combination_from_ds(ds, list_inds=[]):
    for ind in list_inds:
        tmp_batch, tmp_names = get_batch_from_ds(ds, batch_size=1, start_ind=ind)
        if ind == list_inds[0]:
            batch = tmp_batch
            names_patches = tmp_names
        else:
            batch[0] = torch.cat([batch[0], tmp_batch[0]], dim=0)
            batch[1] = torch.cat([batch[1], tmp_batch[1]], dim=0)
            names_patches.append(tmp_names[0])
    return batch, names_patches

def predict_single_batch_from_testdl_or_batch(model, test_dl=None, batch=None, names_patches=None,
                                              plot_prediction=True, preprocessing_fun=None,
                                              lc_class_name_list=None, unique_labels_array=None,
                                              title_2022_annot=True):
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
                                         unique_labels_array=unique_labels_array, title_2022_annot=title_2022_annot)
    if len(batch) == 3:
        return (batch[0], batch[1], batch[2], predicted_labels)    
    elif len(batch) == 2:
        return (batch[0], batch[1], predicted_labels)

def prediction_one_tile(model, trainer=None, tilepath='', tilename='', patch_size=512, padding=0,
                        batch_size=10, save_raster=False, save_shp=False,
                        create_shp=False, verbose=1, df_schema=None,
                        dissolve_small_pols=False, area_threshold=100,
                        use_class_dependent_area_thresholds=False,
                        class_dependent_area_thresholds=dict(),
                        name_combi_area_thresholds=None,
                        reconstruct_padded_tile_edges=True,
                        clip_to_main_class=False, main_class_clip_label='C', col_name_class=None,  # col_name_class used for BOTH clip_to_main_class and dissolve_small_pols
                        parent_dir_tile_mainpred='/home/tplas/predictions/predictions_LCU_2023-01-23-2018_dissolved1000m2_padding44_FGH-override/',
                        tile_outlines_shp_path='../content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp',
                        save_folder='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_predicted/predictions_LCU_2022-11-30-1205'):
    if trainer is None:
        trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=1, enable_progress_bar=False)  # run on GPU; and set max_epochs.
    if save_shp or dissolve_small_pols:
        create_shp = True  # is needed to save or dissolve
    if padding > 0:
        assert padding % 2 == 0, 'Padding should be even number'

    ## Load tile
    im_tile = lca.load_tiff(tiff_file_path=tilepath, datatype='da')
    im_tile = im_tile.assign_coords({'ind_x': ('x', np.arange(len(im_tile.x))),
                                     'ind_y': ('y', np.arange(len(im_tile.y)))})
    ## Copy of full tile
    mask_tile = copy.deepcopy(im_tile.sel(band=1, drop=True))
    mask_tile[:, :] = 0  # set everything to no class

    ## Split up tile in main + right side + bottom
    assert len(im_tile.x) == len(im_tile.y)
    n_pix = len(im_tile.x)
    step_size = patch_size - padding  # effective step size
    n_patches_per_side = int(np.floor(n_pix / step_size  - padding / step_size))
    n_pix_fit = n_patches_per_side * step_size + padding
    if padding == 0:
        assert n_pix_fit % step_size == 0
    
    im_main = im_tile.where(im_tile.ind_x < n_pix_fit, drop=True)
    im_main = im_main.where(im_tile.ind_y < n_pix_fit, drop=True)
    if verbose > 0:  # print all shapes
        print(f'Original tile shape: {im_tile.shape}, n_pix_fit: {n_pix_fit}, n_patches_per_side: {n_patches_per_side}, step_size: {step_size}, padding: {padding}')
    
    ## Create patches
    patches_im, _ = lca.create_image_mask_patches(image=im_main, mask=None, verbose=1 if verbose > 1 else 0,
                                                  patch_size=patch_size, padding=padding)
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

    ## Reconstruct full tile
    assert pred_masks.shape[0] == patches_im.shape[0] and pred_masks.shape[0] == n_patches_per_side ** 2
    assert pred_masks.shape[-2:] == patches_im.shape[-2:] and pred_masks.shape[-2] == patch_size
    temp_shape = (n_patches_per_side, n_patches_per_side, step_size, step_size)  # need for unpatchifying below:
    if verbose > 1:
        print('Shapes pre pad-removal', pred_masks.shape, temp_shape, im_main.shape)
    half_pad = int(padding / 2) # should be even (asserted above)
    if padding > 0:  # need to remove padding
        pred_masks_padded = pred_masks[:, half_pad:-half_pad, :]
        pred_masks_padded = pred_masks_padded[:, :, half_pad:-half_pad]
        if verbose > 1:
            print('Shapes post pad-removal', pred_masks_padded.shape, temp_shape, im_main.shape)
    elif padding == 0:
        pred_masks_padded = pred_masks
    else:
        raise ValueError('Padding should be 0 or larger')
    
    assert np.product(temp_shape) == np.product(pred_masks_padded.shape)
    assert np.product(temp_shape) + (4 * n_patches_per_side * half_pad * step_size) + (4 * half_pad ** 2) == np.product(im_main.shape[-2:])  # innner part + 4 edges + 4 corners

    if padding == 0:    
        reconstructed_tile_mask = patchify.unpatchify(pred_masks_padded.detach().numpy().reshape(temp_shape), im_main.shape[-2:]) # won't work if im_tile has a remainder
    elif padding > 0:
        reconstructed_tile_mask_inner = patchify.unpatchify(pred_masks_padded.detach().numpy().reshape(temp_shape), (im_main.shape[-2] - padding, im_main.shape[-2] - padding)) 
        reconstructed_tile_mask = np.zeros(im_main.shape[-2:])
        if verbose > 1:
            print('Shapes post unpatchify', reconstructed_tile_mask.shape, reconstructed_tile_mask_inner.shape)
        reconstructed_tile_mask[half_pad:-half_pad, :][:, half_pad:-half_pad] = reconstructed_tile_mask_inner

    if reconstruct_padded_tile_edges:
        ## Idea: 1) only subtract half_pad in x dimension, get full y edges; 2) vice versa; 3) get 4 tile corners (of size half_pad x half_pad) manually
        ## NB: if the padding is such that the patches exactly tesselate the image, it's works perfectly. However if the patches don't exactly tesselate the image, there will some remainder on right & bottom side that isn't predicted (and thus not reconstructed) at the moment.
        ## 1) x edges
        pred_masks_x_edges = pred_masks[:, :, half_pad:-half_pad]
        temp_shape_x_edges = (n_patches_per_side, n_patches_per_side, step_size + padding, step_size)
        assert np.product(temp_shape_x_edges) == np.product(pred_masks_x_edges.shape)
        reconstructed_tile_mask_x_edges = patchify.unpatchify(pred_masks_x_edges.detach().numpy().reshape(temp_shape_x_edges), (im_main.shape[-2], im_main.shape[-2] - padding))
        print(reconstructed_tile_mask_x_edges.shape, reconstructed_tile_mask.shape)
        reconstructed_tile_mask[:half_pad, :][:, half_pad:-half_pad] = reconstructed_tile_mask_x_edges[:half_pad, :]
        reconstructed_tile_mask[-half_pad:, :][:, half_pad:-half_pad] = reconstructed_tile_mask_x_edges[-half_pad:, :]

        ## 2) y edges
        pred_masks_y_edges = pred_masks[:, half_pad:-half_pad, :]
        temp_shape_y_edges = (n_patches_per_side, n_patches_per_side, step_size, step_size + padding)
        assert np.product(temp_shape_y_edges) == np.product(pred_masks_y_edges.shape)
        reconstructed_tile_mask_y_edges = patchify.unpatchify(pred_masks_y_edges.detach().numpy().reshape(temp_shape_y_edges), (im_main.shape[-2] - padding, im_main.shape[-2]))
        reconstructed_tile_mask[:, :half_pad][half_pad:-half_pad, :] = reconstructed_tile_mask_y_edges[:, :half_pad]
        reconstructed_tile_mask[:, -half_pad:][half_pad:-half_pad, :] = reconstructed_tile_mask_y_edges[:, -half_pad:]

        ## 3) corners
        pred_masks_corners = pred_masks 
        temp_shape_corners = (n_patches_per_side, n_patches_per_side, step_size + padding, step_size + padding)
        assert np.product(temp_shape_corners) == np.product(pred_masks_corners.shape)
        reconstructed_tile_mask_corners = patchify.unpatchify(pred_masks_corners.detach().numpy().reshape(temp_shape_corners), (im_main.shape[-2], im_main.shape[-2]))
        reconstructed_tile_mask[:half_pad, :][:, :half_pad] = reconstructed_tile_mask_corners[:half_pad, :half_pad]
        reconstructed_tile_mask[:half_pad, :][:, -half_pad:] = reconstructed_tile_mask_corners[:half_pad, -half_pad:]
        reconstructed_tile_mask[-half_pad:, :][:, :half_pad] = reconstructed_tile_mask_corners[-half_pad:, :half_pad]
        reconstructed_tile_mask[-half_pad:, :][:, -half_pad:] = reconstructed_tile_mask_corners[-half_pad:, -half_pad:]
    
    assert reconstructed_tile_mask.ndim == 2
    shape_predicted_tile_part = reconstructed_tile_mask.shape
    assert shape_predicted_tile_part == im_main.shape[-2:]

    ## Add back geo coord:
    mask_tile[:shape_predicted_tile_part[0], :shape_predicted_tile_part[1]] = reconstructed_tile_mask

    ## Clip to one main class if applicable:
    if clip_to_main_class:
        if verbose > 0:
            print(f'Now clipping to main class {main_class_clip_label}')
        assert main_class_clip_label in ['C', 'D', 'E' , 1, 2, 3], main_class_clip_label
        assert type(tilename) == str and len(tilename) == 6, tilename
        mask_tile = lca.clip_raster_to_main_class_pred(mask_tile, tilename=tilename, 
                                    col_name_class=col_name_class, class_label=main_class_clip_label,
                                    parent_dir_tile_mainpred=parent_dir_tile_mainpred,
                                    tile_outlines_shp_path=tile_outlines_shp_path)

    ## Save & return
    model_name = model.model_name
    tile_name = tilepath.split('/')[-1].rstrip('.tif')
    if create_shp:
        if verbose > 0:
            print('Now creating polygons of prediction')
        shape_gen = ((shapely.geometry.shape(s), v) for s, v in rasterio.features.shapes(mask_tile.to_numpy(), transform=mask_tile.rio.transform()))  # create generator with shapes
        gdf = gpd.GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=mask_tile.rio.crs)
        gdf['Class name'] = 'A'
        for ii, lab in enumerate(model.dict_training_details['class_name_list']):
           gdf['Class name'].iloc[gdf['class'] == ii] = lab 
        if main_class_clip_label in ['C', 'D', 'E']:  # detailed class model 
            if df_schema is None:
                df_schema = lca.create_df_mapping_labels_2022_to_80s() 
            gdf[col_name_class] = gdf['Class name'].map(df_schema.set_index('description_2022')['code_2022'])
        else:  # main class model
            mapping_dict_main_class = {'C': 'Wood and Forest Land',
                                        'D': 'Moor and Heath Land',
                                        'E': 'Agro-Pastoral Land',
                                        'F': 'Water and Wetland',
                                        'G': 'Rock and Coastal Land',
                                        'H': 'Developed Land',
                                        'I': 'Unclassified Land',
                                        '0': 'NO CLASS'} 
            mapping_dict_main_class = {v: k for k, v in mapping_dict_main_class.items()}
            gdf[col_name_class] = gdf['Class name'].map(mapping_dict_main_class)
        if dissolve_small_pols:
            gdf = lca.filter_small_polygons_from_gdf(gdf=gdf, class_col='class', label_col=col_name_class,
                                                     area_threshold=area_threshold, use_class_dependent_area_thresholds=use_class_dependent_area_thresholds,
                                                     class_dependent_area_thresholds=class_dependent_area_thresholds,
                                                     verbose=verbose, exclude_no_class_from_large_pols=False if clip_to_main_class else True)  # if clip is True, then you don't want to exclude no class from large pols because everything that was clipped will be no class
            ## Then convert back to raster so they are consistent: 
            ds_dissolved_tile = lca.convert_shp_mask_to_raster(df_shp=gdf, col_name='class')
            assert ds_dissolved_tile['class'].shape == mask_tile.shape
            assert (ds_dissolved_tile['class'].x == mask_tile.x).all()
            assert (ds_dissolved_tile['class'].y == mask_tile.y).all()
            mask_tile[:, :] = ds_dissolved_tile['class'][:, :]
        gdf['source'] = 'model prediction'
        if save_shp:
            if dissolve_small_pols and (not use_class_dependent_area_thresholds):
                name_file = f'{model_name}_{tile_name}_LC-prediction_dissolved_{area_threshold}m2'
            elif dissolve_small_pols and use_class_dependent_area_thresholds:
                if name_combi_area_thresholds is None:
                    name_combi_area_thresholds = 'custom'
                    print('WARNING: name_combi_area_thresholds not specified, using "custom" as default')
                name_file = f'{model_name}_{tile_name}_LC-prediction_dissolved-{name_combi_area_thresholds}'
            else:
                name_file = f'{model_name}_{tile_name}_LC-prediction'
            save_path = os.path.join(save_folder, name_file)
            gdf.to_file(save_path)
            if verbose > 0:
                print(f'Saved {name_file} with {len(gdf)} polygons to {save_path}')
    else:
        gdf = None

    if save_raster:
        assert False, 'raster saving not yet implemented'
    
    return mask_tile, gdf, shape_predicted_tile_part

def tile_prediction_wrapper(model, trainer=None, dir_im='', list_tile_names_to_predict=None,
                            dir_mask_eval=None, mask_suffix='_lc_2022_mask.tif',
                             patch_size=512, padding=0, save_shp=False, save_raster=False, save_folder=None,
                             dissolve_small_pols=False, area_threshold=100, 
                             use_class_dependent_area_thresholds=False,
                            class_dependent_area_thresholds=dict(),
                             name_combi_area_thresholds=None,
                             skip_factor=None, 
                             clip_to_main_class=False, main_class_clip_label='C', col_name_class=None,
                             parent_dir_tile_mainpred='/home/tplas/predictions/predictions_LCU_2023-01-23-2018_dissolved1000m2_padding44_FGH-override/',
                             tile_outlines_shp_path='../content/evaluation_sample_50tiles/evaluation_sample_50tiles.shp',
                             subsample_tiles_for_testing=False):
    '''Wrapper function that predicts & reconstructs full tile.'''
    if padding > 0:
        assert type(padding) == int and padding % 2 == 0, 'Padding should be even number'
    ## Get list of all image tiles to predict
    list_tiff_tiles = lca.get_all_tifs_from_subdirs(dir_im)
    if list_tile_names_to_predict is not None:
        list_tiff_tiles = [tif for tif in list_tiff_tiles if tif.split('/')[-1][:6] in list_tile_names_to_predict]
    if subsample_tiles_for_testing:
        print('WARNING: subsampling 2 tiles for testing')
        list_tiff_tiles = list_tiff_tiles[:2]
    print(f'Loaded {len(list_tiff_tiles)} tiffs from subdirs of {dir_im}')
    
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

    if use_class_dependent_area_thresholds:
        dissolve_small_pols = True

    if area_threshold == 0 and dissolve_small_pols is True and (not use_class_dependent_area_thresholds):
        dissolve_small_pols = False
        print('WARNING: area_threshold is 0, so no polygons will be dissolved.')
    if area_threshold > 0 and dissolve_small_pols is False:
        print('Warning: area_threshold is > 0, but dissolve_small_pols is False, so NOT dissolving small polygons.')
    if dissolve_small_pols:
        print('Dissolving small polygons. WARNING: this takes considerable overhead computing time')

    if save_shp:
        if save_folder is None:
            save_folder = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/117574_20221122/tile_masks_predicted/predictions_LCU_2022-11-30-1205_dissolved_1000m2'
            print(f'No save folder given, so saving to {save_folder}')
        elif not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print(f'Saving to {save_folder}')
    
    ## Save meta data:
    if save_shp or save_raster:
        with open(os.path.join(save_folder, 'prediction_meta_data.txt'), 'w') as f:
            f.write(f'Prediction of {model} on {len(list_tiff_tiles)}.\n')
            f.write(f'skip_factor: {skip_factor}\n')
            f.write(f'dissolve_small_pols: {dissolve_small_pols}\n')
            f.write(f'area_threshold: {area_threshold}\n')
            f.write(f'use_class_dependent_area_thresholds: {use_class_dependent_area_thresholds}\n')
            f.write(f'class_dependent_area_thresholds: {class_dependent_area_thresholds}\n')
            f.write(f'patch_size: {patch_size}\n')
            f.write(f'padding: {padding}\n')
            f.write(f'dir_im: {dir_im}\n')
            f.write(f'dir_mask_eval: {dir_mask_eval}\n')

    df_schema = lca.create_df_mapping_labels_2022_to_80s()

    ## Loop across tiles:
    print(f'Predicting {len(list_tiff_tiles)} tiles')
    for i_tile, tilepath in tqdm(enumerate(list_tiff_tiles)):
        tilename = tilepath.split('/')[-1].rstrip('.tif')
        # if tilename not in ['SE0503', 'SK1398', 'SK0988', 'SK0896', 'SK2091']:
        #     continue
        mask_tile, mask_shp, shape_predicted_tile = prediction_one_tile(model=model, tilepath=tilepath, trainer=trainer, verbose=0,
                                                      save_shp=save_shp, save_raster=save_raster, save_folder=save_folder,
                                                      create_shp=True, patch_size=patch_size, padding=padding, tilename=tilename,
                                                      clip_to_main_class=clip_to_main_class, main_class_clip_label=main_class_clip_label, 
                                                      col_name_class=col_name_class,
                                                      parent_dir_tile_mainpred=parent_dir_tile_mainpred,
                                                      tile_outlines_shp_path=tile_outlines_shp_path,                             
                                                      dissolve_small_pols=dissolve_small_pols, area_threshold=area_threshold,
                                                      use_class_dependent_area_thresholds=use_class_dependent_area_thresholds,
                                                      class_dependent_area_thresholds=class_dependent_area_thresholds,
                                                      name_combi_area_thresholds=name_combi_area_thresholds,
                                                      df_schema=df_schema)

        if dir_mask_eval is not None:
            tile_path_mask = os.path.join(dir_mask_eval, tilename + mask_suffix)
            mask_tile_true = np.squeeze(lca.load_tiff(tile_path_mask, datatype='np'))
            mask_tile = mask_tile.to_numpy()

            ## Cut off no-class edge
            assert mask_tile_true.shape == mask_tile.shape, f'Predicted mask shape {mask_tile.shape} does not match true mask shape {mask_tile_true.shape}'
            half_pad = padding // 2
            mask_tile = mask_tile[half_pad:(shape_predicted_tile[0] - half_pad), :][:, half_pad:(shape_predicted_tile[1] - half_pad)]
            assert np.sum(mask_tile == 0) == 0, f'Padding not removed correctly OR predicted mask contains no-class pixels in tile {tilename} (index {i_tile})'
            mask_tile_true = mask_tile_true[half_pad:(shape_predicted_tile[0] - half_pad), :][:, half_pad:(shape_predicted_tile[1] - half_pad)]

            ## Compute confusion matrix:
            conf_mat = lca.compute_confusion_mat_from_two_masks(mask_true=mask_tile_true, mask_pred=mask_tile, 
                                                        lc_class_name_list=model.dict_training_details['class_name_list'], 
                                                         skip_factor=skip_factor)
            tmp = lcv.plot_confusion_summary(conf_mat=conf_mat, class_name_list=model.dict_training_details['class_name_list'], 
                                             plot_results=False, normalise_hm=True)
            dict_df_stats[tilename], dict_acc[tilename], _, __ = tmp 
            dict_conf_mat[tilename] = conf_mat

    return dict_acc, dict_df_stats, dict_conf_mat

def two_stage_patch_prediction(model_main, dict_models_detailed, 
                               batch_im, dict_class_names_detailed, skip_first_noclass=True,
                               dict_inds_mainclass_detailed={1: 'C', 2: 'D', 3: 'E'},
                               verbose=0):
    '''Predicts patch with two stage approach.
    '''
    pred_main = model_main.forward(batch_im)
    pred_main = lca.change_tensor_to_max_class_prediction(pred=pred_main)

    mapping_det_classes = {}  # so that they dont overlap
    if skip_first_noclass:
        count = 1
        total_cn_list = ['NO CLASS']
    else:
        count = 0
        total_cn_list = []
    for key, cn_list in dict_class_names_detailed.items():
        mapping_det_classes[key] = {}
        assert type(cn_list) == list, f'Class name list for {key} is not a list, but {type(cn_list)}'
        if skip_first_noclass:
            cn_list = cn_list[1:]
            mapping_det_classes[key][0] = 0
        dict_tmp_override_duplicates = {}
        for i, cn in enumerate(cn_list):
            if cn in total_cn_list:
                ind_cn = total_cn_list.index(cn)
                dict_tmp_override_duplicates[i] = ind_cn
                if verbose > 0:
                    print('dupcliate', i, cn, ind_cn)
        for i in range(len(cn_list)):
            if skip_first_noclass:
                i_use = i + 1
            else:
                i_use = i
            if i in dict_tmp_override_duplicates.keys():
                mapping_det_classes[key][i_use] = dict_tmp_override_duplicates[i]
            else:
                mapping_det_classes[key][i_use] = count + i
        total_cn_list += [x for x in cn_list if x not in total_cn_list]
        count += len(cn_list)

    pred_det = {}
    for key, model_det in dict_models_detailed.items():
        pred_det[key] = model_det.forward(batch_im)
        pred_det[key] = lca.change_tensor_to_max_class_prediction(pred=pred_det[key])
        pred_det[key] = pred_det[key].clone()
        pred_det[key] = pred_det[key].apply_(lambda x: mapping_det_classes[key][x])

    pred_final = pred_main.clone()
    for i_class, main_class in dict_inds_mainclass_detailed.items():
        inds = pred_main == i_class
        pred_final[inds] = pred_det[main_class][inds]

    return pred_final, total_cn_list

def relabel_twostage_pred_to_testds(class_name_list_twostage, class_name_list_testds,
                                    pred_tensor):
    '''Relabels two stage prediction to test ds'''
    assert type(class_name_list_twostage) == list and type(class_name_list_testds) == list
    assert class_name_list_testds[0] == 'NO CLASS'
    assert class_name_list_twostage[0] == 'NO CLASS'
    assert type(pred_tensor) == torch.Tensor
    dict_mapping = {}
    for i, cn in enumerate(class_name_list_twostage):
        if cn in class_name_list_testds:
            dict_mapping[i] = class_name_list_testds.index(cn)
        else:
            dict_mapping[i] = 0

    pred_tensor = pred_tensor.clone()
    pred_tensor = pred_tensor.apply_(lambda x: dict_mapping[x])
    return pred_tensor  



def save_details_trainds_to_model(model, train_ds):
    '''Save details of train data set to model'''
    assert type(model) == LandCoverUNet, f'{type(model)} not recognised'
    assert type(train_ds) == DataSetPatches, f'{type(train_ds)} not recognised'
    assert type(model.dict_training_details) == dict 

    assert len(model.dict_training_details) == 0, f'training dictionary is not empty but contains {model.dict_training_details.keys()}. Consider a new function that adds info of second training procedure?'

    list_names_attrs = ['df_patches', 'im_dir', 'mask_dir', 'path_mapping_dict', 
                        'preprocessing_func', 'rgb_means', 'rgb_std', 'shuffle_order_patches', 
                        'frac_subsample', 'unique_labels_arr', 'mapping_label_to_new_dict', 
                        'class_name_list', 'n_classes', 'list_tile_names']

    for name_attr in list_names_attrs:  # add to model one by one:
        model.dict_training_details[name_attr] = getattr(train_ds, name_attr)

    print(f'Details of training data set {train_ds} have been added to {model}')