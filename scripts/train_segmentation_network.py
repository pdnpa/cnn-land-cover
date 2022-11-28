## File tor train a LCU

import os, sys
import datetime
import loadpaths
import land_cover_analysis as lca
import land_cover_visualisation as lcv
import land_cover_models as lcm
import torch
import pytorch_lightning as pl

path_dict = loadpaths.loadpaths()
lca.check_torch_ready(check_gpu=True, assert_versions=True)

## Parameters:
batch_size = 10
n_cpus = 8
n_max_epochs = 10
learning_rate = 1e-3
save_full_model = True
use_valid_ds = True
path_mapping_dict = '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__main_categories__2022-11-17-1512.pkl'

## Dirs training data:
# dir_ds = path_dict['tiles_few_changes_path']
dir_ds = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/'
dir_im_patches = os.path.join(dir_ds, 'images/')
dir_mask_patches = os.path.join(dir_ds, 'masks/')

## Dirs test data:
dir_test_im_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images'
dir_test_mask_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_2022/'

## Define model:
n_classes = 7
LCU = lcm.LandCoverUNet(n_classes=n_classes, lr=learning_rate)  # load model 

## Create train & validation dataloader:
train_ds = lcm.DataSetPatches(im_dir=dir_im_patches, mask_dir=dir_mask_patches, 
                              mask_suffix='_lc_80s_mask.npy',
                              preprocessing_func=LCU.preprocessing_func,
                              shuffle_order_patches=True, relabel_masks=True,
                              subsample_patches=False, path_mapping_dict=path_mapping_dict)
assert train_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'

if use_valid_ds:
    ## Create validation set:
    train_ds_size = int(len(train_ds) * 0.85)
    valid_ds_size = len(train_ds) - train_ds_size
    train_ds, valid_ds = torch.utils.data.random_split(train_ds, [train_ds_size, valid_ds_size])  #, generator=seed)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus)
if use_valid_ds:
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=n_cpus)

## Create test dataloader:
test_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                            mask_suffix='_lc_2022_mask.npy',
                            preprocessing_func=LCU.preprocessing_func, path_mapping_dict=path_mapping_dict,
                            shuffle_order_patches=True, relabel_masks=False,
                            subsample_patches=False)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=n_cpus)

## Save details to model:
lcm.save_details_trainds_to_model(model=LCU, train_ds=train_ds)
LCU.dict_training_details['batch_size'] = batch_size
LCU.dict_training_details['n_cpus'] = n_cpus 
LCU.dict_training_details['n_max_epochs'] = n_max_epochs
LCU.dict_training_details['learning_rate'] = learning_rate
LCU.dict_training_details['use_valid_ds'] = use_valid_ds

timestamp_start = datetime.datetime.now()
print(f'Training {LCU} in {n_max_epochs} epochs. Starting at {timestamp_start}\n')

## Train using PL API - saves automatically.
trainer = pl.Trainer(max_epochs=n_max_epochs, accelerator='gpu', devices=1)  # run on GPU; and set max_epochs.
if use_valid_ds:
    trainer.fit(model=LCU, train_dataloaders=train_dl, valid_dataloaders=valid_dl) 
else:
    trainer.fit(model=LCU, train_dataloaders=train_dl) 

timestamp_end = datetime.datetime.now() 
duration = timestamp_end - timestamp_start
LCU.dict_training_details['duration_training'] = duration 
print(f'Training finished at {timestamp_end}')

if save_full_model is False:
    LCU.base = None 
LCU.save_model()
LCU.eval() 

## Test on unseen evaluation set:
trainer.test(model=LCU, dataloaders=test_dl)