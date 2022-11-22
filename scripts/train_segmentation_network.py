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
n_max_epochs = 100
learning_rate = 1e-3
save_full_model = True
path_mapping_dict = '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__main_categories__2022-11-17-1512.pkl'

## Dirs training data:
dir_ds = path_dict['tiles_few_changes_path']
dir_im_patches = os.path.join(dir_ds, 'images/')
dir_mask_patches = os.path.join(dir_ds, 'masks/')

## Define model:
n_classes = 7
LCU = lcm.LandCoverUNet(n_classes=n_classes, lr=learning_rate)  # load model 

## Create train dataloader:
train_ds = lcm.DataSetPatches(im_dir=dir_im_patches, mask_dir=dir_mask_patches, 
                              preprocessing_func=LCU.preprocessing_func,
                              subsample_patches=False, path_mapping_dict=path_mapping_dict)
assert train_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus)
lcm.save_details_trainds_to_model(model=LCU, train_ds=train_ds)
lcm.dict_details_training['batch_size'] = batch_size
lcm.dict_details_training['n_cpus'] = n_cpus 
lcm.dict_details_training['n_max_epochs'] = n_max_epochs
lcm.dict_details_training['learning_rate'] = learning_rate

timestamp_start = datetime.datetime.now()
print(f'Training {LCU} in {n_max_epochs} epochs. Starting at {timestamp_start}\n')

## Train using PL API - saves automatically.
trainer = pl.Trainer(max_epochs=n_max_epochs, accelerator='gpu', devices=1)  # run on GPU; and set max_epochs.
trainer.fit(model=LCU, train_dataloaders=train_dl)  # could include validation set here to determine convergence

timestamp_end = datetime.datetime.now() 
duration = timestamp_end - timestamp_start
lcm.dict_details_training['duration_training'] = duration 
print(f'Training finished at {timestamp_end}')

if save_full_model is False:
    LCU.base = None 
LCU.save_model()
