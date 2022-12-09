## File tor train a LCU

import os, sys, json
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
n_max_epochs = 15
learning_rate = 1e-3
loss_function = 'focal_loss'
save_full_model = True
use_valid_ds = True
path_mapping_dict = '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__main_categories__2022-11-17-1512.pkl'

## Dirs training data:
# dir_ds = path_dict['tiles_few_changes_path']
# dir_ds = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/'
# dir_im_patches = os.path.join(dir_ds, 'images/')
# dir_mask_patches = os.path.join(dir_ds, 'masks/')
dir_im_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images'
dir_mask_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_2022/'
with open('/home/tplas/repos/cnn-land-cover/content/evaluation_sample_50tiles/10_training_tiles_from_eval.json', 'r') as f:
    dict_tile_names_sample = json.load(f)
    
## Dirs test data:
dir_test_im_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images'
dir_test_mask_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_2022/'

## Define model:
n_classes = 7
LCU = lcm.LandCoverUNet(n_classes=n_classes, lr=learning_rate, loss_function=loss_function)  # load model 
LCU.change_description(new_description='10 training tiles from eval. 2022 masks.', add=True)

## Create train & validation dataloader:
train_ds = lcm.DataSetPatches(im_dir=dir_im_patches, mask_dir=dir_mask_patches, 
                              mask_suffix='_lc_2022_mask.npy',
                              list_tile_names=dict_tile_names_sample['sample'],
                              preprocessing_func=LCU.preprocessing_func,
                              shuffle_order_patches=True, relabel_masks=False,
                              subsample_patches=False, path_mapping_dict=path_mapping_dict)
assert train_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus)

if use_valid_ds:
    ## Create validation set:
    valid_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                mask_suffix='_lc_2022_mask.npy',
                              list_tile_names=dict_tile_names_sample['remainder'],
                                preprocessing_func=LCU.preprocessing_func,
                                shuffle_order_patches=True, relabel_masks=False,
                                subsample_patches=True, frac_subsample=0.1, 
                                path_mapping_dict=path_mapping_dict)
    assert valid_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=n_cpus)

## Create test dataloader:
test_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                            mask_suffix='_lc_2022_mask.npy',
                              list_tile_names=dict_tile_names_sample['remainder'],
                            preprocessing_func=LCU.preprocessing_func, path_mapping_dict=path_mapping_dict,
                            shuffle_order_patches=True, relabel_masks=False,
                            subsample_patches=False)

test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=n_cpus)

## Save details to model:
## TODO: unique labels array isn't correct (range(39) instaed of range(7) for main cats)
lcm.save_details_trainds_to_model(model=LCU, train_ds=train_ds)
LCU.dict_training_details['batch_size'] = batch_size
LCU.dict_training_details['n_cpus'] = n_cpus 
LCU.dict_training_details['n_max_epochs'] = n_max_epochs
LCU.dict_training_details['learning_rate'] = learning_rate
LCU.dict_training_details['use_valid_ds'] = use_valid_ds
LCU.dict_training_details['loss_function'] = loss_function

timestamp_start = datetime.datetime.now()
print(f'Training {LCU} in {n_max_epochs} epochs. Starting at {timestamp_start}\n')

## Train using PL API - saves automatically.
trainer = pl.Trainer(max_epochs=n_max_epochs, accelerator='gpu', devices=1)  # run on GPU; and set max_epochs.
if use_valid_ds:
    # trainer.fit(model=LCU, train_dataloaders=train_dl, valid_dataloaders=valid_dl) 
    trainer.fit(LCU, train_dl, valid_dl) 
else:
    trainer.fit(model=LCU, train_dataloaders=train_dl) 

timestamp_end = datetime.datetime.now() 
duration = timestamp_end - timestamp_start
LCU.dict_training_details['duration_training'] = duration 
print(f'Training finished at {timestamp_end}')

## Test on unseen evaluation set:
LCU.eval() 
trainer.test(model=LCU, dataloaders=test_dl)

## Save:
if save_full_model is False:
    LCU.base = None 
LCU.save_model()  

#TODO save name to model