## File tor train a LCU

import os, sys, json, pickle
import datetime
import loadpaths
import land_cover_analysis as lca
# import land_cover_visualisation as lcv
import land_cover_models as lcm
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

path_dict = loadpaths.loadpaths()
lca.check_torch_ready(check_gpu=True, assert_versions=True)
tb_logger = pl_loggers.TensorBoardLogger(save_dir='/home/tplas/models/')
# pl.seed_everything(86, workers=True)

## Parameters:
batch_size = 10
n_cpus = 8
n_max_epochs = 10
optimise_learning_rate = False
transform_training_data = True
learning_rate = 1e-3
loss_function = 'focal_loss'
save_full_model = True
mask_suffix_train = '_lc_nfi_mask.npy'
mask_dir_name_train = 'masks_nfi'  # only relevant if no dir_mask_patches is given
use_valid_ds = False
evaluate_on_test_ds = False
# path_mapping_dict = '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__main_categories__2022-11-17-1512.pkl'
path_mapping_dict = '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__C_subclasses_only__2023-02-01-1518.pkl'
# path_mapping_dict = '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__D_subclasses_only__2023-02-09-1449.pkl'

## Dirs training data:
# dir_im_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images'
# dir_mask_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_2022/'
# with open('/home/tplas/repos/cnn-land-cover/content/evaluation_sample_50tiles/10_training_tiles_from_eval.json', 'r') as f:
#     dict_tile_names_sample = json.load(f)  # give tile names to use 
    
# dir_im_patches = ['/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/images/']#,
#                 #   '/home/tplas/data/gis/most recent APGB 12.5cm aerial/urban_tiles/images/']  # give multiple folders 
dir_mask_patches = None   # auto find masks 

dir_im_patches = ['/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/images/',
                  '/home/tplas/data/gis/most recent APGB 12.5cm aerial/forest_tiles_2/images/']

# dir_im_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/images/'
# dir_mask_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/masks_nfi/'

## Dirs test data:
dir_test_im_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images'
dir_test_mask_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_2022/'

## Define model:
tmp_path_dict = pickle.load(open(path_mapping_dict, 'rb'))
n_classes = len(tmp_path_dict['dict_new_names'])
LCU = lcm.LandCoverUNet(n_classes=n_classes, lr=learning_rate, loss_function=loss_function)  # load model 
LCU.change_description(new_description='C only. 11 training tiles CDE using NFI', add=True)

## Create train & validation dataloader:
print('\nCreating train dataloader...')
train_ds = lcm.DataSetPatches(im_dir=dir_im_patches, mask_dir=dir_mask_patches, 
                              mask_suffix=mask_suffix_train, mask_dir_name=mask_dir_name_train,
                            #   list_tile_names=dict_tile_names_sample['sample'],
                              preprocessing_func=LCU.preprocessing_func,
                              shuffle_order_patches=True, relabel_masks=True,
                              subsample_patches=False, path_mapping_dict=path_mapping_dict,
                              random_transform_data=transform_training_data)
train_ds.remove_no_class_patches()  # remove all patches that have no class                              
assert train_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus)

if use_valid_ds:
    ## Create validation set:
    print('\nCreating validation dataloader...')
    valid_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                mask_suffix='_lc_2022_mask.npy',
                            #   list_tile_names=dict_tile_names_sample['remainder'],
                                preprocessing_func=LCU.preprocessing_func,
                                shuffle_order_patches=True, relabel_masks=False,
                                subsample_patches=True, frac_subsample=0.1, 
                                path_mapping_dict=path_mapping_dict)
    assert valid_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=n_cpus)

## Create test dataloader:
if evaluate_on_test_ds:
    print('\nCreating test dataloader...')
    test_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                mask_suffix='_lc_2022_mask.npy',
                                #   list_tile_names=dict_tile_names_sample['remainder'],
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
trainer = pl.Trainer(max_epochs=n_max_epochs, accelerator='gpu', devices=1, logger=tb_logger)#, auto_lr_find='lr')  # run on GPU; and set max_epochs.
# # no accumulation for epochs 1-4. accumulate 3 for epochs 5-10. accumulate 20 after that
# trainer = Trainer(accumulate_grad_batches={5: 3, 10: 20})

## Optimise learning rate:
if optimise_learning_rate:
    lr_finder = trainer.tuner.lr_find(LCU, train_dataloaders=train_dl, val_dataloaders=valid_dl, num_training=100)
    print(f'Optimised learning rate to {lr_finder.suggestion()}')
    LCU.lr = lr_finder.suggestion() 
    # bs_finder = trainer.tuner.scale_batch_size(LCU, train_dataloaders=train_dl, val_dataloaders=valid_dl)
    # print(f'Optimised learning rate to {bs_finder.suggestion()}')

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
if evaluate_on_test_ds:
    trainer.test(model=LCU, dataloaders=test_dl)

## Save:
if save_full_model is False:  # to save memory, don't save weights
    LCU.base = None 
LCU.save_model()  