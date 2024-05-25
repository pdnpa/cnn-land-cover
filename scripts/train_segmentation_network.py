## File tor train a LCU

import os, pickle
import datetime
import land_cover_analysis as lca
import loadpaths
import land_cover_models as lcm
import custom_losses as cl
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from prediction_of_trained_network import predict_segmentation_network
import argparse

# Setup if using different training folders
parser = argparse.ArgumentParser(description='Train segmentation network')
parser.add_argument('--dir_im_patches', type=str, help='Directory containing image patches', required=True)
args = parser.parse_args()  # Parse arguments


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # double check GPU ID
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # catch errors during memory allocation
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # For TensorFlow compatibility

path_dict = loadpaths.loadpaths()
    
def train_segmentation_network(
        batch_size=5, # /david/ cuda memory issue
        n_cpus=16, # set to 16 /david/
        use_mac_sil=False,
        n_max_epochs=30,
        optimise_learning_rate=False,
        transform_training_data=True,
        learning_rate=1e-3,
        dissolve_small_pols=True,
        dissolve_threshold=1000,
        loss_function='cross_entropy',  # 'focal_loss'
        encoder_name='resnet50',  #'efficientnet-b1'
        save_full_model=True,
        mask_suffix_train='_lc_2022_detailed_mask.npy',
        mask_suffix_test_ds='_lc_2022_detailed_mask.npy',
        use_valid_ds=True,
        evaluate_on_test_ds=True,
        perform_and_save_predictions=False,
        clip_to_main_class=True,
        tile_patch_train_test_split_dict_path='../content/evaluation_sample_50tiles/train_test_split_80tiles_2023-03-22-2131.pkl',
        main_class_clip_label='D',
        description_model='D class training using habitat data. Focal loss resnet 30 epochs',
        path_mapping_dict="../content/label_mapping_dicts/label_mapping_dict__all_relevant_subclasses__2023-04-20-1540.pkl", # 2023-03-10-1154.pkl /david/
        dir_im_patches=path_dict['im_patches'],
        dir_mask_patches=None,  # if None, mask_dir_name_train is used
        dir_test_im_patches=None,  # if None, dir_im_patches is used    
        dir_test_mask_patches=None, # if None, mask_dir_name_test is used
        mask_dir_name_train='masks_python_all',  # only relevant if no dir_mask_patches is given
        mask_dir_name_test='masks_python_all',  # only relevant if no dir_mask_patches is given
        dir_tb=path_dict['models'],
        n_bands=3
                                ):
    
    # Adjust to accept the correct number of input bands
    if n_bands == 3:
        pass  
    elif n_bands == 4:
        original_weight = LCU.resnet.conv1.weight.clone()
        LCU.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            LCU.resnet.conv1.weight[:, :3] = original_weight
            LCU.resnet.conv1.weight[:, 3] = original_weight[:, 0]  # Copy weights from the first channel
    else:
        raise ValueError(f'Unsupported number of bands: {n_bands}')

    if tile_patch_train_test_split_dict_path is not None:
        with open(tile_patch_train_test_split_dict_path, 'rb') as f:
            dict_tile_patches = pickle.load(f)
            tile_patch_train = dict_tile_patches['train']
            tile_patch_test = dict_tile_patches['test']
    else:
        tile_patch_train = None
        tile_patch_test = None

    assert os.path.exists(dir_im_patches), f'Path to image patches does not exist: {dir_im_patches}'
    if dir_test_im_patches is None:
        dir_test_im_patches = dir_im_patches

    if use_mac_sil:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=dir_tb)
        n_cpus = 12
        acc_use = 'gpu'
        lca.check_torch_ready(check_mps=False, check_gpu=False, assert_versions=True)
    else:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=dir_tb)
        n_cpus = 8
        acc_use = 'gpu'
        lca.check_torch_ready(check_gpu=True, assert_versions=True)
    folder_save = dir_tb
    # pl.seed_everything(86, workers=True)

    ## Define model:
    print("Path to mapping dictionary:", path_mapping_dict)
    tmp_path_dict = pickle.load(open(path_mapping_dict, 'rb'))
    n_classes = len(tmp_path_dict['dict_new_names'])

    in_channels = 4 if n_bands == 4 else 3 

    LCU = lcm.LandCoverUNet(n_classes=n_classes, lr=learning_rate, in_channels=in_channels,
                            loss_function=loss_function, encoder_name=encoder_name)
    LCU.change_description(new_description=description_model, add=True)

    #LCU = lcm.LandCoverUNet(n_classes=n_classes, lr=learning_rate, 
    #                        loss_function=loss_function, encoder_name=encoder_name)  # load model 
    
    LCU.change_description(new_description=description_model, add=True)

    ## Create train & validation dataloader:
    print('\nCreating train dataloader...')
    train_ds = lcm.DataSetPatches(im_dir=dir_im_patches, mask_dir=dir_mask_patches, 
                                mask_suffix=mask_suffix_train, mask_dir_name=mask_dir_name_train,
                                #   list_tile_names=dict_tile_names_sample['train'],
                                list_tile_patches_use=tile_patch_train,
                                preprocessing_func=LCU.preprocessing_func,
                                shuffle_order_patches=True, relabel_masks=True,
                                subsample_patches=False, path_mapping_dict=path_mapping_dict,
                                random_transform_data=transform_training_data)
    train_ds.remove_no_class_patches()  # remove all patches that have no class                              
    assert train_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus, persistent_workers=True)

    assert LCU.n_classes == train_ds.n_classes, f'LCU has {LCU.n_classes} classes but train DS has {train_ds.n_classes} classes'  # Defined in LCU by arg, in train_ds automatically from data
    if train_ds.class_name_list[0] in ['NO CLASS', '0']:
        LCU.first_class_is_no_class = True  # for accuracy calculation
    assert LCU.first_class_is_no_class, 'First class is not no class, check if this is correct'

    if use_valid_ds:
        ## Create validation set:mask_dir_name=mask_dir_name_test,
        print('\nCreating validation dataloader...')
        valid_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds, mask_dir_name=mask_dir_name_test,
                                #   list_tile_names=dict_tile_names_sample['test'],
                                    list_tile_patches_use=tile_patch_test,
                                    preprocessing_func=LCU.preprocessing_func,
                                    shuffle_order_patches=True, relabel_masks=True,
                                    subsample_patches=False, # frac_subsample=0.1, 
                                    path_mapping_dict=path_mapping_dict)
        valid_ds.remove_no_class_patches()
        assert valid_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=n_cpus, persistent_workers=True)

    ## Create test dataloader:
    if evaluate_on_test_ds:
        print('\nCreating test dataloader...')
        test_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds, mask_dir_name=mask_dir_name_test,
                                #   list_tile_names=dict_tile_names_sample['test'],
                                    list_tile_patches_use=tile_patch_test,
                                    preprocessing_func=LCU.preprocessing_func, 
                                    shuffle_order_patches=True, relabel_masks=True,
                                    subsample_patches=False,
                                    path_mapping_dict=path_mapping_dict)
        test_ds.remove_no_class_patches()
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=n_cpus, persistent_workers=True)

    assert LCU.n_classes == train_ds.n_classes, f'LCU has {LCU.n_classes} classes but train DS has {train_ds.n_classes} classes'  # Defined in LCU by arg, in train_ds automatically from data
    # assert LCU.n_classes == len(train_ds.list_unique_classes),  'Not all classes occur in train DS (or vice versa)'
    # assert (train_ds.list_unique_classes == test_ds.list_unique_classes).all(), 'Train and test DS have different classes'
    # if use_valid_ds:
    #     assert (train_ds.list_unique_classes == valid_ds.list_unique_classes).all(), 'Train and valid DS have different classes'

    ## Save details to model:
    lcm.save_details_trainds_to_model(model=LCU, train_ds=train_ds)
    LCU.dict_training_details['batch_size'] = batch_size
    LCU.dict_training_details['n_cpus'] = n_cpus 
    LCU.dict_training_details['n_max_epochs'] = n_max_epochs
    LCU.dict_training_details['learning_rate'] = learning_rate
    LCU.dict_training_details['use_valid_ds'] = use_valid_ds
    LCU.dict_training_details['loss_function'] = loss_function
    LCU.dict_training_details['tile_patch_train_test_split_dict_path'] = tile_patch_train_test_split_dict_path

    timestamp_start = datetime.datetime.now()
    print(f'Training {LCU} in {n_max_epochs} epochs. Starting at {timestamp_start}\n')

    ## Train using PL API - saves automatically.
    cb_metrics = cl.MetricsCallback()
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min',
                                            filename="best_checkpoint_val-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}"),
                 pl.callbacks.ModelCheckpoint(monitor='train_loss', save_top_k=1, mode='min',
                                            filename="best_checkpoint_train-{epoch:02d}-{val_loss:.2f}-{train_loss:.2f}"),
                #  pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                 cb_metrics]
    trainer = pl.Trainer(max_epochs=n_max_epochs, accelerator=acc_use, devices=1, 
                         logger=tb_logger, callbacks=callbacks)#, auto_lr_find='lr')  # run on GPU; and set max_epochs.
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
    path_lcu = LCU.save_model(folder=folder_save, metrics=cb_metrics.metrics)  

    if perform_and_save_predictions:
        predict_segmentation_network(datapath_model=path_lcu.lstrip(folder_save),
                                     clip_to_main_class=clip_to_main_class, 
                                     main_class_clip_label=main_class_clip_label,
                                     dissolve_small_pols=dissolve_small_pols,
                                     dissolve_threshold=dissolve_threshold,
                                     dir_mask_eval=None)

if __name__ == '__main__':
    # Setup command-line argument parsing
    loss_functions_list = [
        'cross_entropy', 
        # 'focal_loss'
                          ] 
    mapping_dicts_list = [
          '../content/label_mapping_dicts/label_mapping_dict__all_relevant_subclasses__2023-04-20-1540.pkl'
        # '../content/label_mapping_dicts/label_mapping_dict__C_subclasses_only__2023-04-20-1540.pkl',
        # '../content/label_mapping_dicts/label_mapping_dict__D_subclasses_only__2023-04-20-1540.pkl',
        # '../content/label_mapping_dicts/label_mapping_dict__E_subclasses_and_F3d_only__2023-04-20-1541.pkl',
        # '../content/label_mapping_dicts/label_mapping_dict__main_categories_F3inDE_noFGH__2023-04-21-1315.pkl'
                         ]
    list_encoder_names = [
        #'resnet50' 
         'resnet18'
        # 'efficientnet-b1'
                         ]
    n_repetitions = 1
    
    ## loop through all combinations of loss functions and mapping dicts:
    print('starting training')
    count = -1
    for i in range(n_repetitions):
        for current_encoder_name in list_encoder_names:
            for current_loss_function in loss_functions_list:
                for current_mapping_dict in mapping_dicts_list:
                    count += 1
                    print(f'\n\n\nIteration {i + 1}/{n_repetitions} of loss function {current_loss_function}, encoder {current_encoder_name}, mapping {current_mapping_dict.split("/")[-1].split("__")[1]} \n\n\n')
                    
                    train_segmentation_network(
                        use_mac_sil=False,
                        batch_size=5,
                        loss_function=current_loss_function,
                        #dir_im_patches='/home/david/Documents/ADP/pd_lc_annotated_patches_data/python_format/images_python_all/',
                        dir_im_patches=args.dir_im_patches,
                        perform_and_save_predictions=False,
                        # main_class_clip_label='E',
                        clip_to_main_class=False,
                        dissolve_small_pols=False,
                        dissolve_threshold=20,
                        n_max_epochs=10,
                        encoder_name=current_encoder_name,
                        path_mapping_dict=current_mapping_dict,
                        description_model=f'{current_mapping_dict.split("/")[-1].split("__")[1]} training using randomly split eval patch data. {current_loss_function} {current_encoder_name} 60 epochs'
                    )