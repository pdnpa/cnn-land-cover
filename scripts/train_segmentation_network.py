## File tor train a LCU

import os, sys, json, pickle
import datetime
import land_cover_analysis as lca
# import land_cover_visualisation as lcv
import land_cover_models as lcm
import custom_losses as cl
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from prediction_of_trained_network import predict_segmentation_network
    
def train_segmentation_network(
        batch_size=10,
        n_cpus=16, # set to 16 /david/
        n_max_epochs=30,
        optimise_learning_rate=False,
        transform_training_data=True,
        learning_rate=1e-3,
        dissolve_small_pols=True,
        dissolve_threshold=1000,
        loss_function='focal_loss',  # 'cross_entropy'
        encoder_name='resnet50',  #'efficientnet-b1'
        save_full_model=True,
        mask_suffix_train='_lc_hab_mask.npy',
        mask_suffix_test_ds='_lc_2022_detailed_mask.npy',
        mask_dir_name_train='masks_detailed_annotation',  # only relevant if no dir_mask_patches is given
        mask_dir_name_test='masks_detailed_annotation',  # only relevant if no dir_mask_patches is given
        use_valid_ds=True,
        evaluate_on_test_ds=True,
        perform_and_save_predictions=False,
        clip_to_main_class=True,
        tile_patch_train_test_split_dict_path=None,  # '../content/evaluation_sample_50tiles/train_test_split_80tiles.pkl'
        main_class_clip_label='D',
        description_model='D class training using habitat data. Focal loss resnet 30 epochs',
        #path_mapping_dict='/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__D_subclasses_only__2023-03-10-1154.pkl',
        #dir_im_patches='/home/tplas/data/gis/habitat_training/images/',
        #dir_mask_patches='/home/tplas/data/gis/habitat_training/masks_hab/',
        #dir_test_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images_detailed_annotation/',
        #dir_test_mask_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_detailed_annotation/'
        path_mapping_dict='/home/david/Documents/GitHub/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__D_subclasses_only__2023-03-10-1154.pkl',
        dir_im_patches='/home/tplas/data/gis/habitat_training/images/',
        dir_mask_patches='/home/tplas/data/gis/habitat_training/masks_hab/',
        dir_test_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images_detailed_annotation/',
        dir_test_mask_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_detailed_annotation/'
                                ):

    if tile_patch_train_test_split_dict_path is not None:
        with open(tile_patch_train_test_split_dict_path, 'rb') as f:
            dict_tile_patches = pickle.load(f)
            tile_patch_train = dict_tile_patches['train']
            tile_patch_test = dict_tile_patches['test']
    else:
        tile_patch_train = None
        tile_patch_test = None

    lca.check_torch_ready(check_gpu=True, assert_versions=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='/home/david/models/') # set to /david/
    # pl.seed_everything(86, workers=True)

    ## Define model:
    tmp_path_dict = pickle.load(open(path_mapping_dict, 'rb'))
    n_classes = len(tmp_path_dict['dict_new_names'])
    LCU = lcm.LandCoverUNet(n_classes=n_classes, lr=learning_rate, 
                            loss_function=loss_function, encoder_name=encoder_name)  # load model 
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
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=n_cpus)

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
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=n_cpus)

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
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=n_cpus)

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
    trainer = pl.Trainer(max_epochs=n_max_epochs, accelerator='gpu', devices=1, 
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
    path_lcu = LCU.save_model(metrics=cb_metrics.metrics)  

    if perform_and_save_predictions:
        predict_segmentation_network(datapath_model=path_lcu.lstrip('/home/tplas/models'),
                                     clip_to_main_class=clip_to_main_class, 
                                     main_class_clip_label=main_class_clip_label,
                                     dissolve_small_pols=dissolve_small_pols,
                                     dissolve_threshold=dissolve_threshold,
                                     dir_mask_eval=None)

if __name__ == '__main__':
    loss_functions_list = [
        'cross_entropy', 
        # 'focal_loss'
                          ] 
    mapping_dicts_list = [
          '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__all_relevant_subclasses__2023-04-20-1540.pkl'
        # '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__C_subclasses_only__2023-04-20-1540.pkl',
        # '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__D_subclasses_only__2023-04-20-1540.pkl',
        # '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__E_subclasses_and_F3d_only__2023-04-20-1541.pkl',
        # '/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__main_categories_F3inDE_noFGH__2023-04-21-1315.pkl'
                         ]
    list_encoder_names = [
        'resnet50' 
        # 'efficientnet-b1'
                         ]
    n_repetitions = 5
    count = -1

    ## loop through all combinations of loss functions and mapping dicts:
    print('starting training')
    for i in range(n_repetitions):
        for current_encoder_name in list_encoder_names:
            for current_loss_function in loss_functions_list:
                for current_mapping_dict in mapping_dicts_list:
                    count += 1
                    # if count < 36:
                    #     continue
                    print(f'\n\n\nIteration {i + 1}/{n_repetitions} of loss function {current_loss_function}, encoder {current_encoder_name}, mapping {current_mapping_dict.split("/")[-1].split("__")[1]} \n\n\n')
                    # train_segmentation_network(
                    #     loss_function=current_loss_function,
                    #     dir_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images_detailed_annotation/',
                    #     dir_mask_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_detailed_annotation/',
                    #     dir_test_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/eval_2_tiles/images_detailed_annotation/',
                    #     dir_test_mask_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/eval_2_tiles/masks_detailed_annotation/',
                    #     mask_suffix_train='_lc_2022_detailed_mask.npy',
                    #     mask_suffix_test_ds='_lc_2022_detailed_mask.npy',
                    #     perform_and_save_predictions=False,
                    #     # main_class_clip_label='E',
                    #     clip_to_main_class=False,
                    #     dissolve_small_pols=True,
                    #     dissolve_threshold=20,
                    #     n_max_epochs=60,
                    #     path_mapping_dict='/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__main_categories_F3inDE_noFGH__2023-03-17-0957.pkl',
                    #     # path_mapping_dict='../content/label_mapping_dicts/label_mapping_dict__main_categories_F3inDE_noFGH__2023-03-17-0957.pkl',
                    #     description_model=f'main class training using eval patch data. {current_loss_function} resnet 60 epochs'
                    # )

                    train_segmentation_network(
                        loss_function=current_loss_function,
                        dir_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/eval_all_tiles/images_detailed_annotation/',
                        dir_mask_patches=None,
                        dir_test_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/eval_all_tiles/images_detailed_annotation/',
                        dir_test_mask_patches=None,
                        mask_suffix_train='_lc_2022_detailed_mask.npy',
                        mask_suffix_test_ds='_lc_2022_detailed_mask.npy',
                        perform_and_save_predictions=False,
                        # main_class_clip_label='E',
                        clip_to_main_class=False,
                        dissolve_small_pols=True,
                        dissolve_threshold=20,
                        n_max_epochs=60,
                        encoder_name=current_encoder_name,
                        # tile_patch_train_test_split_dict_path='../content/evaluation_sample_50tiles/train_test_split_80tiles_2023-03-21-1600.pkl',
                        tile_patch_train_test_split_dict_path='../content/evaluation_sample_50tiles/train_test_split_80tiles_2023-03-22-2131.pkl',
                        path_mapping_dict=current_mapping_dict,
                        description_model=f'{current_mapping_dict.split("/")[-1].split("__")[1]} training using randomly split eval patch data. {current_loss_function} {current_encoder_name} 60 epochs'
                    )