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
from prediction_of_trained_network import predict_segmentation_network
    
def train_segmentation_network(
        batch_size=10,
        n_cpus=8,
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
        mask_dir_name_train='masks',  # only relevant if no dir_mask_patches is given
        use_valid_ds=True,
        evaluate_on_test_ds=True,
        perform_and_save_predictions=False,
        clip_to_main_class=True,
        main_class_clip_label='D',
        description_model='D class training using habitat data. Focal loss resnet 30 epochs',
        path_mapping_dict='/home/tplas/repos/cnn-land-cover/content/label_mapping_dicts/label_mapping_dict__D_subclasses_only__2023-03-10-1154.pkl',
        dir_im_patches='/home/tplas/data/gis/habitat_training/images/',
        dir_mask_patches='/home/tplas/data/gis/habitat_training/masks_hab/',
        dir_test_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images',
        dir_test_mask_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_detailed_annotation/'
                                ):

    ## Dirs training data:
    # dir_im_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images'
    # dir_mask_patches = '/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_2022/'
    # with open('/home/tplas/repos/cnn-land-cover/content/evaluation_sample_50tiles/10_training_tiles_from_eval.json', 'r') as f:
    #     dict_tile_names_sample = json.load(f)  # give tile names to use 
        
    # dir_im_patches = ['/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/images/']#,
    #                 #   '/home/tplas/data/gis/most recent APGB 12.5cm aerial/urban_tiles/images/']  # give multiple folders 
    # dir_mask_patches = None   # auto find masks 

    # dir_im_patches = ['/home/tplas/data/gis/most recent APGB 12.5cm aerial/CDE_training_tiles/images/',
    #                 '/home/tplas/data/gis/most recent APGB 12.5cm aerial/forest_tiles_2/images/']
    
    lca.check_torch_ready(check_gpu=True, assert_versions=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir='/home/tplas/models/')
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
                                #   list_tile_names=dict_tile_names_sample['sample'],
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

    if use_valid_ds:
        ## Create validation set:
        print('\nCreating validation dataloader...')
        valid_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds,
                                    preprocessing_func=LCU.preprocessing_func,
                                    shuffle_order_patches=True, relabel_masks=True,
                                    subsample_patches=False, frac_subsample=0.1, 
                                    path_mapping_dict=path_mapping_dict)
        valid_ds.remove_no_class_patches()
        assert valid_ds.n_classes == n_classes, f'Train DS has {train_ds.n_classes} classes but n_classes for LCU set to {n_classes}'
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, num_workers=n_cpus)

    ## Create test dataloader:
    if evaluate_on_test_ds:
        print('\nCreating test dataloader...')
        test_ds = lcm.DataSetPatches(im_dir=dir_test_im_patches, mask_dir=dir_test_mask_patches, 
                                    mask_suffix=mask_suffix_test_ds,
                                    preprocessing_func=LCU.preprocessing_func, 
                                    shuffle_order_patches=True, relabel_masks=True,
                                    subsample_patches=False,
                                    path_mapping_dict=path_mapping_dict)
        test_ds.remove_no_class_patches()
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=n_cpus)

    ## Save details to model:
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
    path_lcu = LCU.save_model()  

    if perform_and_save_predictions:
        predict_segmentation_network(datapath_model=path_lcu.lstrip('/home/tplas/models'),
                                     clip_to_main_class=clip_to_main_class, 
                                     main_class_clip_label=main_class_clip_label,
                                     dissolve_small_pols=dissolve_small_pols,
                                     dissolve_threshold=dissolve_threshold,
                                     dir_mask_eval=None)

if __name__ == '__main__':
    loss_functions_list = ['focal_loss']
    for current_loss_function in loss_functions_list:
        print(f'\n\n\nNEW LOSS FUNCTION {current_loss_function}\n\n\n')
        train_segmentation_network(
            loss_function=current_loss_function,
            dir_im_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/images/',
            dir_mask_patches='/home/tplas/data/gis/most recent APGB 12.5cm aerial/evaluation_tiles/masks_detailed_annotation/',
            mask_suffix_train='_lc_2022_detailed_mask.npy',
            perform_and_save_predictions=True,
            # main_class_clip_label='E',
            clip_to_main_class=False,
            dissolve_small_pols=True,
            dissolve_threshold=20,
            n_max_epochs=120,
            path_mapping_dict='../content/label_mapping_dicts/label_mapping_dict__main_categories_F3inDE_noFGH__2023-03-17-0957.pkl',
            description_model=f'main class training using EVAL data. {current_loss_function} resnet 120 epochs'
        )