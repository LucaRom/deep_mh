import torch
#import torchvision
#import wandb
from dataset import estrie_rasterio_3_inputs, kenauk_rasterio_3_inputs
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np
import random

# Define all paths
# Paths (linux)
# Paths estrie (sen2_ete est le dossier de reference pour le filtre des fichiers selon la taille de ceux-ci)

# e_img_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/512/"
# e_mask_bin_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/512/mask_bin"
# e_mask_multi_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/512/mask_multiclass"
# e_lidar_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/512/mnt"

# e_img_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/256/"
# e_mask_bin_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/256/mask_bin"
# e_mask_multi_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/256/mask_multiclass"
# e_lidar_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/256/mnt"

# 256
# e_img_dir = "/mnt/SN750/00_Donnees_SSD/256/"
# e_mask_bin_dir = "/mnt/SN750/00_Donnees_SSD/256/mask_bin"
# e_mask_multi_dir = "/mnt/SN750/00_Donnees_SSD/256/mask_multiclass"
# e_lidar_dir = "/mnt/SN750/00_Donnees_SSD/256/mnt"

# Estrie 256 overlap
e_img_dir = "/mnt/SN750/00_Donnees_SSD/256_over50p/"
e_mask_bin_dir = "/mnt/SN750/00_Donnees_SSD/256_over50p/mask_bin"
e_mask_multi_dir = "/mnt/SN750/00_Donnees_SSD/256_over50p/mask_multiclass"
e_lidar_dir = "/mnt/SN750/00_Donnees_SSD/256_over50p/mnt"

# Kenauk full - linux
# k_img_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/256"
# k_mask_bin_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/256/mask_bin"
# k_mask_multi_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/256/mask_multiclass"

# Kenauk 2016 - linux
# k_img_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/256"
# k_mask_bin_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/256/mask_bin"
# k_mask_multi_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/256/mask_multiclass"

# # path kenauk (old)
# k_img_dir= "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/train"
# k_mask_dir = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/train"
# k_lidar_dir = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/train"

# k_val_img_dir = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/val"
# k_val_mask_dir = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/val"
# k_val_mnt_dir = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/val"

# # Path Kenauk Full (test)
# k_test_img = "/mnt/Data/00_Donnees/01_trainings/03_kenauk_test_full/sen2_print"
# k_test_mask = "/mnt/Data/00_Donnees/01_trainings/03_kenauk_test_full/mask_bin"
# k_test_lid = "/mnt/Data/00_Donnees/01_trainings/03_kenauk_test_full/lidar_mnt"

# Path Kenauk Full (test)
k_test_img = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/256" # TODO FIX THIS (paths in datasets...)
k_test_mask = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/256/mask_bin"
k_test_lid = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/256/lidar_mnt"

def get_datasets(
    train_region,
    test_region,
    classif_mode,
    batch_size,
    #train_transform,
    #val_transform,
    num_workers=1,
    pin_memory=True,
):

# Defining training paths by region
    if train_region == "kenauk_2016":
        img_train_dir = e_img_dir

        train_ds = kenauk_rasterio_3_inputs(
            train_dir=k_img_dir,
            classif_mode=classif_mode
            #transform=train_transform
            )


        # Creating train, val, test datasets
        if test_region == 'local_split':
            train_set_size = int(len(train_ds) * 0.80)
            valid_set_size = (len(train_ds) - train_set_size) // 2
            test_set_size =  len(train_ds) - (train_set_size + valid_set_size)
            train_ds, val_ds, test_ds = random_split(train_ds, [train_set_size, valid_set_size, test_set_size])

        elif test_region == 'kenauk_full':
            print("This is not available/logical for this dataset", train_region, test_region)

    elif train_region == "estrie":
        img_train_dir = e_img_dir 
        #mnt_train_dir = e_lidar_dir

        # Choosing mask paths by classif mode
        # if classif_mode == "bin":
        #     print("Using training paths from Estrie for a binary classification")
        #     #train_maskdir = e_mask_bin_dir

        #     # Initiate dataset
        #     train_ds = estrie_rasterio_3_inputs(
        #         train_dir=img_train_dir,
        #         classif_mode=classif_mode
        #         #transform=train_transform
        #     )

        # elif classif_mode == "multiclass":
        #     print("Using training paths from Estrie to train for a multi-class classification")
        #     train_maskdir = e_mask_multi_dir

            # # Initiate dataset
            # train_ds = estrie_rasterio(
            #     train_dir=img_train_dir,
            #     classif_mode=classif_mode
            #     #transform=train_transform
            # )

            # TODO make selectable datasets in options (ex. : stack, sen2, sen 2 + sen 1, sen2 + lidar, etc.)
            # Initiate dataset

        train_ds = estrie_rasterio_3_inputs(
            train_dir=img_train_dir,
            classif_mode=classif_mode
            #transform=train_transform
        )

        # Creating train, val, test datasets
        if test_region == 'local_split':
            # train_set_size = int(len(train_ds) * 0.80)
            # valid_set_size = (len(train_ds) - train_set_size) // 2
            # test_set_size =  len(train_ds) - (train_set_size + valid_set_size)
            #train_ds, val_ds, test_ds = random_split(train_ds, [train_set_size, valid_set_size, test_set_size])
            #train_temp, test_ds = train_test_split(train_ds, [train_set_size, valid_set_size, test_set_size])

            # Spliting Test dataset out and generating random train and val from rest of indices
            # Test for subset sampler #TODO better coding 
            print('WARNING WARNING WARNING ON THE SPLIT')
            indices = list(range(len(train_ds)))
            split_val = len(train_ds) - int(np.floor(0.2 * len(train_ds)))
            split_test = split_val + ((len(train_ds)- split_val ) // 2)

            indices_train_val = indices[:split_test]
            random.shuffle(indices_train_val)

            split_val_rd = len(indices_train_val) - int(np.floor(0.2 *len(indices_train_val)))

            train_idx, val_idx, test_idx = indices_train_val[:split_val_rd], indices_train_val[split_val_rd:], indices[split_test:]

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            # train_ds = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
            # val_ds = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=val_sampler)
            # test_ds = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=test_sampler)

        elif test_region == 'kenauk_full':
            # Initiate kenauk full dataset
            # Choosing mask paths by classif mode

            # if classif_mode == "bin":
            #     print("binary classification for estrie needs to be verified in utils.py")
            #     #print("Testing will be made on the full Kenauk dataset for a binary classification")
            #     #train_maskdir = e_mask_bin_dir

            # # test_kenauk_full_ds = KenaukDataset_stack2(
            # # image_dir=k_test_img,
            # # mask_dir=k_test_mask,
            # # mnt_dir=k_test_lid,
            # # #transform=train_transform
            # # )

            # elif classif_mode == "multiclass":
            #     print("Testing on Kenauk full dataset after training on Estrie dataset for a multiclass classification")
            #     train_maskdir = k_mask_multi_dir

#TODO classif_mode from datasets SHOULD TAKE CARE OF CHOOSING RIGHT MASK and thus if and elif might be removable

            test_kenauk_full_ds = kenauk_rasterio_3_inputs(
            train_dir=k_img_dir,
            classif_mode=classif_mode
            #transform=train_transform
            )

            train_set_size = int(len(train_ds) * 0.80)
            valid_set_size = (len(train_ds) - train_set_size)
            train_ds, val_ds = random_split(train_ds, [train_set_size, valid_set_size])
            test_ds = test_kenauk_full_ds


        elif test_region == 'kenauk_2016':

            test_kenauk_full_ds = kenauk_rasterio_3_inputs(
            train_dir=k_img_dir,
            classif_mode=classif_mode
            #transform=train_transform
            )

            train_set_size = int(len(train_ds) * 0.80)
            valid_set_size = (len(train_ds) - train_set_size)
            train_ds, val_ds = random_split(train_ds, [train_set_size, valid_set_size])
            test_ds = test_kenauk_full_ds

        else:
            print("Something is wrong with your Estrie paths")
    
    else:
        print("Something is wrong with your overall paths")


    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=True,
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )

    # test_loader = DataLoader(
    #     test_ds,
    #     batch_size=1,
    #     num_workers=0,
    #     pin_memory=False,
    #     shuffle=False,
    # )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        #shuffle=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        #shuffle=False,
        sampler=val_sampler
    )

    test_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        #shuffle=False,
        sampler=test_sampler
    )

    return train_loader, val_loader, test_loader

def get_datasets_inference(
    train_region,
    test_region,
    classif_mode,
    batch_size,
    #train_transform,
    #val_transform,
    num_workers=1,
    pin_memory=True,
):

# Defining training paths by region
    if train_region == "kenauk":
        img_train_dir = e_img_dir

        train_ds = kenauk_rasterio_3_inputs(
            train_dir=k_img_dir,
            classif_mode=classif_mode
            #transform=train_transform
            )


        # Creating train, val, test datasets
        if test_region == 'local_split':
            train_set_size = int(len(train_ds) * 0.80)
            valid_set_size = (len(train_ds) - train_set_size) // 2
            test_set_size =  len(train_ds) - (train_set_size + valid_set_size)
            train_ds, val_ds, test_ds = random_split(train_ds, [train_set_size, valid_set_size, test_set_size])

        elif test_region == 'kenauk_full':
            print("This is not available/logical for this dataset", train_region, test_region)

    elif train_region == "estrie":
        img_train_dir = e_img_dir 
        #mnt_train_dir = e_lidar_dir

        # Choosing mask paths by classif mode
        # if classif_mode == "bin":
        #     print("Using training paths from Estrie for a binary classification")
        #     #train_maskdir = e_mask_bin_dir

        #     # Initiate dataset
        #     train_ds = estrie_rasterio_3_inputs(
        #         train_dir=img_train_dir,
        #         classif_mode=classif_mode
        #         #transform=train_transform
        #     )

        # elif classif_mode == "multiclass":
        #     print("Using training paths from Estrie to train for a multi-class classification")
        #     train_maskdir = e_mask_multi_dir

            # # Initiate dataset
            # train_ds = estrie_rasterio(
            #     train_dir=img_train_dir,
            #     classif_mode=classif_mode
            #     #transform=train_transform
            # )

            # TODO make selectable datasets in options (ex. : stack, sen2, sen 2 + sen 1, sen2 + lidar, etc.)
            # Initiate dataset

        train_ds = estrie_rasterio_3_inputs(
            train_dir=img_train_dir,
            classif_mode=classif_mode
            #transform=train_transform
        )

        # Creating train, val, test datasets
        if test_region == 'local_split':
            print("my mannnnnnnnnnnnnnnn")
            # train_set_size = int(len(train_ds) * 0.80)
            # valid_set_size = (len(train_ds) - train_set_size) // 2
            # test_set_size =  len(train_ds) - (train_set_size + valid_set_size)
            # #train_ds, val_ds, test_ds = random_split(train_ds, [train_set_size, valid_set_size, test_set_size])
            # #train_temp, test_ds = train_test_split(train_ds, [train_set_size, valid_set_size, test_set_size])

            # # Test for subset sampler
            # print('WARNING WARNING WARNING ON THE SPLIT')
            # indices = list(range(len(train_ds)))
            # split_val = len(train_ds) - int(np.floor(0.2 * len(train_ds)))
            # split_test = split_val + ((len(train_ds)- split_val ) // 2)

            # train_idx, val_idx, test_idx = indices[:split_val], indices[split_val:split_test], indices[split_test:]

            # train_sampler = SubsetRandomSampler(train_idx)
            # val_sampler = SubsetRandomSampler(val_idx)
            # test_sampler = SubsetRandomSampler(test_idx)

            # train_ds = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
            # val_ds = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=val_sampler)
            # test_ds = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=test_sampler)

        elif test_region == 'kenauk_full':
            # Initiate kenauk full dataset
            # Choosing mask paths by classif mode
            print("kenauk full test region checkpoint")

            # if classif_mode == "bin":
            #     print("binary classification for estrie needs to be verified in utils.py")
            #     #print("Testing will be made on the full Kenauk dataset for a binary classification")
            #     #train_maskdir = e_mask_bin_dir

            # # test_kenauk_full_ds = KenaukDataset_stack2(
            # # image_dir=k_test_img,
            # # mask_dir=k_test_mask,
            # # mnt_dir=k_test_lid,
            # # #transform=train_transform
            # # )

            test_kenauk_full_ds = kenauk_rasterio_3_inputs(
            train_dir=k_test_img,
            classif_mode=classif_mode
            #transform=train_transform
            )

            train_set_size = int(len(train_ds) * 0.80)
            valid_set_size = (len(train_ds) - train_set_size)
            train_ds, val_ds = random_split(train_ds, [train_set_size, valid_set_size])
            test_ds = test_kenauk_full_ds

            # elif classif_mode == "multiclass":
            #     print("Testing on Kenauk full dataset after training on Estrie dataset for a multiclass classification")
            #     train_maskdir = k_mask_multi_dir

#TODO classif_mode from datasets SHOULD TAKE CARE OF CHOOSING RIGHT MASK and thus if and elif might be removable

            # test_kenauk_full_ds = kenauk_rasterio_3_inputs(
            # train_dir=k_img_dir,
            # classif_mode=classif_mode
            # #transform=train_transform
            # )

            # train_set_size = int(len(train_ds) * 0.80)
            # valid_set_size = (len(train_ds) - train_set_size)
            # train_ds, val_ds = random_split(train_ds, [train_set_size, valid_set_size])
            # test_ds = test_kenauk_full_ds


        elif test_region == 'kenauk_2016':

            test_kenauk_2016_ds = kenauk_rasterio_3_inputs(
            train_dir=k_img_dir,
            classif_mode=classif_mode
            #transform=train_transform
            )

            train_set_size = int(len(train_ds) * 0.80)
            valid_set_size = (len(train_ds) - train_set_size)
            train_ds, val_ds = random_split(train_ds, [train_set_size, valid_set_size])
            test_ds = test_kenauk_2016_ds
            
        else:
            print("Something is wrong with your Estrie paths")
    
    else:
        print("Something is wrong with your overall paths")


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# def get_loaders(
#     train_dir,
#     train_maskdir,
#     train_mnt,
#     val_dir,
#     val_maskdir,
#     val_mnt,
#     test_dir,
#     test_maskdir,
#     test_mnt,
#     batch_size,
#     #train_transform,
#     #val_transform,
#     num_workers=1,
#     pin_memory=True,
# ):

#     train_ds = KenaukDataset_stack(
#         image_dir=train_dir,
#         mask_dir=train_maskdir,
#         mnt_dir=train_mnt,
#         #transform=train_transform
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=True,
#     )

#     val_ds = KenaukDataset_stack(
#         image_dir=val_dir,
#         mask_dir=val_maskdir,
#         mnt_dir=val_mnt,
#         #transform=val_transform,
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=False,
#     )

#     # test_ds = KenaukDataset_rasterio(
#     #     image_dir=train_dir,
#     #     mask_dir=train_maskdir,
#     #     mnt_dir=train_mnt,
#     # )

#     # test_loader = DataLoader(
#     #     test_ds,
#     #     batch_size=1,
#     #     num_workers=0,
#     #     pin_memory=False,
#     #     shuffle=False,
#     # )

#     # Dataset Estrie
#     # Estrie 2 encodeurs binaire

#     # train_estrie_ds = estrie_rasterio(
#     #     image_dir=train_dir,
#     #     mask_dir=train_maskdir,
#     #     mnt_dir=train_mnt,
#     #     #transform=train_transform
#     # )

#     # Estrie 1 encodeur binaire
#     # train_estrie_ds = estrie_stack(
#     #     image_dir=train_dir,
#     #     mask_dir=train_maskdir,
#     #     mnt_dir=train_mnt,
#     #     #transform=train_transform
#     # )

#     # Estrie test datasets 
#     estrie_ds = estrie_stack2(
#         image_dir=test_dir,
#         mask_dir=test_maskdir,
#         mnt_dir=test_mnt,
#         #transform=train_transform
#     )

#     test_kenauk_full_ds = KenaukDataset_stack2(
#         image_dir=test_dir,
#         mask_dir=test_maskdir,
#         mnt_dir=test_mnt,
#         #transform=train_transform
#     )

#     # TODO Put a conditional value for dataset if a specific path needs to be 
#     # given for validation set aswell
#     # Random split
#     # train_set_size = int(len(train_estrie_ds) * 0.90)
#     # valid_set_size = (len(train_estrie_ds) - train_set_size) // 2
#     # test_set_size =  len(train_estrie_ds) - (train_set_size + valid_set_size)
#     # train_ds, val_ds, test_ds = random_split(train_estrie_ds, [train_set_size, valid_set_size, test_set_size])
    
#     train_set_size = int(len(estrie_ds) * 0.90)
#     valid_set_size = (len(estrie_ds) - train_set_size) // 2
#     test_set_size =  len(estrie_ds) - (train_set_size + valid_set_size)
#     train_ds, val_ds, test_ds = random_split(estrie_ds, [train_set_size, valid_set_size, test_set_size])
   
#     test_loader = DataLoader(
#         test_kenauk_full_ds,
#         batch_size=1,
#         num_workers=0,
#         pin_memory=False,
#         shuffle=False,
#     )


#     # print(train_set_size, valid_set_size, test_set_size)

#     # train_loader = DataLoader(
#     #     train_ds,
#     #     batch_size=batch_size,
#     #     num_workers=num_workers,
#     #     pin_memory=pin_memory,
#     #     shuffle=True,
#     # )

#     # val_loader = DataLoader(
#     #     val_ds,
#     #     batch_size=batch_size,
#     #     num_workers=num_workers,
#     #     pin_memory=pin_memory,
#     #     shuffle=False,
#     # )

#     # # TODO Change input of getloaders() for specific datasets
#     # test_loader = DataLoader(
#     #     test_kenauk_full_ds,
#     #     batch_size=1,
#     #     num_workers=0,
#     #     pin_memory=False,
#     #     shuffle=False,
#     # )

#     return train_loader, val_loader, test_loader

# # def get_loaders_estrie(
# #     train_dir,
# #     train_maskdir,
# #     train_mnt,
# #     val_dir,
# #     val_maskdir,
# #     val_mnt,
# #     test_dir,
# #     test_maskdir,
# #     test_mnt,
# #     batch_size,
# #     #train_transform,
# #     #val_transform,
# #     num_workers=1,
# #     pin_memory=True,
# # ):

# #     # Dataset Estrie
# #     # Estrie 2 encodeurs binaire

# #     # train_estrie_ds = estrie_rasterio(
# #     #     image_dir=train_dir,
# #     #     mask_dir=train_maskdir,
# #     #     mnt_dir=train_mnt,
# #     #     #transform=train_transform
# #     # )

# #     #Estrie 1 encodeur binaire
# #     train_estrie_ds = estrie_stack2(
# #         image_dir=train_dir,
# #         mask_dir=train_maskdir,
# #         mnt_dir=train_mnt,
# #         #transform=train_transform
# #     )

# #     test_kenauk_full_ds = KenaukDataset_stack2(
# #         image_dir=test_dir,
# #         mask_dir=test_maskdir,
# #         mnt_dir=test_mnt,
# #         #transform=train_transform
# #     )

# #     # test_estrie_ds = estrie_stack2(
# #     #     image_dir=train_dir,
# #     #     mask_dir=train_maskdir,
# #     #     mnt_dir=train_mnt,
# #     #     #transform=train_transform
# #     # )


# #     # TODO Put a conditional value for dataset if a specific path needs to be 
# #     # given for validation set aswell
# #     # Random split
# #     train_set_size = int(len(train_estrie_ds) * 0.90)
# #     valid_set_size = (len(train_estrie_ds) - train_set_size) // 2
# #     test_set_size =  len(train_estrie_ds) - (train_set_size + valid_set_size)
# #     train_ds, val_ds, test_ds = random_split(train_estrie_ds, [train_set_size, valid_set_size, test_set_size])
    
# #     print(train_set_size, valid_set_size, test_set_size)

# #     train_loader = DataLoader(
# #         train_ds,
# #         batch_size=batch_size,
# #         num_workers=num_workers,
# #         pin_memory=pin_memory,
# #         shuffle=True,
# #     )

# #     val_loader = DataLoader(
# #         val_ds,
# #         batch_size=batch_size,
# #         num_workers=num_workers,
# #         pin_memory=pin_memory,
# #         shuffle=False,
# #     )

# #     # TODO Change input of getloaders() for specific datasets
# #     # test_loader = DataLoader(
# #     #     test_kenauk_full_ds,
# #     #     batch_size=1,
# #     #     num_workers=0,
# #     #     pin_memory=False,
# #     #     shuffle=False,
# #     # )

# #     test_loader = DataLoader(
# #         test_ds,
# #         batch_size=1,
# #         num_workers=0,
# #         pin_memory=False,
# #         shuffle=False,
# #     )

# #     return train_loader, val_loader, test_loader

# def get_loaders_estrie(
#     test_region,
#     train_dir,
#     train_maskdir,
#     train_mnt,
#     test_dir,
#     test_maskdir,
#     test_mnt,
#     batch_size,
#     #train_transform,
#     #val_transform,
#     num_workers=1,
#     pin_memory=True,
# ):

#     #Estrie 1 encodeur binaire
#     train_estrie_ds = estrie_stack2(
#         image_dir=train_dir,
#         mask_dir=train_maskdir,
#         mnt_dir=train_mnt,
#         #transform=train_transform
#     )

#     test_kenauk_full_ds = KenaukDataset_stack2(
#         image_dir=test_dir,
#         mask_dir=test_maskdir,
#         mnt_dir=test_mnt,
#         #transform=train_transform
#     )

#     # TODO Put a conditional value for dataset if a specific path needs to be 
#     # given for validation set aswell
#     # Random split
#     if test_region == 'split':
#         train_set_size = int(len(train_estrie_ds) * 0.90)
#         valid_set_size = (len(train_estrie_ds) - train_set_size) // 2
#         test_set_size =  len(train_estrie_ds) - (train_set_size + valid_set_size)
#         train_ds, val_ds, test_ds = random_split(train_estrie_ds, [train_set_size, valid_set_size, test_set_size])
#     elif test_region == 'kenauk':
#         train_set_size = int(len(train_estrie_ds) * 0.90)
#         valid_set_size = (len(train_estrie_ds) - train_set_size)
#         train_ds, val_ds = random_split(train_estrie_ds, [train_set_size, valid_set_size])
#         test_ds = test_kenauk_full_ds

#     print(train_set_size, valid_set_size, test_set_size)

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=True,
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=False,
#     )

#     test_loader = DataLoader(
#         test_ds,
#         batch_size=1,
#         num_workers=0,
#         pin_memory=False,
#         shuffle=False,
#     )

#     return train_loader, val_loader, test_loader

# def get_loaders_kenauk_multi_enco(
#     train_dir,
#     train_maskdir,
#     train_mnt,
#     val_dir,
#     val_maskdir,
#     val_mnt,
#     test_dir,
#     test_maskdir,
#     test_mnt,
#     batch_size,
#     #train_transform,
#     #val_transform,
#     num_workers=1,
#     pin_memory=True,
# ):

#     # Dataset Kenauk
#     # 2 encodeur
#     train_kenauk_ds = KenaukDataset_rasterio(
#         image_dir=train_dir,
#         mask_dir=train_maskdir,
#         mnt_dir=train_mnt,
#         #transform=train_transform
#     )

#     val_kenauk_ds = KenaukDataset_rasterio(
#         image_dir=val_dir,
#         mask_dir=val_maskdir,
#         mnt_dir=val_mnt,
#         #transform=train_transform
#     )

#     test_kenauk_full_ds = KenaukDataset_rasterio(
#         image_dir=test_dir,
#         mask_dir=test_maskdir,
#         mnt_dir=test_mnt,
#         #transform=train_transform
#     )

#     train_loader = DataLoader(
#         train_kenauk_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=True,
#     )

#     val_loader = DataLoader(
#         val_kenauk_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=False,
#     )

#     #TODO Change input of getloaders() for specific datasets
#     test_loader = DataLoader(
#         test_kenauk_full_ds,
#         batch_size=1,
#         num_workers=0,
#         pin_memory=False,
#         shuffle=False,
#     )

#     return train_loader, val_loader, test_loader

if __name__ == "__main__":
    #cli_main()

    PIN_MEMORY = True
    NUM_WORKERS = 0
    BATCH_SIZE = 2
    num_epochs = 50
    optim_main = "Ad"  # 'Ad' ou 'sg'
    lr_main = 0.0001
    num_layers_main = 5
    input_channel_main = 13
    input_channel_lidar = 1

    # Paths (windows)
    # # Paths estrie
    # e_img_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/sen2"
    # e_mask_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_bin"
    # #e_mask_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_multi" # Multiclass # TODO CHANGE DATASET TO CHANGE NAMES :\
    # e_lidar_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/lidar_mnt"

    # # path kenauk
    # k_img_dir= "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/train"
    # k_mask_dir = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/train"
    # k_lidar_dir = "D:/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/train"

    # VAL_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/val"
    # VAL_MASK_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/val"
    # VAL_MNT_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/val"

    # # Path Kenauk Full (test)
    # k_test_img = "D:/00_Donnees/01_trainings/03_kenauk_test_full/sen2_print"
    # k_test_mask = "D:/00_Donnees/01_trainings/03_kenauk_test_full/mask_bin"
    # k_test_lid = "D:/00_Donnees/01_trainings/03_kenauk_test_full/lidar_mnt"

    # Paths (linux)
    # Paths estrie
    e_img_dir = "/mnt/Data/00_Donnees/01_trainings/02_mh_double_stack/estrie/sen2"
    e_mask_dir = "/mnt/Data/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_bin"
    #e_mask_dir = "/mnt/Data/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_multi" # Multiclass # TODO CHANGE DATASET TO CHANGE NAMES :\
    e_lidar_dir = "/mnt/Data/00_Donnees/01_trainings/02_mh_double_stack/estrie/lidar_mnt"

    # path kenauk
    k_img_dir= "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/train"
    k_mask_dir = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/train"
    k_lidar_dir = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/train"

    VAL_IMG_DIR = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/val"
    VAL_MASK_DIR = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/val"
    VAL_MNT_DIR = "/mnt/Data/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/val"

    # Path Kenauk Full (test)
    k_test_img = "/mnt/Data/00_Donnees/01_trainings/03_kenauk_test_full/sen2_print"
    k_test_mask = "/mnt/Data/00_Donnees/01_trainings/03_kenauk_test_full/mask_bin"
    k_test_lid = "/mnt/Data/00_Donnees/01_trainings/03_kenauk_test_full/lidar_mnt"

    # train_loader, val_loader, test_loader = get_loaders(
    # e_img_dir,
    # e_mask_dir,
    # e_lidar_dir,
    # BATCH_SIZE,
    # # train_transform,
    # # val_transforms,
    # NUM_WORKERS,
    # PIN_MEMORY,
    # )

    # Training Kenauk, Test Estrie
    train_loader, val_loader, test_loader = get_loaders(
    k_img_dir,
    k_mask_dir,
    k_lidar_dir,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    VAL_MNT_DIR,
    e_img_dir,
    e_mask_dir,
    e_lidar_dir,
    BATCH_SIZE,
    # train_transform,
    # val_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
    )

    e_train_loader, e_val_loader, e_test_loader = get_loaders_estrie(
    e_img_dir,
    e_mask_dir,
    e_lidar_dir,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    VAL_MNT_DIR,
    e_img_dir,
    e_mask_dir,
    e_lidar_dir,
    BATCH_SIZE,
    # train_transform,
    # val_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
    )
    
    

    # # Sanity check
    # inputs, labels = next(iter(train_loader))
    # e_inputs, e_labels, img_path = next(iter(e_train_loader))

    # unique, counts = np.unique(e_labels, return_counts=True)

    # #print('kenauk : \n', inputs, '\n estrie : \n', e_inputs)
    # #print('kenauk labels : \n', labels, '\n estrie labels: \n', e_labels)

    # assert len(unique) == 2

    import numpy as np

    print('Asserting ...')
    for e_inputs, e_labels, img_path in e_train_loader:
        unique, counts = np.unique(e_labels, return_counts=True)

        # print(unique, counts)
        # print(img_path)
        
        ratio = counts[0] // counts[1]

        #assert len(unique) == 2

        print(ratio)

    print('If you see this, all is good')