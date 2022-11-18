import torch
import torchvision
from dataset import estrie_rasterio_3_inputs, kenauk_rasterio_3_inputs
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np
import random
import rasterio
import img_paths
import matplotlib.pyplot as plt
import tifffile as tiff
from rasterio import windows
from rasterio.windows import Window
from itertools import product
from tqdm import tqdm

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


### colected paths to sort
# Estrie sentinel 2 image : 
path_sen2_ete = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/ete/S2_estrie_3m_ete_septembre2020.tif"
path_sen2_prin = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/print/S2_estrie_3m_printemps_mai2020.tif"

path_sen1_ete = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1/ete/S1_estrie_3m_ete_septembre2020.tif"
path_sen1_prin = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1/print/S1_estrie_3m_printemps_mai2020.tif"

path_mhc = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/mhc_estrie_3m.tif"
path_slopes = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/pente_estrie_3m.tif"
path_tpi = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tpi_estrie_3m.tif"
path_tri = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tri_estrie_3m.tif"
path_twi  = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/twi_estrie_3m.tif"


# image_ete = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/ete/S2_estrie_3m_ete_septembre2020.tif"
# image_prin = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/print/S2_estrie_3m_printemps_mai2020.tif"

# k_image_ete = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/ete/s2_kenauk2016_3m_ete.tif"
# k_image_prin = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/print/s2_kenauk2016_3m_print.tif"

#Kenauk
#k_sen2_ete = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/ete/s2_kenauk_3m_ete_aout2020.tif"
#k_sen2_prin = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/print/S2_de_kenauk_3m_printemps_mai2020.tif"

#k_sen1_ete = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/ete/s1_kenauk_3m_ete_aout2020.tif"
#k_sen1_prin = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/print/S1_kenauk_3m_printemps_mai2020.tif"

k_mhc = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mhc_kenauk_3m.tif"
k_slopes = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/pentes_kenauk_3m.tif"
k_tpi = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tpi_kenauk_3m.tif"
k_tri = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif"
k_twi  = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/twi_kenauk_3m.tif"


def get_project_labels():
    dict_labels = {
                    0: 'EP',
                    1: 'MS',
                    2: 'PH',
                    3: 'ME',
                    4: 'BG',
                    5: 'FN',
                    6: 'TB',
                    7: 'NH',
                    8: 'SH'
                  }
    return dict_labels

def get_tiled_datasets_estrie(
    input_format,
    classif_mode,
    batch_size,
    #train_transform,
    #val_transform,
    num_workers=1,
    pin_memory=True,
):

    # Defining training paths by region
    if input_format == 'estrie_over0p':
        paths_list = img_paths.estrie_256over0p_paths_lst
    elif input_format == 'estrie_over50p':
        paths_list = img_paths.estrie_256over50p_paths_lst

    print("Using following path as root for training dataset : ")
    print(paths_list[0])

    # Define train dir from paths_list
    img_train_dir = paths_list[0]

    # Loadings means and stdevs
    sen2_e_means = np.load('stats/estrie/estrie_sen2_ete_means.npy')
    sen2_p_means = np.load('stats/estrie/estrie_sen2_pri_means.npy')
    sen1_e_means = np.load('stats/estrie/estrie_sen1_ete_means.npy')
    sen1_p_means = np.load('stats/estrie/estrie_sen1_pri_means.npy')
    lidar_means  = np.load('stats/estrie/estrie_lidar_means.npy')
    
    # Combine mean lists for sen1 and sen2
    sen2_means = [*sen2_e_means, *sen2_p_means]
    sen1_means = [*sen1_e_means, *sen1_p_means]

    sen2_e_stdevs = np.load('stats/estrie/estrie_sen2_ete_stds.npy')
    sen2_p_stdevs = np.load('stats/estrie/estrie_sen2_pri_stds.npy')
    sen1_e_stdevs = np.load('stats/estrie/estrie_sen1_ete_stds.npy')
    sen1_p_stdevs = np.load('stats/estrie/estrie_sen1_pri_stds.npy')
    lidar_stdevs  = np.load('stats/estrie/estrie_lidar_stds.npy')

    # Combine stdev lists for sen1 and sen2
    sen2_stdevs = [*sen2_e_stdevs, *sen2_p_stdevs]
    sen1_stdevs = [*sen1_e_stdevs, *sen1_p_stdevs]

    mean_lst2 = [sen2_e_means, sen2_p_means, sen1_e_means, sen1_p_means, lidar_means]
    stdev_lst2 = [sen2_e_stdevs, sen2_p_stdevs, sen1_e_stdevs, sen1_p_stdevs, lidar_stdevs]

    mean_lst = [sen2_means, sen1_means, lidar_means]
    stdev_lst = [sen2_stdevs, sen1_stdevs, lidar_stdevs]

    train_ds = estrie_rasterio_3_inputs(    
        train_dir=img_train_dir,
        classif_mode=classif_mode,
        #transform=train_transform
        mean_lst=mean_lst,
        stdev_lst=stdev_lst
    )

    # Creating train, val, test datasets
    # Spliting Test dataset out and generating random train and val from rest of indices
    # #TODO better coding
    print()
    print('********** IMPORTANT **********')
    print("Test dataset is split alone before train and val to avoid model seeing parts of the test tiles from "
          "the overlapping when selected completely randomly")
    print('*******************************')

    # Defining indices
    #indices = list(range(len(train_ds)))

    # Defining split index numbers
    '''
        Details : For a 80-10-10 split, we first calculate the split index number at 80% (100 - ~20%).
                  split_test index number is then optained by adding ~10% value from split_val
                  indices for train and val are sliced with split_test index number 
    '''
    # split_val = len(train_ds) - int(np.floor(0.2 * len(train_ds)))
    # split_test = split_val + ((len(train_ds)- split_val ) // 2)
    # indices_train_val = indices[:split_test]
    # random.shuffle(indices_train_val) # Shuffling train and val only
    # split_val_rd = len(indices_train_val) - int(np.floor(0.2 *len(indices_train_val)))

    trainval_idx_lst = np.load('results/estrie_trainval_idx_list.npy')
    test_idx_lst = np.load('results/estrie_test_idx_list.npy')

    shuffled_trainval = np.random.permutation(trainval_idx_lst)
    val_size = round(len(trainval_idx_lst)*0.1) #10%

    val_idx = shuffled_trainval[:val_size]
    train_idx = [x for x in shuffled_trainval if x not in val_idx]
    test_idx = test_idx_lst

    #print()

    # Creating idx for each set from split number
    #train_idx, val_idx, test_idx = indices_train_val[:split_val_rd], indices_train_val[split_val_rd:], indices[split_test:]
    #train_idx, val_idx, test_idx = 

    # Creating sampler callable in dataloaders
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Returning data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, 
                              sampler=train_sampler)

    val_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, pin_memory=pin_memory, 
                            sampler=val_sampler)

    test_loader = DataLoader(train_ds, batch_size=1, num_workers=0, pin_memory=False, sampler=test_sampler)

    return train_loader, val_loader, test_loader


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
        print("I am ERROR")
    elif train_region == "estrie":
        img_train_dir = e_img_dir 

        train_ds = estrie_rasterio_3_inputs(
            train_dir=img_train_dir,
            classif_mode=classif_mode
            #transform=train_transform
        )

        # Creating train, val, test datasets
        if test_region == 'local_split':
            # Spliting Test dataset out and generating random train and val from rest of indices
            # Test for subset sampler #TODO better coding
            print()
            print('********** IMPORTANT **********')
            print('Test dataset is split alone before train and val to avoid model seeing parts of the test tiles from \
                   the overlapping when selected completely randomly')
            print('*******************************')

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
            print('I am error')

        elif test_region == 'kenauk_2016':
            print('I am error')

        else:
            print("Something is wrong with your Estrie paths")
    
    else:
        print("Something is wrong with your overall paths")

    # Returning data loaders
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


def get_datasets_quick_mode(
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
            print("Quick mode not implemented yet for kenauk_2016")
            # train_set_size = int(len(train_ds) * 0.80)
            # valid_set_size = (len(train_ds) - train_set_size) // 2
            # test_set_size =  len(train_ds) - (train_set_size + valid_set_size)
            # train_ds, val_ds, test_ds = random_split(train_ds, [train_set_size, valid_set_size, test_set_size])

        elif test_region == 'kenauk_full':
            print("This is not available/logical for this dataset", train_region, test_region)

    elif train_region == "estrie":
        img_train_dir = e_img_dir 
        #mnt_train_dir = e_lidar_dir

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
            number_of_idx_needed = len(indices) // 10

            quick_mode_indices = random.choices(indices, k=number_of_idx_needed)

            split_val = len(quick_mode_indices) - int(np.floor(0.2 * len(quick_mode_indices)))
            split_test = split_val + ((len(quick_mode_indices)- split_val ) // 2)

            indices_train_val = indices[:split_test]
            indices_train_val = quick_mode_indices[:split_test]
            random.shuffle(indices_train_val)

            split_val_rd = len(indices_train_val) - int(np.floor(0.2 *len(indices_train_val)))

            # splitting everythin
            # original_train_idx = indices_train_val[:split_val_rd]
            # number_of_idx_needed = len(original_train_idx ) // 10
            # modified_train_idx_for_quick_mode = random.choices(original_train_idx , k=number_of_idx_needed)
            train_idx, val_idx, test_idx = indices_train_val[:split_val_rd], indices_train_val[split_val_rd:], quick_mode_indices[split_test:]

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


def iter_windows(src_ds, stepsize, width, height, strict_shape=True, boundless=False):
    # offsets creates tuples for col_off, row_off
    offsets = product(range(0, src_ds.meta['width'], stepsize), range(0, src_ds.meta['height'], stepsize))
    big_window = windows.Window(col_off=0, row_off=0, width=src_ds.meta['width'], height=src_ds.meta['height'])

    # Creates windows from offsets as start pixels and uses specified window size
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

        if boundless:
            window = window
        else:
            window = window.intersection(big_window)

        # Strict shape limits output windows with only width x height shape
        if strict_shape:
            if window.width < width or window.height < height:
                pass
            else:
                yield window

if __name__ == "__main__":
    
    # Create mask and zones stack
    def create_stack_mask_zones(mask_path, zones_path):
        mask_im  = np.array(tiff.imread(mask_path), dtype=np.float32)
        zones_im = np.array(tiff.imread(zones_path), dtype=np.float32)

        stacked_img = np.dstack((mask_im, zones_im))
        stacked_img = stacked_img.transpose(2,0,1)

        with rasterio.open(mask_path) as ds:
            profile = ds.profile
            profile['count'] = 2  # 2 bands
            with rasterio.open('./results/stack_mask_zones.tif', "w", **profile) as out_ds:
                out_ds.write(stacked_img)

    mask_path = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/mask/estrie_mask_multiclass_3m_9c.tif'
    zones_path = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/zones/zones_limits.tif'

    #create_stack_mask_zones(mask_path, zones_path)

    # Generate train/val and test indexes from zones and mask
    def generate_indexes(img_path, wd_size, step_size):
        with rasterio.open(img_path) as ds:
            windows_num = len(list(iter_windows(ds, step_size, wd_size, wd_size)))
            #indices_ori = list(range(windows_num))
            test_indices = []
            train_indices = []
            full_nh_tiles_idx = []
            for idx, a_window in tqdm(enumerate(iter_windows(ds, step_size, wd_size, wd_size)), total=windows_num, desc='Windows'):
                    mask = ds.read(1, window=a_window)
                    zones = ds.read(2, window=a_window)

                    nh_pixels = np.count_nonzero(mask == 7)
                    test_pixels = np.count_nonzero(zones == 1)

                    tile_pixels_num = wd_size * wd_size

                    if test_pixels == tile_pixels_num:
                        test_indices.append(idx)
                    elif nh_pixels == tile_pixels_num:
                        full_nh_tiles_idx.append(idx)
                    else:
                        train_indices.append(idx)
            
            # print('Original indices len :', len(indices_ori))
            # print('Train val idx len:', len(train_indices))
            # print('Test idx len :', len(test_indices))
            # print('Full NH idx len :', len(full_nh_tiles_idx))

            np.save('results/estrie_trainval_idx_list', train_indices)
            np.save('results/estrie_test_idx_list', test_indices)


    img_path = 'results/stack_mask_zones.tif'

    #generate_indexes(img_path, wd_size=256, stepsize=128)


    ############################
    # Sanity check draft below #
    ############################

    # train_region = "estrie_over50p"
    # input_format = train_region
    # test_region = "local_split"

    # num_epochs = 10
    # classif_mode = "bin"
    # BATCH_SIZE = 6

    # PIN_MEMORY = True
    # NUM_WORKERS = 4
    # BATCH_SIZE = 6


    # train_loader, val_loader, test_loader = get_tiled_datasets_estrie(
    #                                             input_format,
    #                                             classif_mode,
    #                                             BATCH_SIZE,
    #                                             #train_transform,
    #                                             #val_transform,
    #                                             num_workers=4,
    #                                             pin_memory=True,
    #                                             )

    # print()

    # images, lidar, mask, radar, img_path = next(iter(test_loader)) # load directly from the dataset, not from the dataloader

    # # Images sentinel2 été
    # img = images[0].squeeze()

    # red   = img[[3],:,:][0].numpy()
    # green = img[[2],:,:][0].numpy()
    # blue  = img[[1],:,:][0].numpy()

    # red_2   = (red - min(red[0])) / (max(red[0]) - min(red[0]))
    # green_2 = (green - min(green[0])) / (max(green[0]) - min(green[0]))
    # blue_2  = (blue - min(blue[0])) / (max(blue[0]) - min(blue[0]))

    # stack_sen2_ete = np.dstack((red_2, green_2, blue_2))

    # # Images sentinel2 printemps


    # # Image lidar
    # im_lidar = lidar[0]

    # # Images sentinel1

    # # Image masque
    # mask = mask[0]


    # for im in im_lidar:
    #     plt.imshow(im)
    #     plt.show()
    #     plt.close()

    # # Generating figures
    # fig = plt.figure(figsize=(15, 5))
    # subfig = fig.subfigures(nrows=1, ncols=1)
    # axes = subfig.subplots(nrows=1, ncols=3,sharey=True)

    # cmap = plt.get_cmap('tab10', self.num_classes)

    # # Generating images in axes 
    # im1 = axes[0].imshow(np.transpose(ori_input, (1,2,0))*3)
    # im2 = axes[1].imshow(predict_sig, cmap=cmap,vmin = -0.5, vmax = self.num_classes - 0.5)
    # im3 = axes[2].imshow(ori_target,cmap=cmap,vmin = -0.5, vmax = self.num_classes - 0.5)

    # # Adding colorbar to the right
    # #TODO make ax.set_yticklabels automatic with getproject_labels() ?
    # cbar = subfig.colorbar(im2, shrink=0.7, ax=axes, ticks=np.arange(0,self.num_classes))
    # cbar.ax.set_yticklabels(['0 (EP)','1 (MS)','2 (PH)','3 (ME)','4 (BG)','5 (FN)','6 (TB)', '7 (NH)', '8 (SH)']) # Change colorbar labels
    # cbar.ax.invert_yaxis() # Flip colorbar 

    # # Set axes names
    # axes[0].set_title('Sen2 Input')
    # axes[1].set_title('Predicted')
    # axes[2].set_title('Target')