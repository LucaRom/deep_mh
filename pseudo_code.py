import numpy as np
import torchmetrics
import os
import rasterio
import tifffile as tiff
import time

from itertools import product
from rasterio import windows
#from rasterio.windows import Window
from torch.utils.data import Dataset




class estrie_dataset_unfold(Dataset):
    """ 
        Placeholder
    """
    def __init__(self, paths_lst, transform=None):
        self.sen2_paths = paths_lst[0:2]
        self.sen1_paths = paths_lst[2:4]
        self.lidar_paths = paths_lst[4:]
        self.mask_path = paths_lst[:]

        #src_

        self.windows = iter_windows(ref_img, 256, 256)

        self.transform = transform 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      

        return img_opt, img_lidar, mask, img_rad, sen2_ete_path

class estrie_dataset_rasterio(Dataset):
    """ 
        Placeholder
    """
    def __init__(self, paths_lst, transform=None):
        self.sen2_paths = paths_lst[0:3]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      

        return img_opt, img_lidar, mask, img_rad, sen2_ete_path



# Faire les dataloadoers

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



            test_kenauk_full_ds = kenauk_rasterio_3_inputs(
            train_dir=k_img_dir,
            classif_mode=classif_mode
            #transform=train_transform
            )

            train_set_size = int(len(train_ds) * 0.80)
            valid_set_size = (len(train_ds) - train_set_size)
            train_ds, val_ds = random_split(train_ds, [train_set_size, valid_set_size])
            test_ds = test_kenauk_full_ds

if __name__ == "__main__":

    # pour datasets input avec image full raw

    # Definir les paths de chaque image
    path_to_full_sen2_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s2_kenauk_3m_ete_HMe_STD.tif' 
    path_to_full_sen2_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s2_kenauk_3m_print_HMe_STD.tif' 
    path_to_full_sen1_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s1_kenauk_3m_ete_STD.tif' 
    path_to_full_sen1_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s1_kenauk_3m_print_STD.tif' 
    path_to_full_mhc        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mhc_kenauk_3m.tif' 
    path_to_full_slopes     = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/pentes_kenauk_3m.tif' 
    path_to_full_tpi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tpi_kenauk_3m.tif' 
    path_to_full_tri        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif' 
    path_to_full_twi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/twi_kenauk_3m.tif' 

    # Calculer les statistiques generales 

        # Moyenne
        # ecart type
        # Autres?


    # Definir le datasets

    # paths_list = [path_to_full_sen2_ete, path_to_full_sen2_print, path_to_full_sen1_ete, path_to_full_sen1_print, 
    #             path_to_full_mhc, path_to_full_slopes, path_to_full_tpi, path_to_full_tri, path_to_full_twi]

    # paths_list = [path_to_full_sen2_ete, path_to_full_sen2_print]
    
    # print(paths_list)
    # print(paths_list[0:4])
    # print()

    # Ouvrir les fichiers et faire les stacks
     
    # Load images
    # src_sen2_ete = rasterio.open(path_to_full_sen2_ete)
    # src_sen2_pri = rasterio.open(path_to_full_sen2_print)
    # src_sen1_ete = rasterio.open(path_to_full_sen1_ete)
    # src_sen1_pri = rasterio.open(path_to_full_sen1_print)
    # src_mhc = rasterio.open(path_to_full_mhc)
    # src_slo = rasterio.open(path_to_full_slopes)
    # src_tpi = rasterio.open(path_to_full_tpi)
    # src_tri = rasterio.open(path_to_full_tri)
    # src_twi = rasterio.open(path_to_full_twi)

    # def create_rasterio_arrays(paths_list):
    #     array_list = []
    #     for path in paths_list:
    #         print("Processing : ", path)
    #         src = rasterio.open(path)
    #         all_bands_arr = src.read(out_dtype=np.float32)
    #         for bands in all_bands_arr:
    #             array_list.append(bands)
    #         src.close()
    #     return array_list       

    # def create_tifffile_arrays(paths_list):
    #     array_list = []
    #     for path in paths_list:
    #         print("Processing : ", path)
    #         array_list.append(np.array(tiff.imread(path), dtype=np.float32))
    #     return array_list

    # stack = create_tifffile_arrays(paths_list)
    # nb_bands = len(stack)
    # sensors_name = 'sen2_raw'

    # save_path = time.strftime('results/' + sensors_name + '_' + '%Y-%m-%d_%H-%M-%S', time.localtime())

    # np.save(save_path, stack)

    # load np
    print('loading numpies')
    sen2 = np.load('results/sen2_raw_2022-11-09_23-26-04.npy')
    sen2 = np.transpose(sen2, (0, 3, 1, 2))

    def iter_windows_rasterio(src_ds, width, height, boundless=False):
        offsets = product(range(0, src_ds.meta['width'], width), range(0, src_ds.meta['height'], height))
        big_window = windows.Window(col_off=0, row_off=0, width=src_ds.meta['width'], height=src_ds.meta['height'])
        for col_off, row_off in offsets:

            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

            if boundless:
                yield window
            else:
                yield window.intersection(big_window)

    def iter_windows_rasterio_shape(src_ds, width, height, boundless=False):
        offsets = product(range(0, src_ds.shape[0], width), range(0, src_ds.shape[1], height))
        big_window = windows.Window(col_off=0, row_off=0, width=src_ds.shape[0], height=src_ds.shape[1])
        for col_off, row_off in offsets:

            window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

            if boundless:
                yield window
            else:
                yield window.intersection(big_window)

    def sliding_window(image, stepSize_y, stepSize_x, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[2], stepSize_y):
            for x in range(0, image.shape[1], stepSize_x):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    print()

    sliding_half = sliding_window(sen2[0], stepSize_y=256, stepSize_x=128, windowSize=(256, 256))
    sliding = sliding_window(sen2[0], stepSize_y=256, stepSize_x=256, windowSize=(256, 256))

    print(len(list(sliding_half)))
    print(len(list(sliding)))


    print('testing balls')

    test_my_balls = iter_windows_rasterio_shape(sen2[0], 256, 256)

    print(test_my_balls)
