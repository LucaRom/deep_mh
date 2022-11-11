import numpy as np
import torchmetrics
import os
import rasterio
import tifffile as tiff
import time

# internal import
from img_paths import get_estrie_paths

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

class estrie_dataset_from_nps(Dataset):
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

    # Calculer les statistiques generales 

        # Moyenne
        # ecart type
        # Autres?


    # Loading paths and images
    paths_lst = get_estrie_paths()
    sen2_paths = paths_lst[0:2]
    sen1_paths = paths_lst[2:4]
    mhc_paths  = paths_lst[4]
    slo_paths  = paths_lst[5]
    tpi_paths  = paths_lst[6]
    tri_paths  = paths_lst[7]
    twi_paths  = paths_lst[8]

    loads_paths_len = len(sen2_paths) + len(sen1_paths) + len([mhc_paths]) + len([slo_paths]) + len([tpi_paths]) \
                      + len([tri_paths]) + len([twi_paths])

    assert len(paths_lst) ==  loads_paths_len, f'Got a list of {len(paths_lst)}, but loaded {loads_paths_len}'

    def create_tifffile_arrays(paths_list):
        array_list = []
        for path in paths_list:
            print("Processing : ", path)
            array_list.append(np.array(tiff.imread(path), dtype=np.float32))
        return array_list

    def save_np_intermediate_arr(arr, obj_name):
        print('Saving obj_name with ', len(arr), 'images')
        save_path = time.strftime('results/' + obj_name + '_' + '%Y-%m-%d_%H-%M-%S', time.localtime())
        np.savez(save_path, arr)

    # Create and store Sen2 arrays
    #save_np_intermediate_arr(create_tifffile_arrays(sen2_paths), 'e_sen2_raw')

    # Create and store Sen1 arrays
    save_np_intermediate_arr(create_tifffile_arrays(sen1_paths), 'e_sen1_raw')

    # Create and store Lidar arrays arrays
    #save_np_intermediate_arr(create_tifffile_arrays(paths_lst[4:9]), 'e_lidar_raw')


    # # load np
    # print('loading numpies')
    # sen2 = np.load('results/sen2_raw_2022-11-09_23-26-04.npy') # Output I x C x H x W (I = number of file arrays)
    # sen2_trans = np.transpose(sen2, (0, 3, 1, 2))

    # def sliding_window(image, stepSize_y, stepSize_x, windowSize):
    #     # slide a window across the image
    #     for y in range(0, image.shape[2], stepSize_y):
    #         for x in range(0, image.shape[1], stepSize_x):
    #             # yield the current window
    #             yield (x, y, image[:, y:y + windowSize[1], x:x + windowSize[0]])

    # print()

    # sen2_ete = np.transpose(sen2[0],(1,2,0))
    # sen2_print = np.transpose(sen2[1],(1,2,0))

    # sen2_ete_2 = sen2[0]
    # sen2_print_2 = sen2[1]

    # senstack = np.dstack((sen2_ete, sen2_print))

    # senstack2 = np.stack((sen2_ete_2, sen2_print_2))

    # sliding_half = sliding_window(sen2[0], stepSize_y=256, stepSize_x=128, windowSize=(256, 256))
    # sliding = sliding_window(sen2[0], stepSize_y=256, stepSize_x=256, windowSize=(256, 256))



    # print(len(list(sliding_half)))
    # print(len(list(sliding)))


    # print('testing balls')
