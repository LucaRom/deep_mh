import os
#from PIL import Image # PIL for RGB or RGBI only
import tifffile as tiff
from torch.utils.data import Dataset
import numpy as np
import torch
import rasterio
from multiprocessing import Manager

# Pour tests
from torch.utils.data import DataLoader, random_split

class estrie_rasterio_3_inputs(Dataset):
    """ 
        The folder containing 'sen2_ete' is filtered to respect the wanted files shape for training. If this is not the 
        case, you need to change '/sen2_ete' from the instance variable 'self.images' called in the init phase of the 
        class.

    """
    def __init__(self, train_dir, classif_mode, transform=None):
        self.image_dir = train_dir
        self.classif_mode = classif_mode
        self.transform = transform   

        manager = Manager()    
        self.images = manager.list([x for x in os.listdir(train_dir + "/sen2_ete") if x.endswith(('.tif'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      
        # sentinel 2 images
        sen2_ete_path = os.path.join(self.image_dir, 'sen2_ete', self.images[index])
        sen2_print_path = os.path.join(self.image_dir, 'sen2_print', self.images[index].replace("ete", "print"))

        # lidar images
        #mnt_path = os.path.join(self.image_dir, 'mnt', self.images[index].replace("sen2_ete", "mnt"))
        mhc_path = os.path.join(self.image_dir, 'mhc', self.images[index].replace("sen2_ete", "mhc"))
        slopes_path = os.path.join(self.image_dir, 'pentes', self.images[index].replace("sen2_ete", "pentes"))
        tpi_path = os.path.join(self.image_dir, 'tpi', self.images[index].replace("sen2_ete", "tpi"))
        tri_path = os.path.join(self.image_dir, 'tri', self.images[index].replace("sen2_ete", "tri"))
        twi_path = os.path.join(self.image_dir, 'twi', self.images[index].replace("sen2_ete", "twi"))

        # sentinel-1 images
        sen1_ete_path = os.path.join(self.image_dir, 'sen1_ete', self.images[index]).replace("sen2_ete", "sen1_ete")
        sen1_print_path = os.path.join(self.image_dir, 'sen1_print', self.images[index].replace("sen2_ete", "sen1_print"))


        if self.classif_mode == "bin":
            mask_path = os.path.join(self.image_dir, 'mask_bin', self.images[index].replace("sen2_ete", "mask_bin"))
            #print("USING mask_bin FOR BINARY CLASSIFICATION FOR ESTRIE")
        elif self.classif_mode == "multiclass":
            mask_path = os.path.join(self.image_dir, 'mask_multiclass', self.images[index].replace("sen2_ete", "mask_multiclass"))
            #print("USING mask_multiclass FOR MULTICLASS CLASSIFICATION FOR ESTRIE")
        else:
            print("There is something wrong with your mask dataset or paths (dataset.py)")
        
        # print("Paths debugging")
        # print("Sen_2_ete_path : ", sen2_ete_path)
        # print("Sen_2_printemps_path : ", sen2_print_path)
        # print("MHC: ", mhc_path)
        # print("TPI, TRI, TWO: ", tpi_path, tri_path, twi_path)

        # normalize the bands
        # clip the value between [0 - 10000]
        #TODO function or loop to normalize images instead or repeating
        # sen2_ete clipping
        sen2_ete_img = np.array(tiff.imread(sen2_ete_path), dtype=np.float32)
        sen2_ete_img = np.where(sen2_ete_img < 0, 0, sen2_ete_img)  # clip value under 0
        sen2_ete_img = np.where(sen2_ete_img > 10000, 10000, sen2_ete_img)  # clip value over 10 000

        # Normalize
        #sen2_ete_img = sen2_ete_img/10000 # divide the array by 10000 so all the value are between [0-1]

        # sen2_print clipping
        sen2_print_img = np.array(tiff.imread(sen2_print_path), dtype=np.float32)
        sen2_print_img = np.where(sen2_print_img < 0, 0, sen2_print_img)  # clip value under 0
        sen2_print_img = np.where(sen2_print_img > 10000, 10000, sen2_print_img)  # clip value over 10 000

        # Normalize
        #sen2_print_img = sen2_print_img/10000 # divide the array by 10000 so all the value are between [0-1]


        # stack both sentinel 2 images
        img_opt = np.dstack((sen2_ete_img, sen2_print_img))

        # Lidar images
        # TODO loop / function to expand_dims
        #img_mnt = np.array(tiff.imread(mnt_path))
        #img_mnt = np.expand_dims(img_mnt, axis=2)

        img_mhc = np.array(tiff.imread(mhc_path))
        img_mhc = np.expand_dims(img_mhc, axis=2)

        img_slopes = np.array(tiff.imread(slopes_path))
        img_slopes = np.expand_dims(img_slopes, axis=2)

        img_tpi = np.array(tiff.imread(tpi_path))
        img_tpi = np.expand_dims(img_tpi, axis=2)

        img_tri = np.array(tiff.imread(tri_path))
        img_tri = np.expand_dims(img_tri, axis=2)

        img_twi = np.array(tiff.imread(twi_path))
        img_twi = np.expand_dims(img_twi, axis=2)

        #img_lidar = np.dstack((img_mnt, img_mhc, img_slopes, img_tpi, img_tri, img_twi))
        img_lidar = np.dstack((img_mhc, img_slopes, img_tpi, img_tri, img_twi))

        if img_lidar.dtype != 'float32':
            img_lidar = np.float32(img_lidar) # Only for overlapping dataset #TODO
        else:
            pass

        # Sentinel-1 images
        sen1_ete_img = np.array(tiff.imread(sen1_ete_path), dtype=np.float32)
        sen1_print_img = np.array(tiff.imread(sen1_print_path), dtype=np.float32)

        img_rad = np.dstack((sen1_ete_img, sen1_print_img)) # stack both sen-1 images

        # Mask images
        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

        if mask.dtype != 'float32':
            mask = np.float32(mask) # Only for overlapping dataset #TODO
        else:
            pass

       #print("stop") # Debug breakpoint

        ## Mean and std value lists
        # Sentinel 2
        s2_e_e_mean = torch.tensor([259.971087045696, 277.3490067676725, 520.4650232890134, 342.23574780553645, 906.7611488412249, 2656.3582951694643, 3203.3543093369944, 3389.6250611778078, 3487.079600166239, 3555.416409200909, 1714.2260907527316, 828.2768740555728, 457.4229830346009, 501.79759875320303, 694.4711397083421, 835.1158882308216, 1219.9447441650816, 1823.0661322180392, 2064.6505317461747, 2316.1887302003915, 2363.5869859139643, 2359.4662122932396, 2390.6124116260303, 1586.6126304451745])
        s2_e_e_std  = torch.tensor([525.5551122108338, 526.4768589585602, 515.8903727938966, 527.3656790023017, 561.5222503677404, 836.1454714836563, 984.9190349745415, 1067.0420278801334, 1026.7569263359944, 1066.123618103052, 630.0584359871733, 505.2076063419134, 169.44646075504082, 249.03030944938908, 293.96819726121373, 408.20429488371605, 392.1811051266158, 492.36521601358254, 550.8773405439316, 623.9017038640061, 590.0457818993959, 540.556974947324, 740.4564895487368, 581.7629650224691])
        
        # Sentinel 1
        s1_e_p_mean = torch.tensor([-15.479797, -9.211855, 6.267961, -15.0310545, -9.519093, 5.5120163])
        s1_e_p_std  = torch.tensor([1.622046, 1.8651232, 1.2285297, 2.1044014, 1.9065734, 1.37706]) 

        # Lidar
        estrie_lidar_mean = torch.tensor([7.798849, 5.5523205, 0.0029951811, 0.06429929, 6.7409873])
        estrie_lidar_std  = torch.tensor([7.033332, 5.196636, 1.0641352, 0.06102526, 3.182435])

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt)
        img_opt = img_opt.permute(2,0,1)
        img_rad = torch.from_numpy(img_rad)
        img_rad = img_rad.permute(2,0,1)
        img_lidar = torch.from_numpy(img_lidar)
        img_lidar = img_lidar.permute(2,0,1)        
        mask  = torch.from_numpy(mask)

        # Apply standardization (see : discuss.pytorch.org/t/how-to-normalize-multidimensional-tensor/65304)
        img_opt = img_opt.sub_(s2_e_e_mean[:, None, None]).div_(s2_e_e_std[:, None, None])
        img_lidar = img_lidar.sub_(estrie_lidar_mean[:, None, None]).div_(estrie_lidar_std[:, None, None])
        img_rad = img_rad.sub_(s1_e_p_mean[:, None, None]).div_(s1_e_p_std[:, None, None])

        return img_opt, img_lidar, mask, img_rad, sen2_ete_path

class kenauk_rasterio_3_inputs(Dataset): 
    """ 
        The folder containing 'sen2_ete' is filtered to respect the wanted files shape for training. If this is not the 
        case, you need to change '/sen2_ete' from the instance variable 'self.images' called in the init phase of the 
        class.

    """
    def __init__(self, train_dir, classif_mode, transform=None):
        self.image_dir = train_dir
        self.classif_mode = classif_mode
        self.transform = transform
        self.image_dir_2 = "/mnt/SN750/00_Donnees_SSD/256_HM_ST_kenauk/"     
        
        #self.images = [x for x in os.listdir(train_dir + "/sen2_ete") if x.endswith(('.tif'))]

        #TODO please make this better (TEMP) : 

        print("\n \nWARNING WARNING WARNING WARNING WARNING : bad_images are removed for training and it might be very bad if you are trying to run tests on kenauk dataset\n \n")

        #print("\n \nWARNING WARNING WARNING WARNING WARNING : bad_images are NOT removed for training and \n \n")

        bad_images = ['sen2_ete.247.tif', 'sen2_ete.402.tif', 'sen2_ete.730.tif', 'sen2_ete.731.tif', 'sen2_ete.732.tif', 
        'sen2_ete.733.tif', 'sen2_ete.734.tif', 'sen2_ete.735.tif', 'sen2_ete.736.tif', 'sen2_ete.737.tif', 
        'sen2_ete.738.tif', 'sen2_ete.739.tif', 'sen2_ete.740.tif', 'sen2_ete.741.tif', 'sen2_ete.742.tif', 
        'sen2_ete.743.tif', 'sen2_ete.123.tif', 'sen2_ete.154.tif', 'sen2_ete.185.tif', 'sen2_ete.216.tif', 
        'sen2_ete.278.tif', 'sen2_ete.30.tif', 'sen2_ete.309.tif', 'sen2_ete.340.tif', 'sen2_ete.371.tif', 
        'sen2_ete.433.tif', 'sen2_ete.464.tif', 'sen2_ete.495.tif', 'sen2_ete.526.tif', 'sen2_ete.557.tif', 
        'sen2_ete.588.tif', 'sen2_ete.61.tif', 'sen2_ete.619.tif', 'sen2_ete.650.tif', 'sen2_ete.681.tif', 
        'sen2_ete.712.tif', 'sen2_ete.713.tif', 'sen2_ete.714.tif', 'sen2_ete.715.tif', 'sen2_ete.716.tif', 
        'sen2_ete.717.tif', 'sen2_ete.718.tif', 'sen2_ete.719.tif', 'sen2_ete.720.tif', 'sen2_ete.721.tif', 
        'sen2_ete.722.tif', 'sen2_ete.723.tif', 'sen2_ete.724.tif', 'sen2_ete.725.tif', 'sen2_ete.726.tif', 
        'sen2_ete.727.tif', 'sen2_ete.728.tif', 'sen2_ete.729.tif', 'sen2_ete.92.tif']

        #self.images = [x for x in os.listdir(train_dir + "/sen2_ete") if x.endswith(('.tif')) if x not in bad_images]
        self.images = [x for x in os.listdir(self.image_dir_2 + "/sen2_ete") if x.endswith(('.tif')) if x not in bad_images]
        #self.images = [x for x in os.listdir(train_dir + "/sen2_ete") if x.endswith(('.tif'))]

        print("debug this mess")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      
        # sentinel 2 images
        sen2_ete_path = os.path.join(self.image_dir_2, 'sen2_ete', self.images[index])
        sen2_print_path = os.path.join(self.image_dir_2, 'sen2_print', self.images[index].replace("ete", "print"))

        # lidar images
        #mnt_path = os.path.join(self.image_dir, 'mnt', self.images[index].replace("sen2_ete", "mnt"))
        mhc_path = os.path.join(self.image_dir, 'mhc', self.images[index].replace("sen2_ete", "mhc"))
        slopes_path = os.path.join(self.image_dir, 'pentes', self.images[index].replace("sen2_ete", "pentes"))
        tpi_path = os.path.join(self.image_dir, 'tpi', self.images[index].replace("sen2_ete", "tpi"))
        tri_path = os.path.join(self.image_dir, 'tri', self.images[index].replace("sen2_ete", "tri"))
        twi_path = os.path.join(self.image_dir, 'twi', self.images[index].replace("sen2_ete", "twi"))

        # sentinel-1 images
        sen1_ete_path = os.path.join(self.image_dir_2, 'sen1_ete', self.images[index]).replace("sen2_ete", "sen1_ete")
        sen1_print_path = os.path.join(self.image_dir_2, 'sen1_print', self.images[index].replace("sen2_ete", "sen1_print"))


        if self.classif_mode == "bin":
            mask_path = os.path.join(self.image_dir, 'mask_bin', self.images[index].replace("sen2_ete", "mask_bin"))
            #print("USING mask_bin FOR BINARY CLASSIFICATION FOR KENAUK")
        elif self.classif_mode == "multiclass":
            mask_path = os.path.join(self.image_dir, 'mask_multiclass', self.images[index].replace("sen2_ete", "mask_multiclass"))
            #print("USING mask_bin FOR MULTICLASS CLASSIFICATION FOR KENAUK")
        else:
            print("There is something wrong with your mask dataset or paths (dataset.py)")

        print("Paths debugging")
        print("Sen_2_ete_path : ", sen2_ete_path)
        print("Sen_2_printemps_path : ", sen2_print_path)
        print("Sen_1_ete_path : ", sen1_ete_path)
        print("Sen_1_printemps_path : ", sen1_print_path)
        print("MHC: ", mhc_path)
        print("TPI, TRI, TWO: ", tpi_path, tri_path, twi_path)

        # normalize the bands
        # clip the value between [0 - 10000]
        #TODO function or loop to normalize images instead or repeating
        # sen2_ete normalization
        sen2_ete_img = np.array(tiff.imread(sen2_ete_path), dtype=np.float32)
        # sen2_ete_img = np.where(sen2_ete_img < 0, 0, sen2_ete_img)  # clip value under 0
        # sen2_ete_img = np.where(sen2_ete_img > 10000, 10000, sen2_ete_img)  # clip value over 10 000
        # sen2_ete_img = sen2_ete_img/10000 # divide the array by 10000 so all the value are between [0-1]

        # sen2_print normalization
        sen2_print_img = np.array(tiff.imread(sen2_print_path), dtype=np.float32)
        # sen2_print_img = np.where(sen2_print_img < 0, 0, sen2_print_img)  # clip value under 0
        # sen2_print_img = np.where(sen2_print_img > 10000, 10000, sen2_print_img)  # clip value over 10 000
        # sen2_print_img = sen2_print_img/10000 # divide the array by 10000 so all the value are between [0-1]

        # stack both sentinel 2 images
        img_opt = np.dstack((sen2_ete_img, sen2_print_img))

        # Lidar images
        # TODO loop / function to expand_dims
        #img_mnt = np.array(tiff.imread(mnt_path))
        #img_mnt = np.expand_dims(img_mnt, axis=2)

        img_mhc = np.array(tiff.imread(mhc_path))
        img_mhc = np.expand_dims(img_mhc, axis=2)

        img_slopes = np.array(tiff.imread(slopes_path))
        img_slopes = np.expand_dims(img_slopes, axis=2)

        img_tpi = np.array(tiff.imread(tpi_path))
        img_tpi = np.expand_dims(img_tpi, axis=2)

        img_tri = np.array(tiff.imread(tri_path))
        img_tri = np.expand_dims(img_tri, axis=2)

        img_twi = np.array(tiff.imread(twi_path))
        img_twi = np.expand_dims(img_twi, axis=2)

        #img_lidar = np.dstack((img_mnt, img_mhc, img_slopes, img_tpi, img_tri, img_twi))
        img_lidar = np.dstack((img_mhc, img_slopes, img_tpi, img_tri, img_twi))

        if img_lidar.dtype != 'float32':
            img_lidar = np.float32(img_lidar) # Only for overlapping dataset #TODO
        else:
            pass

        # Sentinel-1 images
        sen1_ete_img = np.array(tiff.imread(sen1_ete_path), dtype=np.float32)
        sen1_print_img = np.array(tiff.imread(sen1_print_path), dtype=np.float32)

        img_rad = np.dstack((sen1_ete_img, sen1_print_img)) # stack both sen-1 images

        # Mask images
        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

        if mask.dtype != 'float32':
            mask = np.float32(mask) # Only for overlapping dataset #TODO
        else:
            pass

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt)
        img_opt = img_opt.permute(2,0,1)

        # Mean and std value lists
        # combined_mean = torch.tensor([
        #                 298.9376736842285, 295.4163414577072, 533.0512058140661, 335.011499544136, 904.7482179833103, 2769.694844441057, 3352.3481529620663, 3507.679339513949, 3628.128361722713, 3681.886714013677, 1769.3827647437795, 832.3841287235765,
        #                 389.73178500319295, 418.7855774087138, 656.2461824601817, 663.8277464836387, 1164.82661126222, 2267.199178786665, 2609.65239794843, 2826.5757717474403, 2885.967306830988, 2885.0676772511297, 2244.2198662295045, 1384.4035653061455
        #                 ])

        # combined_std  = torch.tensor([
        #                 449.1333965424019, 447.09781235125655, 440.1281124724915, 448.5562977082119, 489.71626393358594, 
        #                 882.6080290302698, 1074.64297530801, 1142.3471873971982, 1132.1216994695542, 1121.812465173574, 
        #                 637.2357446197225, 454.657718157629, 182.1481184415827, 252.22961441363324, 271.82436188432376, 
        #                 444.23478736031086, 384.49626301643093, 1043.6492154677742, 1255.1022957221837, 
        #                 1267.5933253040087, 1263.0965697948964, 1229.1994225452902, 745.0200230290296, 607.8933293590522
        #                 ])

        # Apply standardization (see : discuss.pytorch.org/t/how-to-normalize-multidimensional-tensor/65304)
        #img_opt = img_opt.sub_(combined_mean[:, None, None]).div_(combined_std[:, None, None])

        k_lidar_means = torch.tensor([13.348262, 13.45669, -0.006740755, -3.689763, 5.7766604])
        k_lidar_stds  = torch.tensor([7.7406297, 13.942361, 1.3129127, 241.4134, 5.6496654])

        img_lidar = torch.from_numpy(img_lidar)
        img_lidar = img_lidar.permute(2,0,1)

        # Standardization
        img_lidar = img_lidar.sub_(k_lidar_means[:, None, None]).div_(k_lidar_stds[:, None, None])

        mask  = torch.from_numpy(mask)
        img_rad = torch.from_numpy(img_rad)
        img_rad = img_rad.permute(2,0,1)

        return img_opt, img_lidar, mask, img_rad, sen2_ete_path

if __name__ == "__main__":
# Import data with custom loader
    TRAIN_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/train"
    TRAIN_MASK_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/train"
    TRAIN_MNT_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/train"
    VAL_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/val"
    VAL_MASK_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/val"
    VAL_MNT_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/val"


    # Path Estrie
    e_img_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/512/"
    e_mask_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_bin"
    #e_mask_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_multi" # Multiclass
    e_lidar_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/lidar_mnt"

    # Dataset Estrie
    train_estrie_ds = estrie_rasterio(
        train_dir=e_img_dir,
        classif_mode= "multiclass",
        #transform=train_transform
    )

    train_estrie_loader = DataLoader(
        train_estrie_ds,
        batch_size=4,
        num_workers=6,
        pin_memory=True,
        shuffle=True,
    )

    batch = next(iter(train_estrie_loader))
    print(batch)

    # train_ds = KenaukDataset(
    #     image_dir=TRAIN_IMG_DIR,
    #     mask_dir=TRAIN_MASK_DIR,
    #     mnt_dir=TRAIN_MNT_DIR,
    #     #transform=train_transform
    # )

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    #     shuffle=True,
    # )

    # train_rasterio_ds = KenaukDataset_rasterio(
    #     image_dir=TRAIN_IMG_DIR,
    #     mask_dir=TRAIN_MASK_DIR,
    #     mnt_dir=TRAIN_MNT_DIR,
    #     #transform=train_transform
    # )

    # train_rasterio_loader = DataLoader(
    #     train_rasterio_ds,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    #     shuffle=True,
    # )


    # Random split
    # train_set_size = int(len(train_estrie_ds) * 0.7)
    # valid_set_size = len(train_estrie_ds) - train_set_size
    # train_set, valid_set = random_split(train_estrie_ds, [train_set_size, valid_set_size])

    # print('debug')

    # train_estrie_loader = DataLoader(
    #     train_estrie_ds,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     pin_memory=PIN_MEMORY,
    #     shuffle=True,
    # )



    #print(train_loader)
    #print(train_estrie_loader)

    #batch = next(iter(train_loader))
    #print(batch)

    # single_img = TRAIN_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/train/sen2_print.12.tif"
    # dataset = rasterio.open(single_img)

    #img_opt, img_mnt, mask, sample_crs, transform_ori
    #train_opt, train_mnt, train_mask, img_path = next(iter(train_rasterio_loader))

    # # Extract CRS and transforms
    # src = rasterio.open(img_path[0])
    # sample_crs = src.crs
    # transform_ori = src.transform
    # src.close() # Needed?

    # test tri d'images NaN
    # Tri sur les images (shape et pas de nodata)
    # TODO regarder pour faire des nodata qui != 0
    # wanted_shape = (512, 512, 12)
    # all_img = [x for x in os.listdir(e_img_dir) if x.endswith(('.tif'))]
    # images = []

    # for i in all_img :
    #     img_path = os.path.join(e_img_dir, i)
    #     test_img = np.array(tiff.imread(img_path), dtype=np.float32)
        
    #     if np.any(test_img == 0) or test_img.shape != wanted_shape:
    #         pass
    #     else:
    #         images.append(i)

    # print(len(images))

    # enlever les images qui on des no_data
    # clean_img = 0
    # bad_img = 0
    # outside = 0
    # no_shape = 0

    # for _ in images:
    #     img_path = os.path.join(e_img_dir, _)
    #     print(img_path)
    #     #print(img_path)
    #     test_img = np.array(tiff.imread(img_path), dtype=np.float32)
    #     shape = (512, 512, 12)
    #     # verifier shape de l'image aussi ...
    
    #     unique, counts = np.unique(test_img, return_counts=True)
    #     print(unique)

    #     if test_img.shape == shape:
    #         if np.any(test_img == 0) and len(unique) == 1: # Enlever les samples avec que des 0
    #             print('Outside')
    #             outside += 1
    #         elif np.any(test_img == 0):
    #             print('Contient des no data')
    #             bad_img += 1
    #         else:
    #             print('all clear')
    #             clean_img += 1
    #     else:
    #         print('no shape')
    #         no_shape += 1

    # print(clean_img, bad_img, outside, no_shape)

    #print(sample_crs)

    #print(train_rasterio_loader)


