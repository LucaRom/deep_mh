import os
#from PIL import Image # PIL for RGB or RGBI only
import tifffile as tiff
from torch.utils.data import Dataset
import numpy as np
import torch
import rasterio

# Pour tests
from torch.utils.data import DataLoader, random_split

class estrie_stack(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)
        self.all_img = [x for x in os.listdir(image_dir) if x.endswith(('.tif'))]

        # Tri sur les images # TODO Optimiser parce que très lent et difficile à debugger
                                # peut être faire en parrallèle? ou juste faire le tri avant
                                # avec une utility
        # Remove images in folder with wrong shape and no datas
        wanted_shape = (512, 512, 12) # might be better to set as variable/parameter
        self.images = []

        for i in self.all_img :
            img_path = os.path.join(image_dir, i)
            test_img = np.array(tiff.imread(img_path), dtype=np.float32)
            
            if np.any(test_img == 0) or test_img.shape != wanted_shape:
                pass
            else:
                self.images.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2", "mask_bin"))
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2", "mask_mutli"))
        mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2", "lidar_mnt"))
        
        image = np.array(tiff.imread(img_path), dtype=np.float32)

        # normalize the bands
        # clip the value between [0 - 10000]
        image = np.where(image < 0, 0, image)  # clip value under 0
        image = np.where(image > 10000, 10000, image)  # clip value over 10 000
        # divide the array by 10000 so all the value are between [0-1]
        image = image/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        # Stack optical and lidar
        image = np.dstack((image, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return image, mask

class estrie_stack2(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)
        self.all_img = [x for x in os.listdir(image_dir) if x.endswith(('.tif'))]

        # Tri sur les images # TODO Optimiser parce que très lent et difficile à debugger
                                # peut être faire en parrallèle? ou juste faire le tri avant
                                # avec une utility
        # Remove images in folder with wrong shape and no datas
        wanted_shape = (512, 512, 12) # might be better to set as variable/parameter
        self.images = []

        for i in self.all_img :
            img_path = os.path.join(image_dir, i)
            test_img = np.array(tiff.imread(img_path), dtype=np.float32)
            
            if np.any(test_img == 0) or test_img.shape != wanted_shape:
                pass
            else:
                self.images.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2", "mask_bin"))
        #mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2", "mask_mutli"))
        mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2", "lidar_mnt"))
        
        #print(img_path, mask_path, mnt_path)

        image = np.array(tiff.imread(img_path), dtype=np.float32)

        # normalize the bands
        # clip the value between [0 - 10000]
        image = np.where(image < 0, 0, image)  # clip value under 0
        image = np.where(image > 10000, 10000, image)  # clip value over 10 000
        # divide the array by 10000 so all the value are between [0-1]
        image = image/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        # Stack optical and lidar
        image = np.dstack((image, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path))
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return image, mask, img_path

class KenaukDataset_stack(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)
        self.images = [x for x in os.listdir(image_dir) if x.endswith(('.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2", "mask_bin"))
        mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2", "lidar"))
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2_print", "mask_bin"))
        # mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2_print", "lidar"))
        
        
        # input depug 
        #img_path = os.path.join(self.image_dir, self.images[68])
        #mask_path = os.path.join(self.mask_dir, self.images[68].replace("sen2_print", "mask_bin"))
        
        image = np.array(tiff.imread(img_path), dtype=np.float32)

        # normalize the bands
        # clip the value between [0 - 10000]
        image = np.where(image < 0, 0, image)  # clip value under 0
        image = np.where(image > 10000, 10000, image)  # clip value over 10 000
        # divide the array by 10000 so all the value are between [0-1]
        image = image/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        # Create the stack
        image = np.dstack((image, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

        # Cast to tensor for better permute
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return image, mask #, img_path

class KenaukDataset_stack2(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)
        self.images = [x for x in os.listdir(image_dir) if x.endswith(('.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2", "mask_bin"))
        mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2", "lidar"))
        # mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2_print", "mask_bin"))
        # mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2_print", "lidar"))
        
        
        # input depug 
        #img_path = os.path.join(self.image_dir, self.images[68])
        #mask_path = os.path.join(self.mask_dir, self.images[68].replace("sen2_print", "mask_bin"))
        
        image = np.array(tiff.imread(img_path), dtype=np.float32)

        # normalize the bands
        # clip the value between [0 - 10000]
        image = np.where(image < 0, 0, image)  # clip value under 0
        image = np.where(image > 10000, 10000, image)  # clip value over 10 000
        # divide the array by 10000 so all the value are between [0-1]
        image = image/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        # Create the stack
        image = np.dstack((image, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

        # Cast to tensor for better permute
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return image, mask, img_path

class KenaukDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)
        self.images = [x for x in os.listdir(image_dir) if x.endswith(('.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2_print", "mask_bin"))
        mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2_print", "lidar"))
        
        # input depug 
        #img_path = os.path.join(self.image_dir, self.images[68])
        #mask_path = os.path.join(self.mask_dir, self.images[68].replace("sen2_print", "mask_bin"))
        
        # debug
        #print(index)
        #print(self.images) #list des images
        #print(img_path)
        #print(mask_path)

        ### Exemple avec PIL pour 3 ou 4 bandes
        # #image = np.array(Image.open(img_path).convert("RGB"))
        # image = np.array(Image.open(img_path))
        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # #mask[mask == 255.0] = 1.0

        ### Version tiffile
        #image = np.array(tiff.imread(img_path).transpose(), dtype=np.float32)

        img_opt = np.array(tiff.imread(img_path), dtype=np.float32)
        #img_opt = np.array(tiff.imread(img_path).transpose([2, 0, 1]), dtype=np.float32)
        #img_opt = np.array(tiff.imread(img_path).transpose(2,1,0), dtype=np.float32)
        #img_opt = np.array(tiff.imread(img_path), dtype=np.float32)

        #print("debug")

        # normalize the bands
        # clip the value between [0 - 10000]
        img_opt = np.where(img_opt < 0, 0, img_opt)  # clip value under 0
        img_opt = np.where(img_opt > 10000, 10000, img_opt)  # clip value over 10 000
        # divide the array by 10000 so all the value are between [0-1]
        img_opt = img_opt/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        #img_opt = np.dstack((img_opt, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt)
        img_opt = img_opt.permute(2,0,1)
        img_mnt = torch.from_numpy(img_mnt)
        img_mnt = img_mnt.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return img_opt, img_mnt, mask

class KenaukDataset_rasterio(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)

        # Tri sur les images 
        self.images = [x for x in os.listdir(image_dir) if x.endswith(('.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2", "mask_bin"))
        mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2", "lidar"))
        
        # # Extract CRS and transforms
        # src = rasterio.open(img_path)
        # sample_crs = src.crs
        # transform_ori = src.transform
        # src.close() # Needed?

        img_opt = np.array(tiff.imread(img_path), dtype=np.float32)


        # normalize the bands
        # clip the value between [0 - 10000]
        img_opt = np.where(img_opt < 0, 0, img_opt)  # clip value under 0
        img_opt = np.where(img_opt > 10000, 10000, img_opt)  # clip value over 10 000

        # divide the array by 10000 so all the value are between [0-1]
        img_opt = img_opt/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        img_opt = np.dstack((img_opt, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt)
        img_opt = img_opt.permute(2,0,1)
        img_mnt = torch.from_numpy(img_mnt)
        img_mnt = img_mnt.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return img_opt, img_mnt, mask, img_path

# classes temporaire à cause des nomenclature des fichier
class KenaukDataset_rasterio2(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        #self.images = os.listdir(image_dir)

        # Tri sur les images 
        self.images = [x for x in os.listdir(image_dir) if x.endswith(('.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2_print", "mask_bin"))
        mnt_path = os.path.join(self.mnt_dir, self.images[index].replace("sen2_print", "lidar"))
        
        # # Extract CRS and transforms
        # src = rasterio.open(img_path)
        # sample_crs = src.crs
        # transform_ori = src.transform
        # src.close() # Needed?

        img_opt = np.array(tiff.imread(img_path), dtype=np.float32)


        # normalize the bands
        # clip the value between [0 - 10000]
        img_opt = np.where(img_opt < 0, 0, img_opt)  # clip value under 0
        img_opt = np.where(img_opt > 10000, 10000, img_opt)  # clip value over 10 000

        # divide the array by 10000 so all the value are between [0-1]
        img_opt = img_opt/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        img_opt = np.dstack((img_opt, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt)
        img_opt = img_opt.permute(2,0,1)
        img_mnt = torch.from_numpy(img_mnt)
        img_mnt = img_mnt.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return img_opt, img_mnt, mask, img_path


# rasterio classe fait juste retourner le path vers l'image et le CRS et les Transforms
# sont extraits en dehors du dataset
class estrie_rasterio(Dataset):
    """ 
        The folder containing 'sen2_ete' is filtered to respect the wanted files shape for training. If this is not the 
        case, you need to change '/sen2_ete' from the instance variable 'self.images' called in the init phase of the 
        class.

    """
    def __init__(self, train_dir, classif_mode, transform=None):
        self.image_dir = train_dir
        self.classif_mode = classif_mode
        self.transform = transform       
        self.images = [x for x in os.listdir(train_dir + "/sen2_ete") if x.endswith(('.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      
        # sentinel 2 images
        sen2_ete_path = os.path.join(self.image_dir, 'sen2_ete', self.images[index])
        sen2_print_path = os.path.join(self.image_dir, 'sen2_print', self.images[index].replace("ete", "print"))

        # lidar images
        mnt_path = os.path.join(self.image_dir, 'mnt', self.images[index].replace("sen2_ete", "mnt"))
        mhc_path = os.path.join(self.image_dir, 'mhc', self.images[index].replace("sen2_ete", "mhc"))
        slopes_path = os.path.join(self.image_dir, 'pentes', self.images[index].replace("sen2_ete", "pentes"))
        tpi_path = os.path.join(self.image_dir, 'tpi', self.images[index].replace("sen2_ete", "tpi"))
        tri_path = os.path.join(self.image_dir, 'tri', self.images[index].replace("sen2_ete", "tri"))
        twi_path = os.path.join(self.image_dir, 'twi', self.images[index].replace("sen2_ete", "twi"))

        #print("my ass")

        if self.classif_mode == "bin":
            mask_path = os.path.join(self.image_dir, 'mask_bin', self.images[index].replace("sen2_ete", "mask_bin"))
        elif self.classif_mode == "multiclass":
            mask_path = os.path.join(self.image_dir, 'mask_multiclass', self.images[index].replace("sen2_ete", "mask_multiclass"))
        else:
            print("There is something wrong with your mask dataset or paths (dataset.py)")
        
        # normalize the bands
        # clip the value between [0 - 10000]
        #TODO function or loop to normalize images instead or repeating
        # sen2_ete normalization
        sen2_ete_img = np.array(tiff.imread(sen2_ete_path), dtype=np.float32)
        sen2_ete_img = np.where(sen2_ete_img < 0, 0, sen2_ete_img)  # clip value under 0
        sen2_ete_img = np.where(sen2_ete_img > 10000, 10000, sen2_ete_img)  # clip value over 10 000
        sen2_ete_img = sen2_ete_img/10000 # divide the array by 10000 so all the value are between [0-1]

        # sen2_print normalization
        sen2_print_img = np.array(tiff.imread(sen2_print_path), dtype=np.float32)
        sen2_print_img = np.where(sen2_print_img < 0, 0, sen2_print_img)  # clip value under 0
        sen2_print_img = np.where(sen2_print_img > 10000, 10000, sen2_print_img)  # clip value over 10 000
        sen2_print_img = sen2_print_img/10000 # divide the array by 10000 so all the value are between [0-1]

        # stack both sentinel 2 images
        img_opt = np.dstack((sen2_ete_img, sen2_print_img))

        # Lidar images
        # TODO loop / function to expand_dims
        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

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

        img_lidar = np.dstack((img_mnt, img_mhc, img_slopes, img_tpi, img_tri, img_twi))

        # Mask images
        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt)
        img_opt = img_opt.permute(2,0,1)
        img_lidar = torch.from_numpy(img_lidar)
        img_lidar = img_lidar.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return img_opt, img_lidar, mask, sen2_ete_path

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
        self.images = [x for x in os.listdir(train_dir + "/sen2_ete") if x.endswith(('.tif'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):      
        # sentinel 2 images
        sen2_ete_path = os.path.join(self.image_dir, 'sen2_ete', self.images[index])
        sen2_print_path = os.path.join(self.image_dir, 'sen2_print', self.images[index].replace("ete", "print"))

        # lidar images
        mnt_path = os.path.join(self.image_dir, 'mnt', self.images[index].replace("sen2_ete", "mnt"))
        mhc_path = os.path.join(self.image_dir, 'mhc', self.images[index].replace("sen2_ete", "mhc"))
        slopes_path = os.path.join(self.image_dir, 'pentes', self.images[index].replace("sen2_ete", "pentes"))
        tpi_path = os.path.join(self.image_dir, 'tpi', self.images[index].replace("sen2_ete", "tpi"))
        tri_path = os.path.join(self.image_dir, 'tri', self.images[index].replace("sen2_ete", "tri"))
        twi_path = os.path.join(self.image_dir, 'twi', self.images[index].replace("sen2_ete", "twi"))

        # sentinel-1 images
        sen1_ete_path = os.path.join(self.image_dir, 'sen1_ete', self.images[index])
        sen1_print_path = os.path.join(self.image_dir, 'sen1_print', self.images[index].replace("ete", "print"))


        if self.classif_mode == "bin":
            mask_path = os.path.join(self.image_dir, 'mask_bin', self.images[index].replace("sen2_ete", "mask_bin"))
        elif self.classif_mode == "multiclass":
            mask_path = os.path.join(self.image_dir, 'mask_multiclass', self.images[index].replace("sen2_ete", "mask_multiclass"))
        else:
            print("There is something wrong with your mask dataset or paths (dataset.py)")
        
        # normalize the bands
        # clip the value between [0 - 10000]
        #TODO function or loop to normalize images instead or repeating
        # sen2_ete normalization
        sen2_ete_img = np.array(tiff.imread(sen2_ete_path), dtype=np.float32)
        sen2_ete_img = np.where(sen2_ete_img < 0, 0, sen2_ete_img)  # clip value under 0
        sen2_ete_img = np.where(sen2_ete_img > 10000, 10000, sen2_ete_img)  # clip value over 10 000
        sen2_ete_img = sen2_ete_img/10000 # divide the array by 10000 so all the value are between [0-1]

        # sen2_print normalization
        sen2_print_img = np.array(tiff.imread(sen2_print_path), dtype=np.float32)
        sen2_print_img = np.where(sen2_print_img < 0, 0, sen2_print_img)  # clip value under 0
        sen2_print_img = np.where(sen2_print_img > 10000, 10000, sen2_print_img)  # clip value over 10 000
        sen2_print_img = sen2_print_img/10000 # divide the array by 10000 so all the value are between [0-1]

        # stack both sentinel 2 images
        img_opt = np.dstack((sen2_ete_img, sen2_print_img))

        # Lidar images
        # TODO loop / function to expand_dims
        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

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

        img_lidar = np.dstack((img_mnt, img_mhc, img_slopes, img_tpi, img_tri, img_twi))

        # Sentinel-1 images
        sen1_ete_img = np.array(tiff.imread(sen1_ete_path), dtype=np.float32)
        sen1_print_img = np.array(tiff.imread(sen1_print_path), dtype=np.float32)

        img_rad = np.dstack((sen1_ete_img, sen1_print_img)) # stack both sen-1 images

        # Mask images
        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt)
        img_opt = img_opt.permute(2,0,1)
        img_lidar = torch.from_numpy(img_lidar)
        img_lidar = img_lidar.permute(2,0,1)
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


