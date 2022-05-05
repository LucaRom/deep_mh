import os
#from PIL import Image # PIL for RGB or RGBI only
import tifffile as tiff
from torch.utils.data import Dataset
import numpy as np
import torch

class KenaukDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mnt_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mnt_dir = mnt_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

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


        image = np.array(tiff.imread(img_path), dtype=np.float32)
        #image = np.array(tiff.imread(img_path).transpose([2, 0, 1]), dtype=np.float32)
        #image = np.array(tiff.imread(img_path).transpose(2,1,0), dtype=np.float32)
        #image = np.array(tiff.imread(img_path), dtype=np.float32)

        #print("debug")

        # normalize the bands
        # clip the value between [0 - 10000]
        image = np.where(image < 0, 0, image)  # clip value under 0
        image = np.where(image > 10000, 10000, image)  # clip value over 10 000
        # divide the array by 10000 so all the value are between [0-1]
        image = image/10000

        img_mnt = np.array(tiff.imread(mnt_path))
        img_mnt = np.expand_dims(img_mnt, axis=2)

        image = np.dstack((image, img_mnt))

        #mask = np.array(tiff.imread(mask_path)) / 255
        mask = np.array(tiff.imread(mask_path)) 
        #mask[mask == 255.0] = 1.0

       #print("stop") # Debug breakpoint

### Example pour tifffile ###
# for img_id in train_ids:
#     img_m = normalize(tiff.imread(os.path.join(data_path, 'mband/{}.tif'.format(img_id))).transpose([1, 2, 0]))
#     mask = tiff.imread(os.path.join(data_path, 'gt_mband/{}.tif'.format(img_id))).transpose([1, 2, 0]) / 255


        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = torch.transpose(augmentations["image"], 1,0)
        #     mask = torch.transpose(augmentations["mask"], 1,0)


        # Cast to tensor for better permute
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        mask  = torch.from_numpy(mask)

        return image, mask