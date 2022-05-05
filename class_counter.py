import os
#from PIL import Image # PIL for RGB or RGBI only
import tifffile as tiff
from torch.utils.data import Dataset
import numpy as np
import torch


TRAIN_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/train"
TRAIN_MASK_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/train"
VAL_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/val"
VAL_MASK_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/val"


images = os.listdir(TRAIN_MASK_DIR)

print(images)

# img_path = os.path.join(self.image_dir, self.images[index])


liste_ratio = []
liste_ratio_moy_ele = []
liste_ratio_5 = []

for im in images:
    mask_path = os.path.join(TRAIN_MASK_DIR, im)
    #print(mask_path)
    #image = np.array(tiff.imread(mask_path).transpose([2, 1, 0]), dtype=np.float32)
    image = np.array(tiff.imread(mask_path), dtype=np.float32)
    unique, counts = np.unique(image, return_counts=True)


    class_0 = counts[0]
    if len(counts) == 2: 
        class_1 = counts[1]
    else:
        class_1 = 0

    ratio_1_0 = class_1/class_0
    liste_ratio.append(ratio_1_0)

    if ratio_1_0 >= 0.07:
        liste_ratio_moy_ele.append(ratio_1_0)

    if ratio_1_0 >= 0.04:
        liste_ratio_5.append(ratio_1_0)

    print(mask_path, unique, counts, "{:.0%}".format(ratio_1_0))

print("nombre d'échantillons scannés: ", len(liste_ratio))
average_ratio = sum(liste_ratio) / len(liste_ratio)
print("Moyenne ratio classe 1: ", average_ratio)
print("Nombre d'échantillon avec ratio au dessus ou égal à la moyenne: ", len(liste_ratio_moy_ele))
print("Nombre d'échantillon avec ratio au dessus de 5%: ", len(liste_ratio_5))



# img_path = os.path.join(self.image_dir, self.images[index])
# mask_path = os.path.join(self.mask_dir, self.images[index].replace("sen2_print", "mask_bin"))


# image = np.array(tiff.imread(img_path).transpose([2, 1, 0]), dtype=np.float32)
# mask = np.array(tiff.imread(mask_path))