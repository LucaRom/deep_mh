import os
import torch
import tifffile as tiff
import numpy as np

data_path = "/mnt/f/00_Donnees_SSD/256/mask_multiclass_3223_9c"
images = [x for x in os.listdir(data_path) if x[-4:] == ".tif"]

labels = (0, 1, 2, 3, 4, 5, 6, 7, 8)
labels_dict = dict.fromkeys(labels, 0)

for im in images: 
    img_path = os.path.join(data_path , im)

    na = np.array(tiff.imread(img_path))

    no_class, counts = np.unique(na, return_counts=True)

    for i, j in enumerate(no_class):
        labels_dict[j] = labels_dict.get(j, 0) + counts[i]

total_pixels = sum(labels_dict.values())
class_weights = {}
for label, count in labels_dict.items():
    class_weights[label] = 1 - count / total_pixels

print(class_weights)