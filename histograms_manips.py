from email.mime import image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import seaborn as sbs

# Windows
# Estrie sentinel 2 image : 
e_im_ete = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/ete/S2_estrie_3m_ete_septembre2020.tif"
e_im_prin = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/print/S2_estrie_3m_printemps_mai2020.tif"

k_im_ete = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/ete/s2_kenauk_3m_ete_aout2020.tif"
k_im_prin = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/print/S2_de_kenauk_3m_printemps_mai2020.tif"

k16_im_ete = "D:/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/ete/s2_kenauk2016_3m_ete.tif"
k16_im_print = "D:/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/print/s2_kenauk2016_3m_print.tif"

# Linux
#image_ete = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/ete/S2_estrie_3m_ete_septembre2020.tif"
#image_prin = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/print/S2_estrie_3m_printemps_mai2020.tif"

#k_image_ete = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/ete/s2_kenauk2016_3m_ete.tif"
#k_image_prin = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/print/s2_kenauk2016_3m_print.tif"

dataset = rasterio.open(k_im_ete)
array = dataset.read()

## Normalization
# Clipping
array = np.where(array < 0, 0, array)  # clip value under 0
array = np.where(array > 10000, 10000, array)  # clip value over 10 000

# Normalize
array = array/10000 # divide the array by 10000 so all the value are between [0-1]

# Standardization #TODO
#array = (array/mean)/std

# Sentinel 2A - Bands list
# B1  - Aerosol     (60m)     (0)
# B2  - Blue        (10m)     (1)
# B3  - Green       (10m)     (2)
# B4  - Red         (10m)     (3)
# B5  - RedEdge 1   (20m)     (4)
# B6  - RedEdge 2   (20m)     (5)
# B7  - RedEdge 3   (20m)     (6)
# B8  - NIR         (10m)     (7)
# B8A - RedEdge 4   (20m)     (8)
# B9  - Water Vapor (60m)     (9)
# B11 - SWIR 1      (20m)     (10)
# B12 - SWIR 2      (20m)     (11)

red = array[3]
green = array[2]
blue = array[1]
nir  = array[7]

# Plot RGB-NIR to compare... 
kwargs = dict(alpha=0.5, bins=256, density=True)

plt.suptitle('Histogram', y=1.05, size=16)
#plt.hist(array, density=True)
# plt.hist(red.flatten(), **kwargs, color='r', label='Red')
# plt.hist(green.flatten(), **kwargs, color='g', label='Green')
# plt.hist(blue.flatten(), **kwargs, color='b', label='Blue')
# plt.hist(nir.flatten(), **kwargs, color='y', label='NIR')
plt.distplot(red.flatten(), color='red', label='Red', **kwargs)
# plt.hist(green.flatten(), **kwargs, color='g', label='Green')
# plt.hist(blue.flatten(), **kwargs, color='b', label='Blue')
# plt.hist(nir.flatten(), **kwargs, color='y', label='NIR')

plt.legend()
plt.show()