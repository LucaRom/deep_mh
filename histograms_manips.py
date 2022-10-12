from email.mime import image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from sys import platform
import os
import rasterio
from rasterio.plot import show_hist
import seaborn as sbs
from skimage.exposure import match_histograms

# Check platform for paths
print("Detected platform is : ", platform)
if platform == "linux" or platform == "linux2":
    print("Using paths for Linux")
    pre_path = "/mnt/Data"
elif platform == "win32":
    print("Using paths for Windows")
    pre_path = "D:"
else:
    "No platform detected"

## Windows
# Estrie sentinel 2
e_sen2_ete = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/ete/S2_estrie_3m_ete_septembre2020.tif")
e_sen2_prin = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/print/S2_estrie_3m_printemps_mai2020.tif")

e_sen1_ete = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1/ete/S1_estrie_3m_ete_septembre2020.tif")
e_sen1_prin = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1/print/S1_estrie_3m_printemps_mai2020.tif")

# e_mhc = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/mhc_estrie_3m.tif")
# e_slopes = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/pente_estrie_3m.tif"
# e_tpi = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tpi_estrie_3m.tif"
# e_tri = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tri_estrie_3m.tif"
# e_twi  = "D:/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/twi_estrie_3m.tif"

# Estrie sentinel 2


# Kenauk 2020 sentinel 2
k_sen2_ete = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/ete/s2_kenauk_3m_ete_aout2020.tif")
k_sen2_prin = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/print/S2_de_kenauk_3m_printemps_mai2020.tif")

k_sen1_ete = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/ete/s1_kenauk_3m_ete_aout2020.tif")
k_sen1_prin =  os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/print/S1_kenauk_3m_printemps_mai2020.tif")

#k_mhc = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mhc_kenauk_3m.tif"
# k_slopes = os.path.join(pre_path, "00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/pentes_kenauk_3m.tif")
# k_tpi = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tpi_kenauk_3m.tif"
# k_tri = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif"
# k_twi  = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/twi_kenauk_3m.tif"

# Kenauk 2020 sentinel 2

# Kenauk 2016 sentinel 2
# k16_im_ete = "D:/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/ete/s2_kenauk2016_3m_ete.tif"
# k16_im_prin = "D:/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/print/s2_kenauk2016_3m_print.tif"

# Linux
#image_ete = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/ete/S2_estrie_3m_ete_septembre2020.tif"
#image_prin = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/print/S2_estrie_3m_printemps_mai2020.tif"

#k_image_ete = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/ete/s2_kenauk2016_3m_ete.tif"
#k_image_prin = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk_2016/processed_raw/sen2/print/s2_kenauk2016_3m_print.tif"

# Images sentinel 2 - printemps
im_estrie = rasterio.open(e_sen1_prin)
im_kenauk20 = rasterio.open(k_sen1_prin)
#im_kenauk16 = rasterio.open(k16_im_prin)

# Images sentinel 2 - été
# im_estrie = rasterio.open(e_im_ete)
# im_kenauk20 = rasterio.open(k_im_ete)
#im_kenauk16 = rasterio.open(k16_im_ete)

ar_e = im_estrie.read()
ar_k = im_kenauk20.read()
#ar_k16 = im_kenauk16.read()

## Normalization
# Clipping
# ar_e = np.where(ar_e < 0, 0, ar_e)  # clip value under 0
# ar_e = np.where(ar_e > 10000, 10000, ar_e)  # clip value over 10 000

# ar_k = np.where(ar_k < 0, 0, ar_k) 
# ar_k = np.where(ar_k > 10000, 10000, ar_k)  

# ar_k16 = np.where(ar_k16 < 0, 0, ar_k16)
# ar_k16 = np.where(ar_k16 > 10000, 10000, ar_k16)


# Histogram Matching kenauk 2020 et Estrie
#ar_k = match_histograms(ar_k, ar_e, channel_axis=0)
#print("debug")

# Standardization temp
ar_k = ar_k.astype('float64')

#print(ar_k[0])

for band in range(len(ar_k)):
    # print("band number: ", band)
    # print(ar_k[band])
    # print(np.average(ar_k[band]))
    # print(np.std(ar_k[band]))
    ar_k[band] = (ar_k[band] - np.average(ar_k[band])) / np.std(ar_k[band])

#ar_k2 = (ar_k - np.average(ar_k)) / np.std(ar_k)

# print("debug")
# print(ar_k[0])



with rasterio.open(
    's1_kenauk_3m_print_STD.tif',
    'w',
    driver='GTiff',
    height=ar_k.shape[1],
    width=ar_k.shape[2],
    count=ar_k.shape[0],
    dtype=ar_k.dtype,
    crs=im_kenauk20.crs,
    transform=im_kenauk20.transform,
) as dst:
    dst.write(ar_k)

# print("debug")

# # Normalize
# #array = array/10000 # divide the array by 10000 so all the value are between [0-1]

# # ar_e = ar_e/10000
# # ar_k = ar_k/10000
# # ar_k16 = ar_k16/10000

# # Standardization
# def standardize_array(ar_name, band_num):
#     return (ar_name[band_num] - np.average(ar_name[band_num])) / np.std(ar_name[band_num])


# # with rasterio.open(
# #     'S2_kenauk_3m_printemps_mai2020_HM_ESTRIE.tif',
# #     'w',
# #     driver='GTiff',
# #     height=ar_k.shape[1],
# #     width=ar_k.shape[2],
# #     count=ar_k.shape[0],
# #     dtype=ar_k.dtype,
# #     crs=im_kenauk20.crs,
# #     transform=im_kenauk20.transform,
# # ) as dst:
# #     dst.write(ar_k) 


# # Sentinel 2A - Bands list
# # B1  - Aerosol     (60m)     (0)
# # B2  - Blue        (10m)     (1)
# # B3  - Green       (10m)     (2)
# # B4  - Red         (10m)     (3)
# # B5  - RedEdge 1   (20m)     (4)
# # B6  - RedEdge 2   (20m)     (5)
# # B7  - RedEdge 3   (20m)     (6)
# # B8  - NIR         (10m)     (7)
# # B8A - RedEdge 4   (20m)     (8)
# # B9  - Water Vapor (60m)     (9)
# # B11 - SWIR 1      (20m)     (10)
# # B12 - SWIR 2      (20m)     (11)

# # red_e = ar_e[3]
# # red_k = ar_k[3]
# # red_k16 = ar_k16[3]

# # green_e = ar_e[2]
# # green_k = ar_k[2]
# # green_k16 = ar_k16[2]

# # blue_e = ar_e[1]
# # blue_k = ar_k[1]
# # blue_k16 = ar_k16[1]

# # nir_e = ar_e[7]
# # nir_k = ar_k[7]
# # nir_k16 = ar_k16[7]

# # Standardization
# red_e = standardize_array(ar_e, 3)
# red_k = standardize_array(ar_k, 3)
# #red_k16 = standardize_array(ar_k16, 3)

# green_e = standardize_array(ar_e, 2)
# green_k = standardize_array(ar_k, 2)
# #green_k16 = standardize_array(ar_k16, 2)

# blue_e = standardize_array(ar_e, 1)
# blue_k = standardize_array(ar_k, 1)
# #blue_k16 = standardize_array(ar_k16, 1)

# nir_e = standardize_array(ar_e, 7)
# nir_k = standardize_array(ar_k, 7)
# #nir_k16 = standardize_array(ar_k16, 7)



# # Plot RGB-NIR to compare... 
# #kwargs = dict(alpha=0.5, bins=255, density=True)
# kwargs = dict(alpha=0.5, bins=255)

# # Red bins
# plt.hist(red_e.flatten(), **kwargs, color='r', weights=np.ones_like(red_e.flatten()) / len(red_e.flatten()), label='Estrie 2020')
# plt.hist(red_k.flatten(), **kwargs, color='g', weights=np.ones_like(red_k.flatten()) / len(red_k.flatten()), label='Kenauk 2020')
# #plt.hist(red_k16.flatten(), **kwargs, color='b', weights=np.ones_like(red_k16.flatten()) / len(red_k16.flatten()), label='Kenauk 2016')

# plt.legend()
# plt.suptitle('Red Bands')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()


# # Red steps (curves)
# plt.hist(red_e.flatten(), **kwargs, color='r', weights=np.ones_like(red_e.flatten()) / len(red_e.flatten()), label='Estrie 2020', histtype="step")
# plt.hist(red_k.flatten(), **kwargs, color='g', weights=np.ones_like(red_k.flatten()) / len(red_k.flatten()), label='Kenauk 2020', histtype="step")
# #plt.hist(red_k16.flatten(), **kwargs, color='b', weights=np.ones_like(red_k16.flatten()) / len(red_k16.flatten()), label='Kenauk 2016', histtype="step")

# plt.legend()
# plt.suptitle('Red Bands - Steps')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Green bins
# plt.hist(green_e.flatten(), **kwargs, color='r', weights=np.ones_like(green_e.flatten()) / len(green_e.flatten()), label='Estrie 2020')
# plt.hist(green_k.flatten(), **kwargs, color='g', weights=np.ones_like(green_k.flatten()) / len(green_k.flatten()), label='Kenauk 2020')
# #plt.hist(green_k16.flatten(), **kwargs, color='b', weights=np.ones_like(green_k16.flatten()) / len(green_k16.flatten()), label='Kenauk 2016')

# plt.legend()
# plt.suptitle('Green Bands')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Green steps (cruves)
# plt.hist(green_e.flatten(), **kwargs, color='r', weights=np.ones_like(green_e.flatten()) / len(green_e.flatten()), label='Estrie 2020', histtype="step")
# plt.hist(green_k.flatten(), **kwargs, color='g', weights=np.ones_like(green_k.flatten()) / len(green_k.flatten()), label='Kenauk 2020', histtype="step")
# #plt.hist(green_k16.flatten(), **kwargs, color='b', weights=np.ones_like(green_k16.flatten()) / len(green_k16.flatten()), label='Kenauk 2016', histtype="step")

# plt.legend()
# plt.suptitle('Green Bands - Steps')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Blue
# plt.hist(blue_e.flatten(), **kwargs, color='r', weights=np.ones_like(blue_e.flatten()) / len(blue_e.flatten()), label='Estrie 2020')
# plt.hist(blue_k.flatten(), **kwargs, color='g', weights=np.ones_like(blue_k.flatten()) / len(blue_k.flatten()), label='Kenauk 2020')
# #plt.hist(blue_k16.flatten(), **kwargs, color='b', weights=np.ones_like(blue_k16.flatten()) / len(blue_k16.flatten()), label='Kenauk 2016')

# plt.legend()
# plt.suptitle('Blue Bands')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Blue steps (cruves)
# plt.hist(blue_e.flatten(), **kwargs, color='r', weights=np.ones_like(blue_e.flatten()) / len(blue_e.flatten()), label='Estrie 2020', histtype="step")
# plt.hist(blue_k.flatten(), **kwargs, color='g', weights=np.ones_like(blue_k.flatten()) / len(blue_k.flatten()), label='Kenauk 2020', histtype="step")
# #plt.hist(blue_k16.flatten(), **kwargs, color='b', weights=np.ones_like(blue_k16.flatten()) / len(blue_k16.flatten()), label='Kenauk 2016', histtype="step")

# plt.legend()
# plt.suptitle('Blue Bands - Steps')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # NIR
# plt.hist(nir_e.flatten(), **kwargs, color='r', weights=np.ones_like(nir_e.flatten()) / len(nir_e.flatten()), label='Estrie 2020')
# plt.hist(nir_k.flatten(), **kwargs, color='g', weights=np.ones_like(nir_k.flatten()) / len(nir_k.flatten()), label='Kenauk 2020')
# #plt.hist(nir_k16.flatten(), **kwargs, color='b', weights=np.ones_like(nir_k16.flatten()) / len(nir_k16.flatten()), label='Kenauk 2016')

# plt.legend()
# plt.suptitle('NIR Bands')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # NIR steps (cruves)
# plt.hist(nir_e.flatten(), **kwargs, color='r', weights=np.ones_like(nir_e.flatten()) / len(nir_e.flatten()), label='Estrie 2020', histtype="step")
# plt.hist(nir_k.flatten(), **kwargs, color='g', weights=np.ones_like(nir_k.flatten()) / len(nir_k.flatten()), label='Kenauk 2020', histtype="step")
# #plt.hist(nir_k16.flatten(), **kwargs, color='b', weights=np.ones_like(nir_k16.flatten()) / len(nir_k16.flatten()), label='Kenauk 2016', histtype="step")

# plt.legend()
# plt.suptitle('NIR Bands - Steps')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Estrie Full
# plt.hist(red_e.flatten(), **kwargs, color='r', label='Red', weights=np.ones_like(red_e.flatten()) / len(red_e.flatten()))
# plt.hist(green_e.flatten(), **kwargs, color='g', label='Green', weights=np.ones_like(green_e.flatten()) / len(green_e.flatten()))
# plt.hist(blue_e.flatten(), **kwargs, color='b', label='Blue', weights=np.ones_like(blue_e.flatten()) / len(blue_e.flatten()))
# plt.hist(nir_e.flatten(), **kwargs, color='y', label='NIR', weights=np.ones_like(nir_e.flatten()) / len(nir_e.flatten()))

# plt.legend()
# plt.suptitle('Estrie RGB-NIR')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Estrie Full - Steps
# plt.hist(red_e.flatten(), **kwargs, color='r', label='Red', weights=np.ones_like(red_e.flatten()) / len(red_e.flatten()), histtype="step")
# plt.hist(green_e.flatten(), **kwargs, color='g', label='Green', weights=np.ones_like(green_e.flatten()) / len(green_e.flatten()), histtype="step")
# plt.hist(blue_e.flatten(), **kwargs, color='b', label='Blue', weights=np.ones_like(blue_e.flatten()) / len(blue_e.flatten()), histtype="step")
# plt.hist(nir_e.flatten(), **kwargs, color='y', label='NIR', weights=np.ones_like(nir_e.flatten()) / len(nir_e.flatten()), histtype="step")

# plt.legend()
# plt.suptitle('Estrie RGB-NIR - Steps')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Kenauk 2020 full
# plt.hist(red_k.flatten(), **kwargs, color='r', label='Red', weights=np.ones_like(red_k.flatten()) / len(red_k.flatten()))
# plt.hist(green_k.flatten(), **kwargs, color='g', label='Green', weights=np.ones_like(green_k.flatten()) / len(green_k.flatten()))
# plt.hist(blue_k.flatten(), **kwargs, color='b', label='Blue', weights=np.ones_like(blue_k.flatten()) / len(blue_k.flatten()))
# plt.hist(nir_k.flatten(), **kwargs, color='y', label='NIR', weights=np.ones_like(nir_k.flatten()) / len(nir_k.flatten()))

# plt.legend()
# plt.suptitle('Kenauk 2020 - RGB-NIR')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # Kenauk 2020 full - Steps
# plt.hist(red_k.flatten(), **kwargs, color='r', label='Red', weights=np.ones_like(red_k.flatten()) / len(red_k.flatten()), histtype="step")
# plt.hist(green_k.flatten(), **kwargs, color='g', label='Green', weights=np.ones_like(green_k.flatten()) / len(green_k.flatten()), histtype="step")
# plt.hist(blue_k.flatten(), **kwargs, color='b', label='Blue', weights=np.ones_like(blue_k.flatten()) / len(blue_k.flatten()), histtype="step")
# plt.hist(nir_k.flatten(), **kwargs, color='y', label='NIR', weights=np.ones_like(nir_k.flatten()) / len(nir_k.flatten()), histtype="step")

# plt.legend()
# plt.suptitle('Kenauk 2020 - RGB-NIR - Steps')

# plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# plt.show()

# # # Kenauk 2016 full
# # plt.hist(red_k16.flatten(), **kwargs, color='r', label='Red', weights=np.ones_like(red_k16.flatten()) / len(red_k16.flatten()))
# # plt.hist(green_k16.flatten(), **kwargs, color='g', label='Green', weights=np.ones_like(green_k16.flatten()) / len(green_k16.flatten()))
# # plt.hist(blue_k16.flatten(), **kwargs, color='b', label='Blue', weights=np.ones_like(blue_k16.flatten()) / len(blue_k16.flatten()))
# # plt.hist(nir_k16.flatten(), **kwargs, color='y', label='NIR', weights=np.ones_like(nir_k16.flatten()) / len(nir_k16.flatten()))

# # plt.legend()
# # plt.suptitle('Kenauk 2016 - RGB-NIR')

# # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# # plt.show()

# # # Kenauk 2016 full - Steps
# # plt.hist(red_k16.flatten(), **kwargs, color='r', label='Red', weights=np.ones_like(red_k16.flatten()) / len(red_k16.flatten()), histtype="step")
# # plt.hist(green_k16.flatten(), **kwargs, color='g', label='Green', weights=np.ones_like(green_k16.flatten()) / len(green_k16.flatten()), histtype="step")
# # plt.hist(blue_k16.flatten(), **kwargs, color='b', label='Blue', weights=np.ones_like(blue_k16.flatten()) / len(blue_k16.flatten()), histtype="step")
# # plt.hist(nir_k16.flatten(), **kwargs, color='y', label='NIR', weights=np.ones_like(nir_k16.flatten()) / len(nir_k16.flatten()), histtype="step")

# # plt.legend()
# # plt.suptitle('Kenauk 2016 - RGB-NIR - Steps')

# # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# # plt.show()


# # Plot side-by-side
# # fig = plt.figure()
# # plt.subplot(1,3,1)
# # plt.hist(red.flatten(), **kwargs, color='r', label='Red')
# # plt.title("Input")
# # plt.subplot(1,3,2)
# # plt.hist(green.flatten(), **kwargs, color='g', label='Green')
# # plt.title("Predict")
# # plt.subplot(1,3,3)
# # plt.hist(blue.flatten(), **kwargs, color='b', label='Blue')
# # plt.title("Target")

# # plt.show()