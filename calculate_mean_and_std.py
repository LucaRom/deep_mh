import rasterio
import numpy as np

def mean_std_solo(path_image):
    dataset = rasterio.open(path_image)
    array = dataset.read()

    #count = 1
    mean_lst = []
    std_lst = []
    for band in array:
        band_flat = band.flatten()
        mean = np.average(band_flat)
        std = np.std(band_flat)
        mean_lst.append(mean)
        std_lst.append(std)

    print("List of means : ")
    print(mean_lst)
    print("List of std: ")
    print(std_lst)

#mean_std_solo(k_image_ete)

def mean_std_pair(path_image1, path_image2):
    im1 = rasterio.open(path_image1)
    array1 = im1.read()

    im2 = rasterio.open(path_image2)
    array2 = im2.read()

    count = 1
    mean_lst = []
    std_lst = []

    if len(array1) == len(array2):
        for x in range(len(array1)):
            band_im1_lst = array1[x].flatten()
            band_im2_lst = array2[x].flatten()
            print("finish cast to list")

            combined_array = np.concatenate((band_im1_lst, band_im2_lst))
            print("combine finish")

            mean = np.average(combined_array)
            print("mean finish")

            std = np.std(combined_array)
            print("std finish")

            mean_lst.append(mean)
            std_lst.append(std)
            print("appends finish")

            print(count)
            count += 1

    print(mean_lst)
    print(std_lst)

# Calculated std and means for standardization
# 12 bands estrie
estrie_mean_e = [259.971087045696, 277.3490067676725, 520.4650232890134, 342.23574780553645, 906.7611488412249, 2656.3582951694643, 3203.3543093369944, 3389.6250611778078, 3487.079600166239, 3555.416409200909, 1714.2260907527316, 828.2768740555728]
estrie_std_e  = [525.5551122108338, 526.4768589585602, 515.8903727938966, 527.3656790023017, 561.5222503677404, 836.1454714836563, 984.9190349745415, 1067.0420278801334, 1026.7569263359944, 1066.123618103052, 630.0584359871733, 505.2076063419134]
estrie_mean_p = [457.4229830346009, 501.79759875320303, 694.4711397083421, 835.1158882308216, 1219.9447441650816, 1823.0661322180392, 2064.6505317461747, 2316.1887302003915, 2363.5869859139643, 2359.4662122932396, 2390.6124116260303, 1586.6126304451745]
estrie_std_p  = [169.44646075504082, 249.03030944938908, 293.96819726121373, 408.20429488371605, 392.1811051266158, 492.36521601358254, 550.8773405439316, 623.9017038640061, 590.0457818993959, 540.556974947324, 740.4564895487368, 581.7629650224691]

estrie_s1_mean_e = [-15.479797, -9.211855, 6.267961]
estrie_s1_std_e  = [1.622046, 1.8651232, 1.2285297]
estrie_s1_mean_p = [-15.0310545, -9.519093, 5.5120163]
estrie_s1_std_p  = [2.1044014, 1.9065734, 1.37706]

s1_e_p_mean = [-15.479797, -9.211855, 6.267961, -15.0310545, -9.519093, 5.5120163]
s1_e_p_std  = [1.622046, 1.8651232, 1.2285297, 2.1044014, 1.9065734, 1.37706] 

estrie_mhc_mean = [7.798849]
estrie_mhc_std  = [7.033332]
estrie_slope_mean = [5.5523205]
estrie_slope_std  = [5.196636]
estrie_tpi_mean = [0.0029951811]
estrie_tpi_std  = [1.0641352]
estrie_tri_mean = [0.06429929]
estrie_tri_std  = [0.06102526]
estrie_twi_mean = [6.7409873]
estrie_twi_std  = [3.182435]

estrie_lidar_mean = [7.798849, 5.5523205, 0.0029951811, 0.06429929, 6.7409873]
estrie_lidar_std  = [7.033332, 5.196636, 1.0641352, 0.06102526, 3.182435]

# 12 bands kenauk 2016
kenauk_2016_mean_e = [396.8236398831345, 340.8023674183238, 564.6683101976018, 316.86383555478693, 899.6916374891973, 3054.401784107393, 3726.6279551992707, 3804.23744469375, 3982.4497302888194, 3999.5862881874696, 1907.939020030273, 842.7017527775387]
kenauk_2016_std_e  = [34.77012868618949, 53.682757226651596, 101.81933949389413, 86.81071558339808, 223.97258987335917, 930.740342457327, 1192.9850511650027, 1264.8917375808307, 1295.2198436031365, 1193.0749293048152, 634.0964063702438, 290.9939493314744]
kenauk_2016_mean_p = [219.68819396631102, 210.2553309489344, 560.2232254622481, 233.54357997780932, 1026.3671730690598, 3382.8830539876685, 3978.7236678620457, 4108.692913817842, 4198.2121516410125, 4205.404193000436, 1876.4746520185688, 876.4445310976309]
kenauk_2016_std_p  = [63.01628839401353, 82.78241512687786, 171.82724437360412, 125.10823529913228, 325.63969711949863, 1214.946174826992, 1466.1660470322224, 1538.541878367054, 1523.519205499712, 1461.2253545375243, 618.8799298527035, 294.7675848153584]

# 12 bands kenauk + estrie
kenauk_estrie_mean_e = [298.9376736842285, 295.4163414577072, 533.0512058140661, 335.011499544136, 904.7482179833103, 2769.694844441057, 3352.3481529620663, 3507.679339513949, 3628.128361722713, 3681.886714013677, 1769.3827647437795, 832.3841287235765]
kenauk_estrie_std_e  = [449.1333965424019, 447.09781235125655, 440.1281124724915, 448.5562977082119, 489.71626393358594, 882.6080290302698, 1074.64297530801, 1142.3471873971982, 1132.1216994695542, 1121.812465173574, 637.2357446197225, 454.657718157629]
kenauk_estrie_mean_p = [389.73178500319295, 418.7855774087138, 656.2461824601817, 663.8277464836387, 1164.82661126222, 2267.199178786665, 2609.65239794843, 2826.5757717474403, 2885.967306830988, 2885.0676772511297, 2244.2198662295045, 1384.4035653061455]
kenauk_estrie_std_p  = [182.1481184415827, 252.22961441363324, 271.82436188432376, 444.23478736031086, 384.49626301643093, 1043.6492154677742, 1255.1022957221837, 1267.5933253040087, 1263.0965697948964, 1229.1994225452902, 745.0200230290296, 607.8933293590522]


if __name__ == "__main__":

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

    k_lidar_means = [13.348262, 13.45669, -0.006740755, -3.689763, 5.7766604]
    k_lidar_stds  = [7.7406297, 13.942361, 1.3129127, 241.4134, 5.6496654]

    mean_std_solo(k_twi)