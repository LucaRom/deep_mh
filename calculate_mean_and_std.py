import numpy as np
import os
import rasterio
import time

import img_paths

def mean_std_solo(path_image):
    dataset = rasterio.open(path_image)
    array = dataset.read()

    #count = 1
    mean_lst = []
    std_lst = []
    num_bands = len(array)

    for band in array:
        band_flat = band.flatten()
        mean = np.average(band_flat)
        std = np.std(band_flat)
        mean_lst.append(mean)
        std_lst.append(std)

    print("Number of bands : ", num_bands)
    print("List of means : ")
    print(mean_lst)
    print("List of std: ")
    print(std_lst)

    return mean_lst, std_lst, num_bands

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

k_lidar_means = [13.348262, 13.45669, -0.006740755, -3.689763, 5.7766604]
k_lidar_stds  = [7.7406297, 13.942361, 1.3129127, 241.4134, 5.6496654]

if __name__ == "__main__":

    paths_lst = img_paths.get_estrie_raw_paths(output_mode='list')

    lidar_means_array = []
    lidar_stdev_array = []
    for path in paths_lst:
        mean_lst, std_lst, num_bands = mean_std_solo(path_image=path)
        sensors_name = path.split("/")[-2]
        current_region = path.split("/")[-4]

        if sensors_name == 'lidar':
            # lidar_prefix = path.split('/')[-1][:3]
            # output_name = os.path.join('stats/', current_region, current_region + "_" + sensors_name + "_" + lidar_prefix)
            lidar_means_array.append(mean_lst[0])
            lidar_stdev_array.append(std_lst[0])
        else:
            output_name = os.path.join('stats/', current_region, current_region + "_" + sensors_name)

            print('Saving means for ', sensors_name)
            np.save(output_name + '_means', mean_lst)
            
            print('Saving stdevs for ', sensors_name)
            np.save(output_name + '_stds', std_lst)

    #current_region = path.split("/")[-4]
    current_region = 'estrie'
    lidar_output_name = os.path.join('stats/', current_region, current_region + "_lidar")
    print('Saving means for LiDAR')
    np.save(lidar_output_name + '_means', lidar_means_array )
    
    print('Saving stdevs for LIDAR')
    np.save(lidar_output_name + '_stds', lidar_stdev_array)

    #print(np.load('stats/estrie/estrie_lidar_means.npy'))