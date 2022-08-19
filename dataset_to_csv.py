import numpy as np
import os
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


csv_path_out = "/mnt/SN750/00_Donnees_SSD/csv_rf"

train_path = "/mnt/SN750/00_Donnees_SSD/256"

img_lst = [x for x in os.listdir(os.path.join(train_path, 'sen2_ete')) if x.endswith('.tif')]

df = pd.DataFrame()

#for img in tqdm(img_lst[:10]): # For shorter test
for img in tqdm(img_lst):
    # sentinel 2 images
    sen2_ete_path = os.path.join(train_path, 'sen2_ete', img)
    sen2_print_path = os.path.join(train_path, 'sen2_print', img.replace("ete", "print"))

    # sentinel-1 images
    sen1_ete_path = os.path.join(train_path, 'sen1_ete', img).replace("sen2_ete", "sen1_ete")
    sen1_print_path = os.path.join(train_path, 'sen1_print', img.replace("sen2_ete", "sen1_print"))

    # lidar images
    mnt_path = os.path.join(train_path, 'mnt', img.replace("sen2_ete", "mnt"))
    mhc_path = os.path.join(train_path, 'mhc', img.replace("sen2_ete", "mhc"))
    slopes_path = os.path.join(train_path, 'pentes', img.replace("sen2_ete", "pentes"))
    tpi_path = os.path.join(train_path, 'tpi', img.replace("sen2_ete", "tpi"))
    tri_path = os.path.join(train_path, 'tri', img.replace("sen2_ete", "tri"))
    twi_path = os.path.join(train_path, 'twi', img.replace("sen2_ete", "twi"))

    # mask path
    mask_bin_path = os.path.join(train_path, 'mask_bin', img.replace("sen2_ete", "mask_bin"))
    mask_multi_c_path = os.path.join(train_path, 'mask_multiclass', img.replace("sen2_ete", "mask_multiclass"))
    
    # Load in rasterio
    img_sen2_ete = rasterio.open(sen2_ete_path)
    img_sen2_print = rasterio.open(sen2_print_path)

    img_sen1_ete = rasterio.open(sen1_ete_path)
    img_sen1_print = rasterio.open(sen1_print_path)

    img_mnt = rasterio.open(mnt_path)
    img_mhc = rasterio.open(mhc_path)
    img_slopes = rasterio.open(slopes_path)
    img_tpi = rasterio.open(tpi_path)
    img_tri = rasterio.open(tri_path)
    img_twi = rasterio.open(twi_path)

    img_mask_bin = rasterio.open(mask_bin_path)
    img_mask_multi_c = rasterio.open(mask_multi_c_path)

    #print('stop... believing...')

    # Arrays (keep them coming...)
    # Sentinel-2 ete
    sen2_e_bd1 = img_sen2_ete.read(1)
    sen2_e_bd2 = img_sen2_ete.read(2)
    sen2_e_bd3 = img_sen2_ete.read(3)
    sen2_e_bd4 = img_sen2_ete.read(4)
    sen2_e_bd5 = img_sen2_ete.read(5)
    sen2_e_bd6 = img_sen2_ete.read(6)
    sen2_e_bd7 = img_sen2_ete.read(7)
    sen2_e_bd8 = img_sen2_ete.read(8)
    sen2_e_bd9 = img_sen2_ete.read(9)
    sen2_e_bd10 = img_sen2_ete.read(10)
    sen2_e_bd11 = img_sen2_ete.read(11)
    sen2_e_bd12 = img_sen2_ete.read(12)

    # Sentinel-2 print
    sen2_p_bd1 = img_sen2_print.read(1)
    sen2_p_bd2 = img_sen2_print.read(2)
    sen2_p_bd3 = img_sen2_print.read(3)
    sen2_p_bd4 = img_sen2_print.read(4)
    sen2_p_bd5 = img_sen2_print.read(5)
    sen2_p_bd6 = img_sen2_print.read(6)
    sen2_p_bd7 = img_sen2_print.read(7)
    sen2_p_bd8 = img_sen2_print.read(8)
    sen2_p_bd9 = img_sen2_print.read(9)
    sen2_p_bd10 = img_sen2_print.read(10)
    sen2_p_bd11 = img_sen2_print.read(11)
    sen2_p_bd12 = img_sen2_print.read(12)

    # Sentinel-1 ete
    sen1_e_bd1 = img_sen1_ete.read(1)
    sen1_e_bd2 = img_sen1_ete.read(2)
    sen1_e_bd3 = img_sen1_ete.read(3)

    # Sentinel-1 print
    sen1_p_bd1 = img_sen1_print.read(1)
    sen1_p_bd2 = img_sen1_print.read(2)
    sen1_p_bd3 = img_sen1_print.read(3)

    # LiDAR
    mnt_bd1 = img_mnt.read(1)
    mhc_bd1 = img_mhc.read(1)
    slopes_bd1 = img_slopes.read(1)
    tpi_bd1 = img_tpi.read(1)
    tri_bd1 = img_tri.read(1)
    twi_bd1 = img_twi.read(1)

    # Mask (label)
    img_mask_bin_bd1 = img_mask_bin.read(1)
    img_mask_class_bd1 = img_mask_multi_c.read(1)

    #TODO Normaliser sentinel

    # Feed dataframe in loop
    df2 = pd.DataFrame()
    
    #sen 2
    df2['s2_e_B1'] = sen2_e_bd1.flatten()
    df2['s2_e_B2'] = sen2_e_bd2.flatten()
    df2['s2_e_B3'] = sen2_e_bd3.flatten()
    df2['s2_e_B4'] = sen2_e_bd4.flatten()
    df2['s2_e_B5'] = sen2_e_bd5.flatten()
    df2['s2_e_B6'] = sen2_e_bd6.flatten()
    df2['s2_e_B7'] = sen2_e_bd7.flatten()
    df2['s2_e_B8'] = sen2_e_bd8.flatten()
    df2['s2_e_B8a'] = sen2_e_bd9.flatten()
    df2['s2_e_B9'] = sen2_e_bd10.flatten()
    df2['s2_e_B11'] = sen2_e_bd11.flatten()
    df2['s2_e_B12'] = sen2_e_bd12.flatten()

    df2['s2_p_B1'] = sen2_p_bd1.flatten()
    df2['s2_p_B2'] = sen2_p_bd2.flatten()
    df2['s2_p_B3'] = sen2_p_bd3.flatten()
    df2['s2_p_B4'] = sen2_p_bd4.flatten()
    df2['s2_p_B5'] = sen2_p_bd5.flatten()
    df2['s2_p_B6'] = sen2_p_bd6.flatten()
    df2['s2_p_B7'] = sen2_p_bd7.flatten()
    df2['s2_p_B8'] = sen2_p_bd8.flatten()
    df2['s2_p_B8a'] = sen2_p_bd9.flatten()
    df2['s2_p_B9'] = sen2_p_bd10.flatten()
    df2['s2_p_B11'] = sen2_p_bd11.flatten()
    df2['s2_p_B12'] = sen2_p_bd12.flatten()

    # sen 1
    df2['s1_e_VH'] = sen1_e_bd1.flatten()
    df2['s1_e_VV'] = sen1_e_bd2.flatten()
    df2['s1_e_ratio'] = sen1_e_bd3.flatten()

    df2['s1_p_VH'] = sen1_p_bd1.flatten()
    df2['s1_p_VV'] = sen1_p_bd2.flatten()
    df2['s1_p_ratio'] = sen1_p_bd3.flatten()

    # LiDAR
    df2['mnt'] = mnt_bd1.flatten()
    df2['mhc'] = mhc_bd1.flatten()
    df2['slopes'] = slopes_bd1.flatten()
    df2['tpi'] = tpi_bd1.flatten()
    df2['tri'] = tri_bd1.flatten()
    df2['twi'] = twi_bd1.flatten()

    # Mask (label)
    df2['mask_bin'] = img_mask_bin_bd1.flatten()
    df2['mask_multi'] = img_mask_class_bd1.flatten()

    # Feed main dataframe out loop
    #df = df.append(df2) # deprecated warning

    df = pd.concat([df, df2], ignore_index=True)

    #print('stop... believing...')

#print('stop... believing...')

df.to_csv(os.path.join(csv_path_out, 'test.csv'))

print('stop... believing...')

#if __name__ == "__main__":



