import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from joblib import dump, load

import rasterio

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report

# # Full
# k_sen2_e_path = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/ete/s2_kenauk_3m_ete_aout2020.tif"
# k_sen2_p_path = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/print/S2_de_kenauk_3m_printemps_mai2020.tif"
# k_sen1_e_path = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/ete/s1_kenauk_3m_ete_aout2020.tif"
# k_sen1_p_path = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/print/S1_kenauk_3m_printemps_mai2020.tif"
# k_mnt_path    = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mnt_kenauk_3m.tif"
# k_mhc_path    = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mhc_kenauk_3m.tif"
# k_pentes_path = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/pentes_kenauk_3m.tif"
# k_tpi_path    = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tpi_kenauk_3m.tif"
# k_tri_path    = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif"
# k_twi_path    = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif"

# k_mask_b_path    = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/mask/kenauk_mask_bin_3m.tif"
# k_mask_m_path    = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/mask/kenauk_mask_multiclass_3m.tif"

# Small patch
#image_dir = "D:/00_Donnees/02_maitrise/01_trainings/kenauk/256/"
image_dir = "/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/256/"
image_nom = 'sen2_ete.48.tif'

k_sen2_e_path = os.path.join(image_dir, 'sen2_ete', image_nom)
k_sen2_p_path = os.path.join(image_dir, 'sen2_print', image_nom.replace("ete", "print"))
k_sen1_e_path = os.path.join(image_dir, 'sen1_ete', image_nom.replace("sen2_ete", "sen1_ete"))
k_sen1_p_path = os.path.join(image_dir, 'sen1_print', image_nom.replace("sen2_ete", "sen1_print"))
k_mnt_path    = os.path.join(image_dir, 'mnt', image_nom.replace("sen2_ete", "mnt"))
k_mhc_path    = os.path.join(image_dir, 'mhc', image_nom.replace("sen2_ete", "mhc"))
k_pentes_path = os.path.join(image_dir, 'pentes', image_nom.replace("sen2_ete", "pentes"))
k_tpi_path    = os.path.join(image_dir, 'tpi', image_nom.replace("sen2_ete", "tpi"))
k_tri_path    = os.path.join(image_dir, 'tri', image_nom.replace("sen2_ete", "tri"))
k_twi_path    = os.path.join(image_dir, 'twi', image_nom.replace("sen2_ete", "twi"))

k_mask_b_path    = os.path.join(image_dir, 'mask_bin', image_nom.replace("sen2_ete", "mask_bin"))
k_mask_m_path    = os.path.join(image_dir, 'mask_multiclass', image_nom.replace("sen2_ete", "mask_multiclass"))

# Load in rasterio
img_sen2_ete = rasterio.open(k_sen2_e_path)
img_sen2_print = rasterio.open(k_sen2_p_path)

img_sen1_ete = rasterio.open(k_sen1_e_path)
img_sen1_print = rasterio.open(k_sen1_p_path)

img_mnt = rasterio.open(k_mnt_path)
img_mhc = rasterio.open(k_mhc_path )
img_slopes = rasterio.open(k_pentes_path)
img_tpi = rasterio.open(k_tpi_path)
img_tri = rasterio.open(k_tri_path)
img_twi = rasterio.open(k_twi_path)

img_mask_bin = rasterio.open(k_mask_b_path)
img_mask_multi_c = rasterio.open(k_mask_m_path)

# Arrays (keep them coming...)
# Sentinel-2 ete
#sen2_e_bd1 = img_sen2_ete.read(1)    # s2_e_B1
sen2_e_bd2 = img_sen2_ete.read(2)    # s2_e_B2
sen2_e_bd3 = img_sen2_ete.read(3)    # s2_e_B3
sen2_e_bd4 = img_sen2_ete.read(4)    # s2_e_B4
sen2_e_bd5 = img_sen2_ete.read(5)    # s2_e_B5
sen2_e_bd6 = img_sen2_ete.read(6)    # s2_e_B6
sen2_e_bd7 = img_sen2_ete.read(7)    # s2_e_B7
sen2_e_bd8 = img_sen2_ete.read(8)    # s2_e_B8
sen2_e_bd9 = img_sen2_ete.read(9)    # s2_e_B8a
#sen2_e_bd10 = img_sen2_ete.read(10)  # s2_e_B9
sen2_e_bd11 = img_sen2_ete.read(11)  # s2_e_B11
sen2_e_bd12 = img_sen2_ete.read(12)  # s2_e_B12

# Sentinel-2 print
#sen2_p_bd1 = img_sen2_print.read(1)
sen2_p_bd2 = img_sen2_print.read(2)
sen2_p_bd3 = img_sen2_print.read(3)
sen2_p_bd4 = img_sen2_print.read(4)
sen2_p_bd5 = img_sen2_print.read(5)
sen2_p_bd6 = img_sen2_print.read(6)
sen2_p_bd7 = img_sen2_print.read(7)
sen2_p_bd8 = img_sen2_print.read(8)
sen2_p_bd9 = img_sen2_print.read(9)
#sen2_p_bd10 = img_sen2_print.read(10)
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

# Arrays (keep them coming...)
# Sentinel-2 ete
#sen2_e_bd1 = img_sen2_ete.read(1)    # s2_e_B1
sen2_e_bd2 = img_sen2_ete.read(2)    # s2_e_B2
sen2_e_bd3 = img_sen2_ete.read(3)    # s2_e_B3
sen2_e_bd4 = img_sen2_ete.read(4)    # s2_e_B4
sen2_e_bd5 = img_sen2_ete.read(5)    # s2_e_B5
sen2_e_bd6 = img_sen2_ete.read(6)    # s2_e_B6
sen2_e_bd7 = img_sen2_ete.read(7)    # s2_e_B7
sen2_e_bd8 = img_sen2_ete.read(8)    # s2_e_B8
sen2_e_bd9 = img_sen2_ete.read(9)    # s2_e_B8a
#sen2_e_bd10 = img_sen2_ete.read(10)  # s2_e_B9
sen2_e_bd11 = img_sen2_ete.read(11)  # s2_e_B11
sen2_e_bd12 = img_sen2_ete.read(12)  # s2_e_B12

# Closing party
img_sen2_print.close()
img_sen2_ete.close()
img_sen1_ete.close()
img_sen1_print.close()
img_mnt.close()
img_mhc.close()
img_slopes.close()
img_tpi.close()
img_tri.close()
img_twi.close()
img_mask_bin.close()
img_mask_multi_c.close()

# Feed dataframe
df2 = pd.DataFrame()

#sen 2
#df2['s2_e_B1'] = sen2_e_bd1.flatten()
df2['s2_e_B2'] = sen2_e_bd2.flatten()
df2['s2_e_B3'] = sen2_e_bd3.flatten()
df2['s2_e_B4'] = sen2_e_bd4.flatten()
df2['s2_e_B5'] = sen2_e_bd5.flatten()
df2['s2_e_B6'] = sen2_e_bd6.flatten()
df2['s2_e_B7'] = sen2_e_bd7.flatten()
df2['s2_e_B8'] = sen2_e_bd8.flatten()
df2['s2_e_B8a'] = sen2_e_bd9.flatten()
#df2['s2_e_B9'] = sen2_e_bd10.flatten()
df2['s2_e_B11'] = sen2_e_bd11.flatten()
df2['s2_e_B12'] = sen2_e_bd12.flatten()

#df2['s2_p_B1'] = sen2_p_bd1.flatten()
df2['s2_p_B2'] = sen2_p_bd2.flatten()
df2['s2_p_B3'] = sen2_p_bd3.flatten()
df2['s2_p_B4'] = sen2_p_bd4.flatten()
df2['s2_p_B5'] = sen2_p_bd5.flatten()
df2['s2_p_B6'] = sen2_p_bd6.flatten()
df2['s2_p_B7'] = sen2_p_bd7.flatten()
df2['s2_p_B8'] = sen2_p_bd8.flatten()
df2['s2_p_B8a'] = sen2_p_bd9.flatten()
#df2['s2_p_B9'] = sen2_p_bd10.flatten()
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


# image = np.dstack((sen2_e_bd2, sen2_e_bd3, sen2_e_bd4, sen2_e_bd5, sen2_e_bd6,
#                    sen2_e_bd7, sen2_e_bd8, sen2_e_bd9, sen2_e_bd11, sen2_e_bd12,
#                    sen2_p_bd2, sen2_p_bd3, sen2_p_bd4, sen2_p_bd5, sen2_p_bd6,
#                    sen2_p_bd7, sen2_p_bd8, sen2_p_bd9, sen2_p_bd11, sen2_p_bd12,
#                    sen1_e_bd1, 
                                    
#                     ))


#training_csv_path = "/mnt/SN750/00_Donnees_SSD/csv_rf/estrie_training_full.csv"
training_csv_path = "F:/00_Donnees_SSD/csv_rf/estrie_training_full.csv"

# Load all with chunks
# df_temp = pd.read_csv(training_csv_path, chunksize=100000, iterator=True) # 116 391 936 pixels
#                                                                           #  58 195 968 (half)
#                                                                           #     100 000 (as unit reference)

# df = pd.concat(df_temp, ignore_index=True)

# Load randoms line with skiprows
nlinesfile = 116391936
#nlinesrandomsample = 58195968
#nlinesrandomsample = 116391936 // 4
#nlinesrandomsample = 116391936 // 5
#nlinesrandomsample = 116391936 // 10
nlinesrandomsample = 116391936 // 100
#nlinesrandomsample = 116391936 // 100000

lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False)

print("loading data")
#df = pd.read_csv(training_csv_path, skiprows=lines2skip)
df = df2
print("finish loading data")

feature_names = ['s2_e_B1', 's2_e_B2', 's2_e_B3', 's2_e_B4', 's2_e_B5',
       's2_e_B6', 's2_e_B7', 's2_e_B8', 's2_e_B8a', 's2_e_B9', 's2_e_B11',
       's2_e_B12', 's2_p_B1', 's2_p_B2', 's2_p_B3', 's2_p_B4', 's2_p_B5',
       's2_p_B6', 's2_p_B7', 's2_p_B8', 's2_p_B8a', 's2_p_B9', 's2_p_B11',
       's2_p_B12', 's1_e_VH', 's1_e_VV', 's1_e_ratio', 's1_p_VH', 's1_p_VV',
       's1_p_ratio', 'mnt', 'mhc', 'slopes', 'tpi', 'tri', 'twi']

selected_feature_names = ['s2_e_B2', 's2_e_B3', 's2_e_B4', 's2_e_B5',
       's2_e_B6', 's2_e_B7', 's2_e_B8', 's2_e_B8a', 's2_e_B11',
       's2_e_B12', 's2_p_B2', 's2_p_B3', 's2_p_B4', 's2_p_B5',
       's2_p_B6', 's2_p_B7', 's2_p_B8', 's2_p_B8a', 's2_p_B11',
       's2_p_B12', 's1_e_VH', 's1_e_VV', 's1_e_ratio', 's1_p_VH', 's1_p_VV',
       's1_p_ratio', 'mnt', 'mhc', 'slopes', 'tpi', 'tri', 'twi']

# TODO seperate fonction for
#print("Calculating correlation")
#print(df.corr())
#df_corr = df.corr()

# Preparing graphic 
# print("Generating graphic")
# plt.figure(figsize=(20,20))
# #sns.heatmap(cor, annot=True, cmap=plt.cm.viridis)
# sns.heatmap(df_corr, annot=True, fmt=".1f", cmap=plt.cm.viridis, annot_kws={'fontsize':5})
# plt.show()

# Model for Random Forest

print("Creating data and labels")
data = df[selected_feature_names]
label = df['mask_multi']
print("Deleting original DF")
del df
print("Done creating label and data")

# Split dataset into training set and test set
print("Creating trianing and test sets")
#X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
print("Deleting data and label variables")
# del data
# del label
print("done")


param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_features": ['auto', 'sqrt'],
        "max_depth": [None, 2, 6, 10]
    }

clf = RandomForestClassifier(n_estimators=1000, max_features='sqrt', max_depth=None, n_jobs=-1, verbose=1)

# CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy',
#                      return_train_score=True)
# CV_clf.fit(X_train, y_train)
# print("Best params:", CV_clf.best_params_) 

#set classifier parameters and train classifier
# print("Start RF fit")
# clf = RandomForestClassifier(n_estimators=100,n_jobs=-1, verbose=1) #n_jobs=-1 prend tous les threads possible
# clf.fit(X_train, y_train)
# print("Done")



clf = load('grid_optimised_pixel.joblib') 

#clf = CV_clf.best_estimator_
#dump(clf, 'grid_optimised_pixel.joblib') 

print("Starting prediction on test dataset")
#y_pred = clf.predict(X_test)
y_pred = clf.predict(data)
print("Done")

print("Generating CM et accuracy")
#conf_mat = confusion_matrix(y_test, y_pred)

# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print(classification_report(y_test,y_pred))
#print(conf_mat)

print("Accuracy:", metrics.accuracy_score(label, y_pred))
print(classification_report(label,y_pred))

print("Plotting CM")

# disp = plot_confusion_matrix(clf, X_test, y_test,
#                             cmap=plt.cm.Blues,
#                             values_format='d')

disp = plot_confusion_matrix(clf, data, label,
                            cmap=plt.cm.Blues,
                            values_format='d')

plt.xlabel('Prédit')
plt.ylabel('Réel')

#importances = clf.feature_importances_
#std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

# forest_importances = pd.Series(importances, index=selected_feature_names)
# forest_importances = forest_importances.sort_values(ascending=True)

# fig, ax = plt.subplots()
# forest_importances.plot.barh(ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()

plt.show()

# Exporter la prédiction
