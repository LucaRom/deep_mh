import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier


#training_csv_path = "/mnt/SN750/00_Donnees_SSD/csv_rf/estrie_training_full.csv"
training_csv_path = "F:/00_Donnees_SSD/csv_rf/estrie_training_full.csv"

# Load all with chunks
# df_temp = pd.read_csv(training_csv_path, chunksize=100000, iterator=True) # 116 391 936 pixels
#                                                                           #  58 195 968 (half)
#                                                                           #     100 000 (as unit reference)

# df = pd.concat(df_temp, ignore_index=True)

# Load randoms line with skiprows
nlinesfile = 116391936
nlinesrandomsample = 58195968
#nlinesrandomsample = 116391936 // 4
#nlinesrandomsample = 116391936 // 100

lines2skip = np.random.choice(np.arange(1,nlinesfile+1), (nlinesfile-nlinesrandomsample), replace=False)

print("loading data")
df = pd.read_csv(training_csv_path, skiprows=lines2skip)
print("finish loading data")

feature_names = ['s2_e_B1', 's2_e_B2', 's2_e_B3', 's2_e_B4', 's2_e_B5',
       's2_e_B6', 's2_e_B7', 's2_e_B8', 's2_e_B8a', 's2_e_B9', 's2_e_B11',
       's2_e_B12', 's2_p_B1', 's2_p_B2', 's2_p_B3', 's2_p_B4', 's2_p_B5',
       's2_p_B6', 's2_p_B7', 's2_p_B8', 's2_p_B8a', 's2_p_B9', 's2_p_B11',
       's2_p_B12', 's1_e_VH', 's1_e_VV', 's1_e_ratio', 's1_p_VH', 's1_p_VV',
       's1_p_ratio', 'mnt', 'mhc', 'slopes', 'tpi', 'tri', 'twi']

print("Calculating correlation")
#print(df.corr())
df_corr = df.corr()

# Preparing graphic 
print("Generating graphic")
plt.figure(figsize=(20,20))
#sns.heatmap(cor, annot=True, cmap=plt.cm.viridis)
sns.heatmap(df_corr, annot=True, fmt=".1f", cmap=plt.cm.viridis, annot_kws={'fontsize':5})
plt.show()

#data = df[:-2]

