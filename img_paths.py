from sys import platform
import os

# Check platform for paths
print("Detected platform is : ", platform)
if platform == "linux" or platform == "linux2":
    print("Using paths for Linux")
    pre_path = "/mnt/Data"
elif platform == "win32":
    print("Using paths for Windows")
    pre_path = "D:/"
else:
    "No platform detected"

# Estrie raw
e_sen2_ete = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/ete/S2_estrie_3m_ete_septembre2020.tif')
e_sen2_pri = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2/print/S2_estrie_3m_printemps_mai2020.tif') 
e_sen1_ete = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1/ete/S1_estrie_3m_ete_septembre2020.tif')
e_sen1_pri = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1/print/S1_estrie_3m_printemps_mai2020.tif') 
e_mhc      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/mhc_estrie_3m.tif')
e_slo      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/pente_estrie_3m.tif')
e_tpi      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tpi_estrie_3m.tif')
e_tri      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tri_estrie_3m.tif')
e_twi      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/twi_estrie_3m.tif')

# Kenauk raw
path_to_full_sen2_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s2_kenauk_3m_ete_HMe_STD.tif' 
path_to_full_sen2_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s2_kenauk_3m_print_HMe_STD.tif' 
path_to_full_sen1_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s1_kenauk_3m_ete_STD.tif' 
path_to_full_sen1_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s1_kenauk_3m_print_STD.tif' 
path_to_full_mhc        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mhc_kenauk_3m.tif' 
path_to_full_slopes     = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/pentes_kenauk_3m.tif' 
path_to_full_tpi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tpi_kenauk_3m.tif' 
path_to_full_tri        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif' 
path_to_full_twi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/twi_kenauk_3m.tif' 

estrie_paths_lst =  [
                    e_sen2_ete,
                    e_sen2_pri,
                    e_sen1_ete,
                    e_sen1_pri,
                    e_mhc,
                    e_slo,
                    e_tpi,
                    e_tri,
                    e_twi                    
                    ]

def get_estrie_paths():
    print("Returning : ", len(estrie_paths_lst), "paths")
    return estrie_paths_lst