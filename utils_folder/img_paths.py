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

#TODO COnfig file with datasets

##########################
# RAW IMAGES INPUT PATHS #
##########################

# Estrie raw
e_sen2_ete = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2_ete/S2_estrie_3m_ete_septembre2020.tif')
e_sen2_pri = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen2_pri/S2_estrie_3m_printemps_mai2020.tif') 
e_sen1_ete = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1_ete/S1_estrie_3m_ete_septembre2020.tif')
e_sen1_pri = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/sen1_pri/S1_estrie_3m_printemps_mai2020.tif') 
e_mhc      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/mhc_estrie_3m.tif')
e_slo      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/pente_estrie_3m.tif')
e_tpi      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tpi_estrie_3m.tif')
e_tri      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/tri_estrie_3m.tif')
e_twi      = os.path.join(pre_path, '00_Donnees/02_maitrise/01_trainings/estrie/processed_raw/lidar/twi_estrie_3m.tif')

# Portneuf zone2 raw
p2_sen2_ete = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/sen2_ete/s2_3m_portneuf_ete_2018.tif')
p2_sen2_pri = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/sen2_pri/s2_3m_portneuf_prin_2018.tif') 
p2_sen1_ete = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/sen1_ete/s1_3m_portneuf_ete_2018.tif')
p2_sen1_pri = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/sen1_pri/s1_3m_portneuf_prin_2018.tif') 
p2_mhc      = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/lidar/mhc_3m_portneuf.tif')
p2_slo      = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/lidar/slo_3m_portneuf.tif')
p2_tpi      = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/lidar/tpi_3m_portneuf.tif')
p2_tri      = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/lidar/tri_3m_portneuf.tif')
p2_twi      = os.path.join(pre_path, '/mnt/Data/00_Donnees/02_maitrise/01_trainings/portneuf_zone2/processed_raw/lidar/twi_3m_portneuf.tif')

############################
# PROCESSED RAW INPUT PATH #
############################

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

############################
# TILED IMAGES INPUT PATHS #
############################

# TODO REMAKE PATHS TO CONSIDER that e_img_dir is a root path used in UTILS.PY
# Estrie 256 raw with 0% overlap
e_img_dir        = "/mnt/SN750/00_Donnees_SSD/256/"
e_mask_bin_dir   = "/mnt/SN750/00_Donnees_SSD/256/mask_bin"
e_mask_multi_dir = "/mnt/SN750/00_Donnees_SSD/256/mask_multiclass"
e_lidar_dir      = "/mnt/SN750/00_Donnees_SSD/256/mnt"

# Estrie 256 raw with 50% overlap
e_50p_img_dir        = "/mnt/SN750/00_Donnees_SSD/256_over50p/"
e_50p_mask_bin_dir   = "/mnt/SN750/00_Donnees_SSD/256_over50p/mask_bin"
e_50p_mask_multi_dir = "/mnt/SN750/00_Donnees_SSD/256_over50p/mask_multiclass"
e_50p_lidar_dir      = "/mnt/SN750/00_Donnees_SSD/256_over50p/mnt"

##########################
# RAW IMAGES PATHS LISTS #
##########################

# Raw estrie paths lst
estrie_raw_paths_lst =  [
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

# Raw estrie paths lst
portneuf_z2_raw_paths_lst =  [
                             p2_sen2_ete,
                             p2_sen2_pri,
                             p2_sen1_ete,
                             p2_sen1_pri,
                             p2_mhc,
                             p2_slo,
                             p2_tpi,
                             p2_tri,
                             p2_twi                    
                             ]

############################
# TILED IMAGES PATHS LISTS #
############################

# TODO Put all paths and put list of paths as arguments in datasets
# instead of loading all paths in dataset?

# Estrie - 256 X 256 raw tiles with 50% overlap
estrie_256over0p_paths_lst =  [
                                e_img_dir,
                                e_mask_bin_dir,
                                e_mask_multi_dir,
                                e_lidar_dir,                  
                                ]

# Estrie - 256 X 256 raw tiles with 50% overlap
estrie_256over50p_paths_lst =  [
                                e_50p_img_dir,
                                e_50p_mask_bin_dir,
                                e_50p_mask_multi_dir,
                                e_50p_lidar_dir,                  
                                ]
#############
# Functions # 
#############

def get_estrie_raw_paths(output_mode=list):
    if output_mode == 'list': 
        print("Returning list of ", len(estrie_raw_paths_lst), "paths")
        return estrie_raw_paths_lst
    elif output_mode == 'return': 
        print("Returning paths seperatly")
        return e_sen2_ete, e_sen2_pri, e_sen1_ete, e_sen1_pri, e_mhc, e_slo, e_tpi, e_tri, e_twi
    else:
        print("Wrong argument value for 'output_mode'. Value received is : ", output_mode)

def get_estrie_tiles_paths(output_mode=list, overlap=True):
    if output_mode == 'list': 
        print("Returning list of ", len(estrie_raw_paths_lst), "paths")
        return estrie_raw_paths_lst
    elif output_mode == 'return': 
        print("Returning paths seperatly")
        return e_sen2_ete, e_sen2_pri, e_sen1_ete, e_sen1_pri, e_mhc, e_slo, e_tpi, e_tri, e_twi
    else:
        print("Wrong argument value for 'output_mode'. Value received is : ", output_mode)

def get_portneuf_z2_raw_paths(output_mode=list):
    if output_mode == 'list': 
        print("Returning list of ", len(portneuf_z2_raw_paths_lst), "paths")
        return portneuf_z2_raw_paths_lst
    elif output_mode == 'return': 
        print("Returning paths seperatly")
        return p2_sen2_ete, p2_sen2_pri, p2_sen1_ete, p2_sen1_pri, p2_mhc, p2_slo, p2_tpi, p2_tri, p2_twi
    else:
        print("Wrong argument value for 'output_mode'. Value received is : ", output_mode)