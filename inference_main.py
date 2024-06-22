'''
#TODO Demander si on peut le mettre en ligne (si besoin bien sûr) et comment bien mettre les droits d'auteurs

Adapted from original script author : Marc-Antoine Genest (Cerfo)
''' 

import os

#from unet_3enco_sum import unet_3enco_sum
import tifffile as tiff
import rasterio
from rasterio import windows
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import tifffile
#import tensorflow as tf
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from seg_3_enco_multi import SemSegment
from omegaconf import DictConfig, OmegaConf

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# Output
#save_path = '/mnt/SN750/01_Code_nvme/Unet_lightning/results/inference/'
save_path = '/mnt/SN750/01_Code_nvme/Master_2024/results/inference/'

# Models dictionnary
all_bands = [x for x in range(12)]

models = {
    'm27': {'chkp_path': 'lightning_logs/version_27/checkpoints/epoch=197-step=189882.ckpt',
            'num_layers': 4, 
            'opt_bands': all_bands, 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': ['ndvi']},
    'm150': {'chkp_path': 'lightning_logs/version_150/checkpoints/epoch=23-step=25584.ckpt',
            'num_layers': 3, 
            'opt_bands': all_bands, 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': ['ndvi', 'ndwi', 'ndmi', 'mbwi']},
    'm187': {'chkp_path': 'lightning_logs/version_187/checkpoints/epoch=27-step=28252.ckpt',
            'num_layers': 4, 
            'opt_bands':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11], 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': []},
    'm210': {'chkp_path': 'lightning_logs/version_210/checkpoints/epoch=17-step=18162.ckpt',
            'num_layers': 3, 
            'opt_bands':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11], 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': ['ndvi']},
    'm214': {'chkp_path': 'lightning_logs/version_214/checkpoints/epoch=41-step=42378.ckpt',
            'num_layers': 4, 
            'opt_bands':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11], 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': ['ndvi']},
    'm214_last': {'chkp_path': 'lightning_logs/version_214/checkpoints/last.ckpt',
            'num_layers': 4, 
            'opt_bands':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11], 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': ['ndvi']},
    'm222': {'chkp_path': 'lightning_logs/version_222/checkpoints/epoch=193-step=176152.ckpt',
            'num_layers': 4, 
            'opt_bands':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11], 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': ['ndvi']},
    'm124': {
            'chkp_path': '/mnt/f/01_Code_nvme/Master_2024/lightning_logs/version_124/checkpoints/topk_epoch-epoch=199-val_loss=0.27.ckpt',
            'num_layers': 4,
            'opt_bands': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'],
            'rad_bands': [0,1,2],
            'indices_lst': ['ndvi'],
            'architecture': 'unet_3enco_concat_attention',
            'model': 'model_multi'
            },
}

selected_model = 'm124'
params = models[selected_model]
region_preds = 'estrie' # 'estrie', 'portneuf_sud', 'portneuf_nord'
region_model = 'estrie_delete_me_please_test_my_PP'
model_name = 'region_preds_{}_model_{}_{}'.format(region_preds, selected_model, region_model)

# if region_preds == 'estrie':
#     img_path = '/mnt/SN750/01_Code_nvme/Unet_lightning/results/full_stack/stack_estrie_3m_test_v04.tif'
# elif region_preds == 'portneuf_sud':
#     img_path = '/mnt/SN750/01_Code_nvme/Unet_lightning/results/full_stack/stack_portneuf_3m_zone2.tif'
# elif region_preds == 'portneuf_nord':
#     img_path = '/mnt/d/00_Donnees/02_maitrise/01_trainings/portneuf_zone1/portneuf_nord_stack_inference.tif'

if region_preds == 'estrie':
    img_path = '/mnt/f/01_Code_nvme/Master_2024/results/full_stack/stack_estrie_3m_test_v04.tif'
elif region_preds == 'portneuf_sud':
    img_path = '/mnt/SN750/01_Code_nvme/Unet_lightning/results/full_stack/stack_portneuf_3m_zone2.tif' # Enregistrer les anciens dans oneNote au cas
elif region_preds == 'portneuf_nord':
    img_path = '/mnt/d/00_Donnees/02_maitrise/01_trainings/portneuf_zone1/portneuf_nord_stack_inference.tif'


# Model parameters (load automatically from checkpoint)
num_layers_main = params['num_layers']
# lr_main = 0.001
input_channel_main = (len(params['opt_bands']) + len(params['indices_lst']))*2
input_channel_lidar = len(params['lid_bands'])
input_channel_radar = 6
num_classes = 9 

# Inference parameters
img_size = (256,256)
wanted_res = 3
overlap = 0.5
batch_size = 1
gaussian_filter = False

def merge_prediction(pred_raster, win, model_pred):
    pred = pred_raster.read(window=win)[:-1, :, :]
    if len(model_pred.shape)==3:
        pred = np.moveaxis(pred, 0, -1)
        for i in range(model_pred.shape[-1]):
            model_pred[:,:,i] = model_pred[:,:,i] * avg_filter
    else:
        pred = pred[0]
        model_pred = model_pred * avg_filter
    pred += model_pred
    n_pred = pred_raster.read(pred_raster.count, window=win)
    n_pred += avg_filter
    return pred, n_pred

import numpy as np

def save_prediction(pred_raster, win, pred):
    for i in range(pred.shape[-1]):
        pred_raster.write(pred[:,:,i], window=win, indexes=i+1)

# def save_prediction(pred_raster, win, pred):
#     pred_raster.write(pred, window=win, indexes=1)

def get_new_index(idx, last_bool, done_bool, dim):
    if last_bool:
        done_bool = True
    else:
        idx += int(load_size[dim] * (1.0-overlap))
        if idx+load_size[dim] >= img_file.shape[dim]:
            idx = img_file.shape[dim] - load_size[dim]
            last_bool = True
    return idx, last_bool, done_bool

def clip_image(img, low=0, high=10000):
    return np.clip(img, low, high)

def calculate_indices(image, selected_bands):

    band_names = {
        "B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, "B7": 6, "B8": 7,
        "B8A": 8, "B9": 9, "B11": 10, "B12": 11
    }

    #selected_band_indices = {band: index for band, index in band_names.items() if index in selected_bands}

    selected_band_indices = {band: index for band, index in band_names.items() if index in selected_bands}
    opt_band_indices = {band: idx for idx, band in enumerate(selected_bands)}


    indices = {}
    

    # NDVI (NIR - R) / (NIR + R) | (8 - 4) / (8 + 4)
    ndvi = (image[opt_band_indices[selected_band_indices['B8']]] - image[opt_band_indices[selected_band_indices['B4']]]) / (image[opt_band_indices[selected_band_indices['B8']]] + image[opt_band_indices[selected_band_indices['B4']]])
    indices['ndvi'] = np.nan_to_num(ndvi, -1)

    # NDWI (GREEN - NIR) / (Green + NIR) | (3 - 8) / (3 + 8)
    ndwi = (image[opt_band_indices[selected_band_indices['B3']]] - image[opt_band_indices[selected_band_indices['B8']]]) / (image[opt_band_indices[selected_band_indices['B3']]] + image[opt_band_indices[selected_band_indices['B8']]])
    indices['ndwi'] = np.nan_to_num(ndwi, -1)

    # NDMI (NIR - SWIR) / (NIR + SWIR) | (8 - 11) / (8 + 11)
    ndmi = (image[opt_band_indices[selected_band_indices['B8']]] - image[opt_band_indices[selected_band_indices['B11']]]) / (image[opt_band_indices[selected_band_indices['B8']]] + image[opt_band_indices[selected_band_indices['B11']]])
    indices['ndmi'] = np.nan_to_num(ndmi, -1)

    # MBWI (omega * G) - R - N - S1 - S2 | (omega * 3) - (4 + 8 + 11 + 12)  # omega == 2
    mbwi = (2 * image[opt_band_indices[selected_band_indices['B3']]]) - (image[opt_band_indices[selected_band_indices['B4']]] + image[opt_band_indices[selected_band_indices['B8']]] + image[opt_band_indices[selected_band_indices['B11']]] + image[opt_band_indices[selected_band_indices['B12']]])
    indices['mbwi'] = np.nan_to_num(mbwi, -1)

    return indices


# class SemSegment(LightningModule):
#     def __init__(
#         self,
#         lr: float = 0.001,
#         num_classes: int = 9,
#         num_layers: int = 4,
#         features_start: int = 64,
#         bilinear: bool = True,

#     ):

#         super().__init__()
    
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.features_start = features_start
#         self.bilinear = bilinear
#         self.lr = lr
#         self.new_time = time.time()
#         self.train_time_list = []

#         # Model
#         self.net = unet_3enco_sum(
#             num_classes=num_classes,
#             input_channels=input_channel_main,
#             input_channels_lidar=input_channel_lidar,
#             input_channels_radar=input_channel_radar,
#             num_layers=num_layers_main,
#             features_start=self.features_start,
#             bilinear=self.bilinear,
#         )

#     def forward(self, x, y, z):
#         return self.net(x, y, z)

# Lood model and change weights (in this case remove NDVI)
# old_model = SemSegment.load_from_checkpoint(
#       checkpoint_path=params['chkp_path']
#       )

# # Remove NDVI weights and create new model
# pretrained_state_dict = old_model.state_dict() # Fetch state_dict from first model
# input_channel_main = (len(params['opt_bands']))*2
                      
# model = SemSegment()
# #new_state_dict = new_model.state_dict() # Fetch state_dict from second model

# from collections import OrderedDict
# new_state_dict = OrderedDict()
# params['indices_lst'] = []

# for key, value in pretrained_state_dict.items():
#     if key == 'net.conv_opt.0.net.0.weight':
#         # Remove the last channel (NDVI) from the weights
#         new_weights = value[:, :-2, :, :]
#         new_state_dict[key] = new_weights
#     else:
#         new_state_dict[key] = value

# model.load_state_dict(new_state_dict)

# Normal loading of model 
# model = SemSegment.load_from_checkpoint(
#     checkpoint_path=params['chkp_path']
#     )

def load_config(file_path):
    config = OmegaConf.load(file_path)
    return config

config_file_path = 'conf/model/model_multi.yaml'
config = load_config(config_file_path)
config.architecture = params['architecture']

# Normal loading of model 
model = SemSegment.load_from_checkpoint(
    checkpoint_path=params['chkp_path'],
    cfg=config,
    class_weights=None,
    strict=False
    )

# Read data files
img_file = rasterio.open(img_path, 'r')

# Adjust image resolution if needed
img_res = img_file.transform[0]
load_size = (int(img_size[0]*wanted_res/img_res), int(img_size[0]*wanted_res/img_res))

if (load_size[0]-1)%2 == 0:
    load_size = (int(img_size[0]*wanted_res/img_res)+1, int(img_size[0]*wanted_res/img_res)+1)

# Create averaging filter
if overlap == 0 or gaussian_filter == False:
    avg_filter = np.ones(load_size)
else:
    avg_filter = np.zeros(load_size)
    avg_filter[int(load_size[0]/4):int(3*load_size[0]/4), int(load_size[1]/4):int(3*load_size[1]/4)] = 1
    avg_filter = cv2.GaussianBlur(avg_filter, (load_size[0]-1,load_size[1]-1), 0)
    avg_filter = avg_filter + 0.25
    avg_filter = avg_filter / avg_filter.max()
    # plt.imshow(avg_filter)
    # plt.show()

# create segmentation raster
seg_kwds = img_file.profile
#seg_kwds['count'] = n_labels+1
seg_kwds['count'] = num_classes + 1
seg_kwds['dtype'] = np.float32
seg_kwds['nodata'] = 0
seg_kwds['driver'] = 'GTiff'

seg_file = rasterio.open(
    save_path + model_name + '.tif',
    'w+',
    BIGTIFF=True,
    **seg_kwds
)

# Load means and stdevs
if region_preds == 'estrie':
    print('Loadings means and stdevs (estrie)')
    sen2_e_means = np.load('stats/estrie/estrie_sen2_ete_means.npy')
    sen2_p_means = np.load('stats/estrie/estrie_sen2_pri_means.npy')
    sen1_e_means = np.load('stats/estrie/estrie_sen1_ete_means.npy')
    sen1_p_means = np.load('stats/estrie/estrie_sen1_pri_means.npy')
    lidar_means  = np.load('stats/estrie/estrie_lidar_means_v2.npy')
    
    sen2_e_stdevs = np.load('stats/estrie/estrie_sen2_ete_stds.npy')
    sen2_p_stdevs = np.load('stats/estrie/estrie_sen2_pri_stds.npy')
    sen1_e_stdevs = np.load('stats/estrie/estrie_sen1_ete_stds.npy')
    sen1_p_stdevs = np.load('stats/estrie/estrie_sen1_pri_stds.npy')
    lidar_stdevs  = np.load('stats/estrie/estrie_lidar_stds_v2.npy')

elif region_preds == 'portneuf_sud':
    print('Loadings means and stdevs (portneuf_zone2 sud)')
    sen2_e_means = np.load('stats/portneuf_zone2/portneuf_zone2_sen2_ete_means.npy')
    sen2_p_means = np.load('stats/portneuf_zone2/portneuf_zone2_sen2_pri_means.npy')
    sen1_e_means = np.load('stats/portneuf_zone2/portneuf_zone2_sen1_ete_means.npy')
    sen1_p_means = np.load('stats/portneuf_zone2/portneuf_zone2_sen1_pri_means.npy')
    lidar_means  = np.load('stats/portneuf_zone2/portneuf_zone2_lidar_means.npy')

    sen2_e_stdevs = np.load('stats/portneuf_zone2/portneuf_zone2_sen2_ete_stds.npy')
    sen2_p_stdevs = np.load('stats/portneuf_zone2/portneuf_zone2_sen2_pri_stds.npy')
    sen1_e_stdevs = np.load('stats/portneuf_zone2/portneuf_zone2_sen1_ete_stds.npy')
    sen1_p_stdevs = np.load('stats/portneuf_zone2/portneuf_zone2_sen1_pri_stds.npy')
    lidar_stdevs  = np.load('stats/portneuf_zone2/portneuf_zone2_lidar_stds.npy')

elif region_preds == 'portneuf_nord':
    print('Loadings means and stdevs (portneuf_zone1 nord)')
    sen2_combined_means = np.load('stats/portneuf_zone1/sen2_means_portneuf_nord.npy')
    sen2_combined_stdevs = np.load('stats/portneuf_zone1/sen2_stdev_portneuf_nord.npy')

    sen1_combined_means = np.load('stats/portneuf_zone1/sen1_means_portneuf_nord.npy')
    sen1_combined_stdevs = np.load('stats/portneuf_zone1/sen1_stdev_portneuf_nord.npy')

    sen2_e_means = sen2_combined_means[:12]
    sen2_p_means = sen2_combined_means[12:]
    sen1_e_means = sen1_combined_means[:3]
    sen1_p_means = sen1_combined_means[3:]
    

    sen2_e_stdevs = sen2_combined_stdevs[:12]
    sen2_p_stdevs = sen2_combined_stdevs[12:]
    sen1_e_stdevs = sen1_combined_stdevs[:3]
    sen1_p_stdevs = sen1_combined_stdevs[3:]

    lidar_means  = np.load('stats/portneuf_zone1/lidar_means_portneuf_nord.npy')
    lidar_stdevs  = np.load('stats/portneuf_zone1/lidar_stdev_portneuf_nord.npy')

n_passes = 1
for pass_no in range(n_passes):
    print(f"Pass {pass_no + 1}/{n_passes}")

    # loop in image
    idx_i, last_i, done_i = 0, False, False
    idx_j, last_j, done_j = 0, False, False
    while not (done_j and done_i):
            
        # Create images batch
        cur_batch_size = 0
        img_batch = []
        img_windows = []
        while (cur_batch_size != batch_size):
            
            # get read and write window
            img_win = windows.Window(idx_j, idx_i, load_size[0], load_size[1])
            
            # read windowed data
            img = img_file.read(window=img_win)
            img = np.moveaxis(img, 0, -1)

            if not (img==0).all():
                
                # fill nans
                img[np.isnan(img)] = 0
                
                # resize image
                if load_size != img_size:
                    img = cv2.resize(img, img_size)
                
                # standardization
                #img = img / 255.
                #img = np.true_divide(np.subtract(img, means), stds)
                
                # prepare dimensions for prediction
                img_batch.append(img)
                img_windows.append(img_win)
                cur_batch_size += 1
            
            # get new indexes
            idx_j, last_j, done_j = get_new_index(idx_j, last_j, done_j, 1)
            if done_j:
                idx_i, last_i, done_i = get_new_index(idx_i, last_i, done_i, 0)
                print(round(idx_i / img_file.shape[0] * 100, 2))
                if not done_i:
                    idx_j, last_j, done_j = 0, False, False
                else:
                    break
                
        # model prediction
        #img_batch = np.asarray(img_batch)
        #to_show = np.copy(img_batch)
        #batch_pred = model(img_batch).numpy().squeeze()

        img = np.asarray(img_batch)

        # Layer division
        #sen2_ete_img = clip_image(img[:,:,:,:12].squeeze())
        #sen2_print_img = clip_image(img[:,:,:,12:24].squeeze())
        sen2_ete_img = img[:,:,:,:12].squeeze()
        sen2_print_img = img[:,:,:,12:24].squeeze()

        # Select bands for Sen2
        if params['opt_bands'] is not None:
            sen2_ete_img = sen2_ete_img[:,:,params['opt_bands']]
            sen2_print_img = sen2_print_img[:,:,params['opt_bands']]

        img_opt = np.dstack((sen2_ete_img, sen2_print_img))

        img_rad = img[:,:,:,24:30].squeeze()

        img_lidar = img[:,:,:,30:].squeeze()

        # Select bands for lidar
        if params['lid_bands'] is not None and img_lidar.shape[2] == 6:
            lidar_band_indices = {'mnt': 0, 'mhc': 1, 'pentes': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
        elif params['lid_bands'] is not None and img_lidar.shape[2] == 5:
            lidar_band_indices = {'mhc': 0, 'pentes': 1, 'tpi': 2, 'tri': 3, 'twi': 4}

        lidar_band_indices_selected = [lidar_band_indices[band] for band in params['lid_bands']]
        img_lidar = img_lidar[:,:,lidar_band_indices_selected]

        if img_lidar.dtype != 'float32':
            img_lidar = np.float32(img_lidar) # Only for overlapping dataset #TODO
        else:
            pass

        #estrie_mean_e  = [259.97108705, 277.34900677, 520.46502329, 342.23574781, 906.76114884, 2656.35829517, 3203.35430934, 3389.62506118, 3487.07960017, 3555.4164092, 1714.22609075, 828.27687406]
        #estrie_stdev_e = [525.55511221, 526.47685896, 515.89037279, 527.365679, 561.52225037, 836.14547148, 984.91903497, 1067.04202788, 1026.75692634, 1066.1236181, 630.05843599, 505.20760634]

        # lidar_means = torch.tensor([7.798849, 5.5523205, 0.0029951811, 0.06429929, 6.7409873])
        # lidar_stdevs  = torch.tensor([7.033332, 5.196636, 1.0641352, 0.06102526, 3.182435])

        # Combine mean lists for sen1 and sen2
        sen2_means = [*sen2_e_means, *sen2_p_means]
        sen1_means = [*sen1_e_means, *sen1_p_means]

        # Combine stdev lists for sen1 and sen2
        sen2_stdevs = [*sen2_e_stdevs, *sen2_p_stdevs]
        sen1_stdevs = [*sen1_e_stdevs, *sen1_p_stdevs]

        mean_lst = [sen2_means, sen1_means, lidar_means]
        stdev_lst = [sen2_stdevs, sen1_stdevs, lidar_stdevs]

        # Mean and std value lists
        # Sentinel 2
        #s2_e_e_mean = torch.tensor([259.971087045696, 277.3490067676725, 520.4650232890134, 342.23574780553645, 906.7611488412249, 2656.3582951694643, 3203.3543093369944, 3389.6250611778078, 3487.079600166239, 3555.416409200909, 1714.2260907527316, 828.2768740555728, 457.4229830346009, 501.79759875320303, 694.4711397083421, 835.1158882308216, 1219.9447441650816, 1823.0661322180392, 2064.6505317461747, 2316.1887302003915, 2363.5869859139643, 2359.4662122932396, 2390.6124116260303, 1586.6126304451745])
        #s2_e_e_std  = torch.tensor([525.5551122108338, 526.4768589585602, 515.8903727938966, 527.3656790023017, 561.5222503677404, 836.1454714836563, 984.9190349745415, 1067.0420278801334, 1026.7569263359944, 1066.123618103052, 630.0584359871733, 505.2076063419134, 169.44646075504082, 249.03030944938908, 293.96819726121373, 408.20429488371605, 392.1811051266158, 492.36521601358254, 550.8773405439316, 623.9017038640061, 590.0457818993959, 540.556974947324, 740.4564895487368, 581.7629650224691])
        
        # Sentinel 1
        #s1_e_p_mean = torch.tensor([-15.479797, -9.211855, 6.267961, -15.0310545, -9.519093, 5.5120163])
        #s1_e_p_std  = torch.tensor([1.622046, 1.8651232, 1.2285297, 2.1044014, 1.9065734, 1.37706]) 

        # Lidar
        #estrie_lidar_mean = torch.tensor([7.798849, 5.5523205, 0.0029951811, 0.06429929, 6.7409873])
        #estrie_lidar_std  = torch.tensor([7.033332, 5.196636, 1.0641352, 0.06102526, 3.182435])

        ## Mean and std value lists
        # # Sentinel 2
        # s2_e_e_mean = torch.tensor(np.load('stats/portneuf_zone1/sen2_means_portneuf_nord.npy'))
        # s2_e_e_std  = torch.tensor(np.load('stats/portneuf_zone1/sen2_stdev_portneuf_nord.npy'))

        # # Sentinel 1
        # s1_e_p_mean = torch.tensor(np.load('stats/portneuf_zone1/sen1_means_portneuf_nord.npy'))
        # s1_e_p_std  = torch.tensor(np.load('stats/portneuf_zone1/sen1_stdev_portneuf_nord.npy')) 

        # # Lidar
        # estrie_lidar_mean = torch.tensor(np.load('stats/portneuf_zone1/lidar_means_portneuf_nord.npy'))
        # estrie_lidar_std  = torch.tensor(np.load('stats/portneuf_zone1/lidar_stdev_portneuf_nord.npy'))


        # Combine mean lists for sen1 and sen2
        # sen2_means = [*s2_e_e_mean]
        # sen1_means = [*s1_e_p_mean]

        # sen2_stdevs = [*s2_e_e_std]
        # sen1_stdevs = [*s1_e_p_std]

        # sen2_means = [*sen2_means]
        # sen1_means = [*s1_e_p_mean]

        # sen2_stdevs = [*sen2_stdevs]
        # sen1_stdevs = [*s1_e_p_std]

        # mean_lst = [sen2_means, sen1_means, estrie_lidar_mean]
        # stdev_lst = [sen2_stdevs, sen1_stdevs, estrie_lidar_std]

        # # means and stds by images TEMP
        # sen2_means = torch.tensor(mean_lst[0])
        # sen2_stdevs = torch.tensor(stdev_lst[0])
        
        # sen1_means = torch.tensor(mean_lst[1])
        # sen1_stdevs = torch.tensor(stdev_lst[1])

        # # lidar_means = mean_lst[2].clone().detach()
        # # lidar_stdevs = stdev_lst[2].clone().detach()

        # lidar_means = torch.tensor(mean_lst[2][1:])
        # lidar_stdevs = torch.tensor(stdev_lst[2][1:])

        #Loads means and stds for standardization according to selected bands
        if mean_lst is not None and stdev_lst is not None:
            if params['opt_bands'] is not None:
                # Extract means and stds for selected bands (and add 12 to get the corresponding spring set bands)
                opt_bands_s1s2 = sorted(params['opt_bands'] + [x+12 for x in params['opt_bands']])
                sen2_means = torch.tensor([mean_lst[0][i] for i in opt_bands_s1s2])
                sen2_stdevs = torch.tensor([stdev_lst[0][i] for i in opt_bands_s1s2])
            else:
                sen2_means = torch.tensor(mean_lst[0])
                sen2_stdevs = torch.tensor(stdev_lst[0])

            # TODO Handling mnt not included in means and stds, remove when .npy files update
            if params['lid_bands'] is not None and img_lidar.shape[2] == 6:
                # Extract means and stds for selected LiDAR bands
                lidar_band_indices = {'mnt': 0, 'mhc': 1, 'pentes': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
            elif params['lid_bands'] is not None and img_lidar.shape[2] == 5: 
                lidar_band_indices = {'mhc': 0, 'pentes': 1, 'tpi': 2, 'tri': 3, 'twi': 4}
            else:
                raise ValueError("Conflicting number of LiDAR bands")

            lidar_means = torch.tensor([mean_lst[2][lidar_band_indices[band]] for band in params['lid_bands']])
            lidar_stdevs = torch.tensor([stdev_lst[2][lidar_band_indices[band]] for band in params['lid_bands']])

            # No bands selection for radar
            sen1_means = torch.tensor(mean_lst[1])
            sen1_stdevs= torch.tensor(stdev_lst[1])

        else:
            print('No means or stds available for standardization')

        #Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt).permute(2, 0, 1)
        img_rad = torch.from_numpy(img_rad).permute(2, 0, 1)
        img_lidar = torch.from_numpy(img_lidar).permute(2, 0, 1)

        # Apply standardization
        if mean_lst is not None and stdev_lst is not None:
            img_opt = img_opt.sub_(sen2_means[:, None, None]).div_(sen2_stdevs[:, None, None])
            img_rad = img_rad.sub_(sen1_means[:, None, None]).div_(sen1_stdevs[:, None, None])
            img_lidar = img_lidar.sub_(lidar_means[:, None, None]).div_(lidar_stdevs[:, None, None])

        # # Add indices after standardizationimg_
        # if len(self.indices_lst) > 0 and (sen2_ete_img.shape[2] != 12 or sen2_print_img.shape[2] != 12):
        #     raise ValueError("Number of bands must be 12 to calculate indices (not implemented for other number of bands)")
        if len(params['indices_lst']) > 0:
            sen2_ete_img = torch.from_numpy(sen2_ete_img).permute(2, 0, 1)
            sen2_print_img = torch.from_numpy(sen2_print_img).permute(2, 0, 1)

            # # Calculate indices here
            # ete_indices = {key: value for key, value in self.calculate_indices(sen2_ete_img).items() if key in self.indices_lst}
            # print_indices = {key: value for key, value in self.calculate_indices(sen2_print_img).items() if key in self.indices_lst}

            #ete_indices = {key: value for key, value in calculate_indices(sen2_ete_img, params['opt_bands']).items() if key in params['indices_lst']}
            #print_indices = {key: value for key, value in calculate_indices(sen2_print_img, params['opt_bands']).items() if key in params['indices_lst']}

            ete_indices = {key: value for key, value in calculate_indices(sen2_ete_img, params['opt_bands']).items() if key in params['indices_lst']}
            print_indices = {key: value for key, value in calculate_indices(sen2_print_img, params['opt_bands']).items() if key in params['indices_lst']}

            # Concatenate images and their indices
            ete_img_indices = torch.cat([torch.from_numpy(ete_indices[key]).unsqueeze(0) for key in ete_indices], dim=0)
            print_img_indices = torch.cat([torch.from_numpy(print_indices[key]).unsqueeze(0) for key in print_indices], dim=0)
            img_opt = torch.cat((img_opt, ete_img_indices, print_img_indices), dim=0)


        # Unsqueeze to add batch dimension and simulate batch of 1
        img_opt = img_opt.unsqueeze(0)
        img_lidar = img_lidar.unsqueeze(0)
        img_rad = img_rad.unsqueeze(0)

        # Start evaluation
        model.eval()
        batch_pred = model(img_opt, img_lidar, img_rad) #.detach().numpy() #.squeeze()
                
        # write to mosaic
        for ii in range(len(img_windows)):
            # read prediction and position in raster
            pred = batch_pred[ii]

            #pred = pred.softmax(dim = 0) # important pour le cumul avec Pytorch, pas besoin avec TF
            pred = pred.detach().numpy()
            pred = np.moveaxis(pred, 0, -1)

            img_win = img_windows[ii]
        
            # resize prediction
            if load_size != img_size:
                pred = cv2.resize(pred, (img_win.height, img_win.width))
            
            # merge prediction
            pred, n_pred = merge_prediction(seg_file, img_win, pred)
            pred = np.dstack([pred, n_pred]) # Enlever pour argmax direct

            # Save to prob rasters
            #plt.subplot(121), plt.imshow(img_batch[ii])
            #plt.subplot(122), plt.imshow(pred[...,0] / pred[...,1])
            #plt.show()

            save_prediction(seg_file, img_win, pred)

seg_file.close()