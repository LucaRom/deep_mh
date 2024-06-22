'''
#TODO Demander si on peut le mettre en ligne (si besoin bien sûr) et comment bien mettre les droits d'auteurs

Adapted from original script author : Marc-Antoine Genest (Cerfo)
''' 

import os


from unet_models.unet_3enco_concat_attention import unet_3enco_concat_attention
import tifffile as tiff
import rasterio
from rasterio import windows
from rasterio.windows import Window
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import tifffile
#import tensorflow as tf
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from scipy.stats import mode
import random
from tqdm import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_config(region_preds):
    regions = {
        'estrie': {
            'img_path': '/mnt/SN750/01_Code_nvme/Master_2024/results/full_stack/stack_estrie_3m_test_v04.tif',
            'mean_std_path': 'stats/estrie/',
        },
        'portneuf_sud': {
            'img_path': '/mnt/SN750/01_Code_nvme/Master_2024/results/full_stack/stack_portneuf_3m_zone2.tif',
            'mean_std_path': 'stats/portneuf_zone2/',
        },
        'portneuf_nord': {
            'img_path': '/mnt/d/00_Donnees/02_maitrise/01_trainings/portneuf_zone1/portneuf_nord_stack_inference.tif',
            'mean_std_path': 'stats/portneuf_zone1/',
        }
    }
    
    config = {
        'save_path': '/mnt/SN750/01_Code_nvme/Master_2024/results/inference/',
        'model_details': {
            'm222': {
                'chkp_path': '/mnt/f/01_Code_nvme/Master_2024/lightning_logs/version_124/checkpoints/topk_epoch-epoch=199-val_loss=0.27.ckpt',
                'num_layers': 4,
                'opt_bands': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
                'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'],
                'rad_bands': [0,1,2],
                'indices_lst': ['ndvi'],
            }
        },
        'region': regions[region_preds],
        'selected_model': 'm222',
        'block_size': 256,
        'num_passes': 20,
        'max_offset': 128,
        'overlap_ratio': 0.5,
        'num_classes': 9,
    }

    return config

def load_stats(region_preds):
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


    # Combine mean lists for sen1 and sen2
    sen2_means = [*sen2_e_means, *sen2_p_means]
    sen1_means = [*sen1_e_means, *sen1_p_means]

    # Combine stdev lists for sen1 and sen2
    sen2_stdevs = [*sen2_e_stdevs, *sen2_p_stdevs]
    sen1_stdevs = [*sen1_e_stdevs, *sen1_p_stdevs]

    # mean_lst = [sen2_means, sen1_means, lidar_means]
    # stdev_lst = [sen2_stdevs, sen1_stdevs, lidar_stdevs]

    return sen2_means, sen1_means, sen2_stdevs, sen1_stdevs, lidar_means, lidar_stdevs

class SemSegment(LightningModule):
    def __init__(self, config):
        super().__init__()

        print(config)

        # Model initialization
        self.net = unet_3enco_concat_attention(
            num_classes=config['num_classes'],
            input_channels=(len(config['model_details']['m222']['opt_bands']) + len(config['model_details']['m222']['indices_lst']))*2,
            input_channels_lidar=len(config['model_details']['m222']['lid_bands']),
            input_channels_radar=len(config['model_details']['m222']['rad_bands'])*2,
            #num_layers=self.cfg.num_layers,
            features_start=64,
            bilinear=True,
        )

    def forward(self, x, y, z):
        return self.net(x, y, z)

def pre_process(win_data, config):
    # Split the window data into its components (optical, LiDAR, etc.)
    sen2_ete_img, sen2_print_img, img_rad, img_lidar = split_window_data(win_data, config)

    # Select bands
    sen2_ete_img = sen2_ete_img[config['model_details'][config['selected_model']]['opt_bands'], :, :]
    sen2_print_img = sen2_print_img[config['model_details'][config['selected_model']]['opt_bands'], :, :]
    
    # Load stats
    sen2_means, sen1_means, sen2_stdevs, sen1_stdevs, lidar_means, lidar_stdevs = load_stats(region_preds)

    # Normalization (Assuming normalization parameters are loaded into config)
    img_opt = standardize_bands(sen2_ete_img, sen2_print_img, sen2_means, sen2_stdevs, band_type='optical')
    img_rad = standardize_bands(img_rad, sen1_means, sen1_stdevs, config, band_type='radar')
    img_lidar = standardize_bands(img_lidar, lidar_means, lidar_stdevs, band_type='lidar')

    # Calculate indices and augment data
    img_opt = calculate_and_augment_indices(img_opt, config)

    # Prepare data for the model (add batch dimension, etc.)
    img_opt, img_lidar, img_rad = prepare_for_model(img_opt, img_lidar, img_rad)
    
    return img_opt, img_lidar, img_rad

def split_window_data(win_data, config):
    # Example implementation: Adjust indices based on your data layout
    sen2_ete_img = win_data[:12,:,:]
    sen2_print_img = win_data[12:24,:,:]
    img_lidar = win_data[30:, :, :]  # Next 5 bands for LiDAR data
    img_rad = win_data[24:30, :, :]  # Remaining bands for radar data
    return sen2_ete_img, sen2_print_img, img_lidar, img_rad

def standardize_bands(img, stats_inputs, band_type='optical'):
    """
    Normalize the bands of the image data using pre-defined means and standard deviations.
    Args:
    - img: Numpy array of primary image bands.
    - img_optional: Numpy array of optional image bands, used for dual season images.
    - config: Configuration dictionary containing mean and std values.
    - band_type: The type of bands ('optical', 'radar', 'lidar') to apply normalization to.
    
    Returns:
    - Normalized image bands as a numpy array or a PyTorch tensor.
    """
    
    # Extract mean and std values from config based on the band type
    if band_type == 'optical':
        mean, std = np.array(config['mean_lst'][0]), np.array(config['stdev_lst'][0])
    elif band_type == 'radar':
        mean, std = np.array(config['mean_lst'][1]), np.array(config['stdev_lst'][1])
    elif band_type == 'lidar':
        mean, std = np.array(config['mean_lst'][2]), np.array(config['stdev_lst'][2])
    else:
        raise ValueError("Invalid band type specified for normalization.")
    
    # Apply normalization
    # Ensure dimensions match for broadcasting during normalization
    mean = mean[:img.shape[0]]
    std = std[:img.shape[0]]
    img_normalized = (img - mean[:, None, None]) / std[:, None, None]
    
    # Optionally, convert to a PyTorch tensor
    img_normalized = torch.tensor(img_normalized, dtype=torch.float32)

    return img_normalized

def calculate_and_augment_indices(img_opt, config):
    """
    Calculate indices such as NDVI and augment the optical image data with these indices.
    Args:
    - img_opt: A tensor of optical image bands.
    - config: Configuration dictionary with details on which indices to calculate and how.
    
    Returns:
    - Augmented image data with calculated indices.
    """
    indices = {}
    # Assuming 'opt_bands' and 'indices_lst' are specified in the config for the selected model.
    opt_bands = config['model_details'][config['selected_model']]['opt_bands']
    indices_lst = config['model_details'][config['selected_model']]['indices_lst']
    
    # Example for NDVI calculation
    if 'ndvi' in indices_lst:
        # Find the indices for NIR and Red bands in the opt_bands list
        # Assuming B8 is NIR and B4 is Red, adjust indices as per your dataset
        NIR_index = opt_bands.index(7)  # Assuming band B8 is at index 7 in opt_bands
        Red_index = opt_bands.index(3)  # Assuming band B4 is at index 3 in opt_bands
        
        NIR = img_opt[:, NIR_index, :, :]
        Red = img_opt[:, Red_index, :, :]
        
        NDVI = (NIR - Red) / (NIR + Red + 1e-10)  # Adding a small value to avoid division by zero
        indices['ndvi'] = NDVI
        
    # Add more indices as needed...
    
    # Augment the img_opt with calculated indices
    for index in indices.values():
        img_opt = torch.cat((img_opt, index.unsqueeze(1)), dim=1)
    
    return img_opt

def prepare_for_model(img_opt, img_lidar, img_rad):
    # Convert numpy arrays to PyTorch tensors and add batch dimension
    img_opt = torch.from_numpy(img_opt).unsqueeze(0).float().to(device)
    img_lidar = torch.from_numpy(img_lidar).unsqueeze(0).float().to(device)
    img_rad = torch.from_numpy(img_rad).unsqueeze(0).float().to(device)
    return img_opt, img_lidar, img_rad

def process_image(img_path, model, save_path, config):
    """
    Process the entire image in chunks, predict each chunk, and save the aggregated result.
    Args:
    - img_path: Path to the input image.
    - model: The trained model for prediction.
    - save_path: Path where the output image will be saved.
    - config: Configuration dictionary containing parameters for processing.
    """
    # Load the image with rasterio
    with rasterio.open(img_path) as src:
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': 'float32'})
        
        # Initialize a result matrix to accumulate the predictions
        result_matrix = np.zeros((src.height, src.width), dtype=np.float32)

        # Calculate block size and overlap for window processing
        block_size = config['block_size']
        overlap = int(block_size * config['overlap_ratio'])
        stride = block_size - overlap
        
        # Loop over the image in windows
        for row in tqdm(range(0, src.height, stride), desc='Rows'):
            for col in tqdm(range(0, src.width, stride), desc='Cols', leave=False):
                window = Window(col, row, block_size, block_size)
                if src.read(window=window).shape == (src.count, block_size, block_size):
                    img_window = src.read(window=window)
                    # Preprocess the image window
                    img_opt, img_lidar, img_rad = pre_process(img_window, config)
                    
                    # Predict with the model
                    with torch.no_grad():
                        prediction = model(img_opt.to(device), img_lidar.to(device), img_rad.to(device))
                    prediction = prediction.cpu().numpy()
                    
                    # Aggregate the predictions
                    result_matrix[row:row+block_size, col:col+block_size] = prediction.squeeze()
        
        # Save the result using the meta data
        with rasterio.open(save_path, 'w', **meta) as dst:
            dst.write(result_matrix, 1)

def main():
    region_preds = 'estrie'
    config = load_config(region_preds)
    model = SemSegment.load_from_checkpoint(
        checkpoint_path=config['model_details']['m222']['chkp_path'],
        config=config
        )
    process_image(config['region']['img_path'], model, config['save_path'], config)

if __name__ == "__main__":
    main()
