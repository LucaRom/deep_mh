import datetime
import os
import random
import time

import cv2
import numpy as np
import rasterio
import tifffile
import tifffile as tiff
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer, seed_everything
from rasterio import windows
from rasterio.windows import Window
from scipy.stats import mode
from tqdm import tqdm
import torch.nn.functional as F

from seg_3_enco_multi import SemSegment
from unet_models.unet_3enco_concat_attention import unet_3enco_concat_attention

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def save_intermediate_output(output, row, col, out_meta, output_dir):
    output_path = os.path.join(output_dir, f"output_{row}_{col}.tif")
    with rasterio.open(output_path, 'w', **out_meta) as out_raster:
        out_raster.write(output, indexes=1)  # Save the output as a TIFF file

def pad_to_size(img_window, target_height, target_width):
    # Convert numpy array to a tensor if not already
    if not isinstance(img_window, torch.Tensor):
        img_window = torch.from_numpy(img_window).float()

    # Calculate necessary padding amounts
    pad_height = max(0, target_height - img_window.shape[0])
    pad_width = max(0, target_width - img_window.shape[1])

    # Apply padding
    img_window = F.pad(img_window, (0, pad_width, 0, pad_height), mode='reflect')  # Padding on right and bottom edges

    return img_window

# def process_image(img_path, model, save_path, block_size=256, num_passes=3, max_offset=64, overlap_ratio=0.5):
#     # Calculate the overlap size based on the specified overlap ratio
#     if overlap_ratio == 0:
#         overlap_size = int(block_size * 1)
#     else:
#         overlap_size = int(block_size * overlap_ratio)

#     # Open the input image using rasterio
#     with rasterio.open(img_path) as img_file:
        
#         img_meta = img_file.meta

#         # Create the output segmentation raster with the same properties as the input image
#         seg_kwds = img_meta.copy()
#         seg_kwds.update({'count': 1, 'dtype': np.float32})
        
#         # Initialize a counter array to accumulate class counts
#         #class_counter = np.zeros((model.num_classes, img_meta['height'], img_meta['width']) , dtype=np.uint16)
#         class_counter = np.zeros((model.num_classes, img_meta['height'], img_meta['width']), dtype=np.float32)

#         # Iterate through the passes
#         for n in range(num_passes):
#             print('Pass {} of {}'.format(n+1, num_passes))
#             # Generate random offsets for the starting position
#             row_offset = random.randint(0, max_offset)
#             col_offset = random.randint(0, max_offset)

#             # Iterate through the image using windows with the specified block size and overlap size
#             # for row in tqdm(range(row_offset, img_meta['height'] - block_size + 1, overlap_size), desc="Rows"):
#             #     for col in tqdm(range(col_offset, img_meta['width'] - block_size + 1, overlap_size), desc="Cols", leave=False):
#             for row in tqdm(range(row_offset, img_meta['height'] - block_size, overlap_size), desc="Rows"):
#                 for col in tqdm(range(col_offset, img_meta['width'] - block_size, overlap_size), desc="Cols", leave=False):
#                     win = Window(col_off=col, row_off=row, width=block_size, height=block_size)

#                     # Read the image window and normalize the values
#                     img_win = img_file.read(window=win) #/ img_file.scales[0]

#                     img_opt, img_lidar, img_rad = pre_process(img_win)

#                     img_opt = img_opt.float()
#                     img_lidar = img_lidar.float()
#                     img_rad = img_rad.float()

#                     img_opt = img_opt.to(device)
#                     img_lidar = img_lidar.to(device) 
#                     img_rad = img_rad.to(device) 

#                     # Get the model prediction for the current window
#                     #model_pred = model.predict(img_win[np.newaxis, ...])[0]
#                     model.eval()
#                     batch_pred = model(img_opt, img_lidar, img_rad)

#                     pred = batch_pred[0]  # Use the first (and only) element in the batch
#                     pred_np = pred.detach().cpu()
#                     #model_pred = np.moveaxis(pred_np, 0, -1)
#                     model_pred = pred_np.numpy()

#                     # pred_argmax = pred.argmax(dim=0)
#                     # tmp_meta = img_meta.copy()
#                     # # Update metadata for output specifics
#                     # tmp_meta.update({
#                     #     'driver': 'GTiff',  # GeoTIFF format
#                     #     'height': pred_argmax.shape[0],  # Number of rows in the output
#                     #     'width': pred_argmax.shape[1],  # Number of columns in the output
#                     #     'count': 1,  # Number of layers/bands in the output
#                     #     'dtype': 'float32',  # Data type of the output, adjust as needed
#                     # })
                    
#                     # DEBUG
#                     #save_intermediate_output(pred_argmax, row, col, tmp_meta, save_path_root)

#                     # Merge the model prediction with the current segmentation raster
#                     for band in range(model.num_classes):
#                         #pred_band = model_pred[..., band - 1]
#                         pred_band = model_pred[band, ...]
#                         #class_counter[band - 1, row:row + pred_band.shape[0], col:col + pred_band.shape[1]] += pred_band.astype(np.uint16)
#                         class_counter[band, row:row + pred_band.shape[0], col:col + pred_band.shape[1]] += pred_band #.astype(np.float32)


#                     # Refresh the progress bar display on the same line
#                     tqdm.write("", end="\r")

#     # Calculate the mode for each pixel
#     mode_result = np.argmax(class_counter, axis=0).astype(np.float32)
                    
#     # Calculate the mode for each pixel
#     # mode_result, count = mode(class_counter, axis=0)
#     # mode_result = mode_result.squeeze().astype(np.float32)

#     # Save the final prediction to the output raster
#     with rasterio.open(save_path, 'w', **seg_kwds) as seg_file:
#         seg_file.write(mode_result, indexes=1)

def process_image(np_image, model, save_path, block_size=256, num_passes=1, max_offset=0, overlap_ratio=0):
    _, height, width = np_image.shape
    if overlap_ratio == 0:
        overlap_size = int(block_size * 1)
    else:
        overlap_size = int(block_size * overlap_ratio)

    # Initialize the prediction counter
    prediction_count = np.zeros((model.num_classes, height, width), dtype=np.float32)
    #prediction_count = np.zeros((1, height, width), dtype=np.float32)

    for n in range(num_passes):
        print(f'Pass {n+1} of {num_passes}')
        row_offset = random.randint(0, max_offset)
        col_offset = random.randint(0, max_offset)

        for row in tqdm(range(row_offset, height - block_size + 1, overlap_size), desc="Rows"):
            for col in tqdm(range(col_offset, width - block_size + 1, overlap_size), desc="Cols", leave=False):
                # Extract the window
                img_window = np_image[:, row:row+block_size, col:col+block_size]
                #img_window = pad_to_size(img_window, block_size, block_size)

                # Preprocessing (assuming pre_process function exists and is adapted for numpy arrays)
                img_opt, img_lidar, img_rad = pre_process(img_window)

                # Send to device, assuming preprocessing returns torch tensors
                img_opt = img_opt.to(device)
                img_lidar = img_lidar.to(device)
                img_rad = img_rad.to(device)

                # Model prediction
                #model.eval()
                with torch.no_grad():
                    batch_pred = model(img_opt, img_lidar, img_rad)
                    pred = batch_pred[0].detach().cpu().numpy()

                # Accumulate the predictions
                for band in range(model.num_classes):
                    prediction_count[band, row:row+block_size, col:col+block_size] += pred[band]

    # Normalize predictions by the count of predictions per pixel (if overlapping)
    final_prediction = np.argmax(prediction_count, axis=0).astype(np.float32)

                # pred = batch_pred[0]
                # pred_temp = pred.argmax(dim=0)

                # # Accumulate the predictions
                # prediction_count[0, row:row+block_size, col:col+block_size] += pred_temp.detach().cpu().numpy()

    # Normalize predictions by the count of predictions per pixel (if overlapping)
    final_prediction = np.argmax(prediction_count, axis=0).astype(np.float32)

    # Save the final prediction to the output raster
    with rasterio.open(save_path, 'w', **seg_kwds) as seg_file:
        seg_file.write(final_prediction.squeeze(), indexes=1)

# def clip_image(img, low=0, high=10000):
#     return np.clip(img, low, high)

def blank_process_image(np_image, model, block_size=256, num_passes=2, max_offset=128, overlap_ratio=0.5):
    _, height, width = np_image.shape
    if overlap_ratio == 0:
        overlap_size = int(block_size * 1)
    else:
        overlap_size = int(block_size * overlap_ratio)

    for n in range(num_passes):
        print(f'Pass {n+1} of {num_passes}')
        row_offset = random.randint(0, max_offset)
        col_offset = random.randint(0, max_offset)

        for row in tqdm(range(row_offset, height - block_size + 1, overlap_size), desc="Rows"):
            for col in tqdm(range(col_offset, width - block_size + 1, overlap_size), desc="Cols", leave=False):
                # Extract the window
                img_window = np_image[:, row:row+block_size, col:col+block_size]

                # Preprocessing (assuming pre_process function exists and is adapted for numpy arrays)
                img_opt, img_lidar, img_rad = pre_process(img_window)

                # Send to device, assuming preprocessing returns torch tensors
                img_opt = img_opt.to(device)
                img_lidar = img_lidar.to(device)
                img_rad = img_rad.to(device)

                # Model prediction
                #model.eval()
                with torch.no_grad():
                    batch_pred = model(img_opt, img_lidar, img_rad)

def load_means_and_stdevs():
    # Loading means and stdevs
    sen2_means = np.concatenate((np.load('stats/estrie/estrie_sen2_ete_means.npy'), np.load('stats/estrie/estrie_sen2_pri_means.npy')))
    sen1_means = np.concatenate((np.load('stats/estrie/estrie_sen1_ete_means.npy'), np.load('stats/estrie/estrie_sen1_pri_means.npy')))
    lidar_means = np.load('stats/estrie/estrie_lidar_means_v2.npy')

    sen2_stdevs = np.concatenate((np.load('stats/estrie/estrie_sen2_ete_stds.npy'), np.load('stats/estrie/estrie_sen2_pri_stds.npy')))
    sen1_stdevs = np.concatenate((np.load('stats/estrie/estrie_sen1_ete_stds.npy'), np.load('stats/estrie/estrie_sen1_pri_stds.npy')))
    lidar_stdevs = np.load('stats/estrie/estrie_lidar_stds_v2.npy')

    return [sen2_means, sen1_means, lidar_means], [sen2_stdevs, sen1_stdevs, lidar_stdevs]

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

def pre_process(win_data): 
        sen2_ete_img = win_data[:12,:,:].squeeze()
        sen2_print_img = win_data[12:24,:,:].squeeze()

        img_opt = np.vstack((sen2_ete_img, sen2_print_img))

        img_rad = win_data[24:30,:,:].squeeze()

        img_lidar = win_data[30:37,:,:].squeeze()

        lidar_band_indices = {'mnt': 0, 'mhc': 1, 'pentes': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
        lidar_band_indices_selected = [lidar_band_indices[band] for band in params['lid_bands']]
        img_lidar = img_lidar[lidar_band_indices_selected,:,:]

        if img_lidar.dtype != 'float32':
            img_lidar = np.float32(img_lidar) # Only for overlapping dataset #TODO
        else:
            pass

        mean_lst, stdev_lst = load_means_and_stdevs()
        

        #Loads means and stds for standardization according to selected bands
        if mean_lst is not None and stdev_lst is not None:
            sen2_means = torch.tensor(mean_lst[0])
            sen2_stdevs = torch.tensor(stdev_lst[0])

            if params['lid_bands']  is not None:
                # Extract means and stds for selected LiDAR bands
                lidar_band_indices = {'mnt': 0, 'mhc': 1, 'pentes': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
                lidar_means = torch.tensor([mean_lst[2][lidar_band_indices[band]] for band in params['lid_bands'] ])
                lidar_stdevs = torch.tensor([stdev_lst[2][lidar_band_indices[band]] for band in params['lid_bands'] ])
            else:
                lidar_means = torch.tensor(mean_lst[2])
                lidar_stdevs = torch.tensor(stdev_lst[2])

            # No bands selection for radar
            sen1_means = torch.tensor(mean_lst[1])
            sen1_stdevs= torch.tensor(stdev_lst[1])

        else:
            print('No means or stds available for standardization')
        #Cast to tensor
        img_opt = torch.from_numpy(img_opt)
        img_rad = torch.from_numpy(img_rad)
        img_lidar = torch.from_numpy(img_lidar)

        if mean_lst is not None and stdev_lst is not None:
            img_opt = img_opt.sub_(sen2_means[:, None, None]).div_(sen2_stdevs[:, None, None])
            img_rad = img_rad.sub_(sen1_means[:, None, None]).div_(sen1_stdevs[:, None, None])
            #img_rad = img_rad.sub_(self.sar_means[:, None, None]).div_(self.sar_stdevs[:, None, None])
            img_lidar = img_lidar.sub_(lidar_means[:, None, None]).div_(lidar_stdevs[:, None, None])

        if len(params['indices_lst']) > 0:
            sen2_ete_img = torch.from_numpy(sen2_ete_img.copy())
            sen2_print_img = torch.from_numpy(sen2_print_img.copy())

            ete_indices = {key: value for key, value in calculate_indices(sen2_ete_img, params['opt_bands']).items() if key in params['indices_lst']}
            print_indices = {key: value for key, value in calculate_indices(sen2_print_img, params['opt_bands']).items() if key in params['indices_lst']}

            # Concatenate images and their indices
            ete_img_indices = torch.cat([torch.from_numpy(ete_indices[key]).unsqueeze(0) for key in ete_indices], dim=0)
            print_img_indices = torch.cat([torch.from_numpy(print_indices[key]).unsqueeze(0) for key in print_indices], dim=0)
            #img_opt = torch.cat((img_opt, ete_img_indices, print_img_indices), dim=0)

        opt_bands_s1s2 = sorted(params['opt_bands'] + [x+12 for x in params['opt_bands']])
        img_opt = img_opt[opt_bands_s1s2,:,:]

        img_opt = torch.cat((img_opt, ete_img_indices, print_img_indices), dim=0)

        # Unsqueeze to add batch dimension and simulate batch of 1
        img_opt = img_opt.unsqueeze(0)
        img_lidar = img_lidar.unsqueeze(0)
        img_rad = img_rad.unsqueeze(0)

        # DEBUG FOR INTERMEDIATE OUTPUT BEFORE MODEL
        # import rasterio 
        # from rasterio.transform import from_origin

        # metadata = {
        #     'driver': 'GTiff',
        #     'dtype': 'float32',
        #     'nodata': None,
        #     'width': 256,
        #     'height': 256,
        #     'count': 22,
        #     'crs': rasterio.crs.CRS.from_epsg(32198),  # Updated EPSG code
        #     'transform': from_origin(-180, 90, 0.1, 0.1)
        #     }

        # #img_lidar = img_lidar.detach().cpu().numpy()
        # img_opt = img_opt.squeeze()

        # with rasterio.open('results/debug/img_opt.tif', 'w', **metadata) as dst:
        #     for band in range(img_opt.shape[0]):
        #         dst.write(img_opt[band], band+1)  # Write each band

        return img_opt, img_lidar, img_rad

# def pre_process(win_data): 
#         sen2_ete_img = win_data[:12,:,:].squeeze().copy()
#         sen2_print_img = win_data[12:24,:,:].squeeze().copy()

#         img_opt = np.vstack((sen2_ete_img, sen2_print_img))

#         img_rad = win_data[24:30,:,:].squeeze().copy()

#         img_lidar = win_data[30:37,:,:].squeeze().copy()

#         lidar_band_indices = {'mnt': 0, 'mhc': 1, 'pentes': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
#         lidar_band_indices_selected = [lidar_band_indices[band] for band in params['lid_bands']]
#         img_lidar = img_lidar[lidar_band_indices_selected,:,:]

#         if img_lidar.dtype != 'float32':
#             img_lidar = np.float32(img_lidar) # Only for overlapping dataset #TODO
#         else:
#             pass

#         mean_lst, stdev_lst = load_means_and_stdevs()
        

#         #Loads means and stds for standardization according to selected bands
#         if mean_lst is not None and stdev_lst is not None:
#             sen2_means = torch.tensor(mean_lst[0])
#             sen2_stdevs = torch.tensor(stdev_lst[0])

#             if params['lid_bands']  is not None:
#                 # Extract means and stds for selected LiDAR bands
#                 lidar_band_indices = {'mnt': 0, 'mhc': 1, 'pentes': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
#                 lidar_means = torch.tensor([mean_lst[2][lidar_band_indices[band]] for band in params['lid_bands'] ])
#                 lidar_stdevs = torch.tensor([stdev_lst[2][lidar_band_indices[band]] for band in params['lid_bands'] ])
#             else:
#                 lidar_means = torch.tensor(mean_lst[2])
#                 lidar_stdevs = torch.tensor(stdev_lst[2])

#             # No bands selection for radar
#             sen1_means = torch.tensor(mean_lst[1])
#             sen1_stdevs= torch.tensor(stdev_lst[1])

#         else:
#             print('No means or stds available for standardization')
#         #Cast to tensor
#         img_opt = torch.from_numpy(img_opt)
#         img_rad = torch.from_numpy(img_rad)
#         img_lidar = torch.from_numpy(img_lidar)

#         if mean_lst is not None and stdev_lst is not None:
#             img_opt = img_opt.sub_(sen2_means[:, None, None]).div_(sen2_stdevs[:, None, None])
#             img_rad = img_rad.sub_(sen1_means[:, None, None]).div_(sen1_stdevs[:, None, None])
#             #img_rad = img_rad.sub_(self.sar_means[:, None, None]).div_(self.sar_stdevs[:, None, None])
#             img_lidar = img_lidar.sub_(lidar_means[:, None, None]).div_(lidar_stdevs[:, None, None])

#         if len(params['indices_lst']) > 0:
#             sen2_ete_img = torch.from_numpy(sen2_ete_img.copy())
#             sen2_print_img = torch.from_numpy(sen2_print_img.copy())

#             ete_indices = {key: value for key, value in calculate_indices(sen2_ete_img, params['opt_bands']).items() if key in params['indices_lst']}
#             print_indices = {key: value for key, value in calculate_indices(sen2_print_img, params['opt_bands']).items() if key in params['indices_lst']}

#             # Concatenate images and their indices
#             ete_img_indices = torch.cat([torch.from_numpy(ete_indices[key]).unsqueeze(0) for key in ete_indices], dim=0)
#             print_img_indices = torch.cat([torch.from_numpy(print_indices[key]).unsqueeze(0) for key in print_indices], dim=0)
#             #img_opt = torch.cat((img_opt, ete_img_indices, print_img_indices), dim=0)

#         opt_bands_s1s2 = sorted(params['opt_bands'] + [x+12 for x in params['opt_bands']])
#         img_opt = img_opt[opt_bands_s1s2,:,:]

#         img_opt = torch.cat((img_opt, ete_img_indices, print_img_indices), dim=0)

#         # Unsqueeze to add batch dimension and simulate batch of 1
#         img_opt = img_opt.unsqueeze(0)
#         img_lidar = img_lidar.unsqueeze(0)
#         img_rad = img_rad.unsqueeze(0)

#         return img_opt, img_lidar, img_rad

def load_config(file_path):
    config = OmegaConf.load(file_path)
    return config

# Output
save_path_root = '/mnt/SN750/01_Code_nvme/Master_2024/results/inference/'

# Models dictionnary
all_bands = [x for x in range(12)]

models = {
    'm222': {
            'chkp_path': 'lightning_logs/version_222/checkpoints/epoch=193-step=176152.ckpt',
            'num_layers': 4, 
            'opt_bands':[1, 2, 3, 4, 5, 6, 7, 8, 10, 11], 
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'], 
            'indices_lst': ['ndvi']
            },
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
    'm129': {
            'chkp_path': '/mnt/f/01_Code_nvme/Master_2024/lightning_logs/version_129/checkpoints/topk_epoch-epoch=47-val_loss=0.53.ckpt',
            #'chkp_path': '/mnt/f/01_Code_nvme/Master_2024/lightning_logs/version_129/checkpoints/epoch=74-step=71925.ckpt',
            'num_layers': 4,
            'opt_bands': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
            'lid_bands': ['mhc', 'pentes', 'tpi', 'tri', 'twi'],
            'rad_bands': [0,1,2],
            'indices_lst': ['ndvi'],
            'architecture': 'unet_3enco_concat',
            'model': 'model_multi'
            },
    'm132': {
            #'chkp_path': '/mnt/f/01_Code_nvme/Master_2024/lightning_logs/version_132/checkpoints/epoch=49-step=11450.ckpt',
            'chkp_path': '/mnt/f/01_Code_nvme/Master_2024/lightning_logs/version_132/checkpoints/topk_epoch-epoch=24-val_loss=0.28.ckpt',
            'num_layers': 4,
            'opt_bands': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
            'lid_bands': ['mnt', 'mhc', 'pentes', 'tpi', 'tri', 'twi'],
            'rad_bands': [0,1,2],
            'indices_lst': ['ndvi', 'ndwi', 'ndmi', 'mbwi'],
            'architecture': 'unet_3enco_concat',
            'model': 'modelbin_no_overlap',
            'num_class': 2,
            },
}

selected_model = 'm129'
params = models[selected_model]
region_preds = 'estrie' # 'estrie', 'portneuf_sud', 'portneuf_nord'
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
region_model = 'estrie_2024_124_5pass_mode_new_test'
model_name = 'region_preds_{}_model_{}_{}_{}'.format(region_preds, selected_model, region_model, current_time)

if region_preds == 'estrie':
    #img_path = '/mnt/f/01_Code_nvme/Master_2024/results/full_stack/stack_estrie_3m_test_v04.tif'
    #img_path = '/mnt/f/01_Code_nvme/Master_2024/results/full_stack/stack_estrie_3m_test_v04_2024.tif'
    ### Zone test v04 ###

    main_path = '/mnt/d/00_Donnees/02_maitrise/01_trainings/estrie/full_image_zone_test/v_04'

    # load sen2 ete print
    # paths
    sen2_e_path = os.path.join(main_path, 's2_estrie_3m_ete_test_v04.tif')
    sen2_p_path = os.path.join(main_path, 's2_estrie_3m_prin_test_v04.tif')

    # load
    sen2_e_img = np.array(tiff.imread(sen2_e_path), dtype=np.float32)
    sen2_p_img = np.array(tiff.imread(sen2_p_path), dtype=np.float32)
    sen2_e_img = sen2_e_img.transpose(2,0,1)
    sen2_p_img = sen2_p_img.transpose(2,0,1)
    print('done sen2')

    # load sentinel1
    # paths
    sen1_e_path = os.path.join(main_path, 's1_estrie_3m_ete_test_v04.tif')
    sen1_p_path = os.path.join(main_path, 's1_estrie_3m_prin_test_v04.tif')

    # load
    sen1_e_img = np.array(tiff.imread(sen1_e_path), dtype=np.float32)
    sen1_p_img = np.array(tiff.imread(sen1_p_path), dtype=np.float32)
    sen1_e_img = sen1_e_img.transpose(2,0,1)
    sen1_p_img = sen1_p_img.transpose(2,0,1)

    print('done sen1')

    # load lidar 
    # paths
    mnt_path = os.path.join(main_path, 'mnt_estrie_3m_test_v04.tif') 
    mhc_path = os.path.join(main_path, 'mhc_estrie_3m_test_v04.tif')
    slo_path = os.path.join(main_path, 'pente_estrie_3m_test_v04.tif')
    tpi_path = os.path.join(main_path, 'tpi_estrie_3m_test_v04.tif')
    tri_path = os.path.join(main_path, 'tri_estrie_3m_test_v04.tif')
    twi_path = os.path.join(main_path, 'twi_estrie_3m_test_v04.tif')

    img_mnt = np.array(tiff.imread(mnt_path))
    img_mnt = np.expand_dims(img_mnt, axis=0)

    img_mhc = np.array(tiff.imread(mhc_path))
    img_mhc = np.expand_dims(img_mhc, axis=0)

    img_slopes = np.array(tiff.imread(slo_path))
    img_slopes = np.expand_dims(img_slopes, axis=0)

    img_tpi = np.array(tiff.imread(tpi_path))
    img_tpi = np.expand_dims(img_tpi, axis=0)

    img_tri = np.array(tiff.imread(tri_path))
    img_tri = np.expand_dims(img_tri, axis=0)

    img_twi = np.array(tiff.imread(twi_path))
    img_twi = np.expand_dims(img_twi, axis=0)

    full_img = np.vstack((sen2_e_img, sen2_p_img, sen1_e_img, sen1_p_img, img_mnt, img_mhc, img_slopes, img_tpi, img_tri, img_twi))

    with rasterio.open(sen2_e_path) as img_file:
        img_meta = img_file.meta

        # Create the output segmentation raster with the same properties as the input image
        seg_kwds = img_meta.copy()
        seg_kwds.update({'count': 1, 'dtype': np.float32})

elif region_preds == 'portneuf_sud':
    img_path = '/mnt/f/01_Code_nvme/Master_2024/results/full_stack/stack_portneuf_3m_zone2.tif' # Enregistrer les anciens dans oneNote au cas
elif region_preds == 'portneuf_nord':
    img_path = '/mnt/f/01_Code_nvme/Master_2024/results/full_stack/portneuf_nord_stack_inference.tif'

# Model parameters (load automatically from checkpoint)
num_layers_main = params['num_layers']
# lr_main = 0.001
input_channel_main = (len(params['opt_bands']) + len(params['indices_lst']))*2
input_channel_lidar = len(params['lid_bands'])
input_channel_radar = 6

num_classes = params.get('num_class', 9)
if num_classes == 2:
    num_classes = 2
else:
    num_classes = 9

conf_model = params['model']
config_file_path = f'conf/model/{conf_model}.yaml'
config = load_config(config_file_path)
config.architecture = params['architecture']

SEED = 1234
seed_everything(SEED, workers=True)

# Normal loading of model 
model = SemSegment.load_from_checkpoint(
    checkpoint_path=params['chkp_path'],
    cfg=config,
    class_weights=None,
    )



model.cuda()
#model.eval()

# Fonctionne, mais est-ce que c<est la meme chose que juste pas mettre .eval()
# for m in model.modules():
#     if isinstance(m, torch.nn.BatchNorm2d):
#         #m.track_running_stats = False
#         m.train()
#         m.momentum = 0.0

# count = 0
# for m in model.modules():
#     if isinstance(m, torch.nn.BatchNorm2d):
#         # count += 1 #skip the first BatchNorm layer in my ResNet50 based encoder
#         # if count >= 2:
#             m.eval()
#             m.weight.requires_grad = False
#             m.bias.requires_grad = False

save_path = save_path_root + model_name + '.tif'

model.train()
blank_process_image(full_img, model, block_size=256, num_passes=10, max_offset=128, overlap_ratio=0.5)

model.eval()
process_image(full_img, model, save_path, block_size=256, num_passes=2, max_offset=128, overlap_ratio=0.5)