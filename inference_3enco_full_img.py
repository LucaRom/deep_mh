from argparse import ArgumentParser

import logging
import time
from sklearn import multiclass
import torch
import torchmetrics
from torchmetrics import ConfusionMatrix
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import rasterio
from rasterio import windows
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.crs import CRS

from tqdm import tqdm
from skimage.exposure import match_histograms

from itertools import product
import matplotlib.pyplot as plt

#from pl_bolts.models.vision.unet import UNet
from unet_3enco_sum import unet_3enco_sum
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor


# Custom LR
from utils import get_datasets_inference, get_datasets
from dataset import inference_on_full_image_dataset

# Custom loss
from custom_loss import FocalLoss

# Add to specified some tensor to GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class SemSegment(LightningModule):
    def __init__(
        self,
        #lr: float = 0.001,
        #num_classes: int = 19,
        num_classes: int = 8,
        #num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = True,

    ):
        """
        Adapted from the implementation of `Annika Brundyn <https://github.com/annikabrundyn>` in PyTorch lightning

        Args:
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()
    
        self.num_classes = num_classes
        self.num_layers = num_layers_main
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr_main
        self.new_time = time.time()
        self.train_time_list = []

        # Metrics 
        # self.train_accuracy = torchmetrics.Accuracy(mdmc_average='samplewise')
        # self.val_accuracy = torchmetrics.Accuracy(mdmc_average='samplewise')
        # self.train_f1 = torchmetrics.F1Score(mdmc_average='samplewise')
        # self.val_f1 = torchmetrics.F1Score(mdmc_average='samplewise')

        # Model
        self.net = unet_3enco_sum(
            num_classes=num_classes,
            input_channels=input_channel_main,
            input_channels_lidar=input_channel_lidar,
            input_channels_radar=input_channel_radar,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    # def forward(self, x):
    #     return self.net(x)

    def forward(self, x, y, z):
        return self.net(x, y, z)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.trainer.model.eval()

        #img, lidar, mask, radar, img_path = batch 
        #img, lidar, radar, img_path = batch 



        # # stack both sentinel 2 images
        # img_opt = np.dstack((sen2_ete_img, sen2_print_img))

        # #img_lidar = np.dstack((img_mnt, img_mhc, img_slopes, img_tpi, img_tri, img_twi))
        # img_lidar = np.dstack((img_mhc, img_slopes, img_tpi, img_tri, img_twi))

        # # if img_lidar.dtype != 'float32':
        # #     img_lidar = np.float32(img_lidar) # Only for overlapping dataset #TODO
        # # else:
        # #     pass


        # img_rad = np.dstack((sen1_ete_img, sen1_print_img)) # stack both sen-1 images

        # # Cast to tensor for better permute
        # img_opt = torch.from_numpy(img_opt)
        # img_opt = img_opt.permute(2,0,1)

        # # Apply standardization (see : discuss.pytorch.org/t/how-to-normalize-multidimensional-tensor/65304)
        # #img_opt = img_opt.sub_(combined_mean[:, None, None]).div_(combined_std[:, None, None])

        # k_lidar_means = torch.tensor([13.348262, 13.45669, -0.006740755, -3.689763, 5.7766604])
        # k_lidar_stds  = torch.tensor([7.7406297, 13.942361, 1.3129127, 241.4134, 5.6496654])

        # img_lidar = torch.from_numpy(img_lidar)
        # img_lidar = img_lidar.permute(2,0,1)

        # # Standardization
        # img_lidar = img_lidar.sub_(k_lidar_means[:, None, None]).div_(k_lidar_stds[:, None, None])

        # img_rad = torch.from_numpy(img_rad)
        # img_rad = img_rad.permute(2,0,1)

        # img = img_opt.float()   # x
        # lidar = img_lidar.float()
        # #mask = mask.long()  # y 
        # radar = img_rad.float() # z

        img = bands[0:23]
        radar = bands[24:29]
        lidar = bands[30:34]

        preds = self(img, lidar, radar)   # predictions

        # preds_temp   = preds.argmax(dim=1).unsqueeze(1)
        # preds_recast = preds_temp.type(torch.IntTensor).to(device=device)     

        # confmat = ConfusionMatrix(num_classes=8).to(device=device)
        # conf_print = confmat(preds_recast, mask)
       
        #return {'conf matrice': conf_print, 'preds' : preds, 'img' : img, 'lidar' : lidar, 'mask' : mask, 'radar' : radar, 'img_path' : img_path}
        #return {'preds' : preds, 'img' : img, 'lidar' : lidar, 'radar' : radar, 'img_path' : img_path}
        
    # @torch.no_grad()
    # def test_epoch_end(self, outputs):
    #     # TODO Add logs to test aswell?

    #     for x in range(len(outputs)):
    #         fig = plt.figure()
    #         cm = outputs[x]['conf matrice'].cpu().numpy()
    #         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #         disp.plot()
    #         plt.savefig("lightning_logs/inference_{version}/cm_{num}.png".format(version = log_version, num = x))
    #         plt.close(fig)

    #         # Extract CRS and transforms
    #         img_path = outputs[x]['img_path']
    #         src = rasterio.open(img_path[0])
    #         sample_crs = src.crs
    #         transform_ori = src.transform
    #         src.close() # Needed?

    #         ori_input = outputs[x]['img'][0].cpu().numpy()
    #         ori_target = outputs[x]['mask'][0].cpu().numpy()
    #         predict_sig = outputs[x]['preds'][0].cpu().argmax(dim=0).numpy().astype(np.int32)

    #         # write predict image to file
    #         tiff_save_path = "lightning_logs/inference_{version}/predict_geo_{num}.tif".format(version = log_version, num = x)
    #         #tiff_save_path = "lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = log_version, num = x)

    #         predict_img = rasterio.open(tiff_save_path, 'w', driver='GTiff',
    #                         height = input_tile_size, width = input_tile_size,
    #                         count=1, dtype=str(predict_sig.dtype),
    #                         crs=sample_crs,
    #                         transform=transform_ori)

    #         predict_img.write(predict_sig, 1)
    #         predict_img.close()

    #         fig = plt.figure()
    #         plt.subplot(1,3,1)
    #         plt.imshow(np.transpose(ori_input[[3,2,1],:,:],(1,2,0))*3)
    #         plt.title("Input")
    #         plt.subplot(1,3,2)
    #         plt.imshow(predict_sig)
    #         plt.title("Predict")
    #         plt.subplot(1,3,3)
    #         plt.imshow(ori_target)
    #         plt.title("Target")

    #         plt.savefig("lightning_logs/inference_{version}/fig_{num}.png".format(version = log_version, num = x))
    #         plt.close(fig)

    #     self.trainer.model.train()

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
    #     parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
    #     parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
    #     parser.add_argument(
    #         "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
    #     )

    #     return parser

# Utils fonction outside of main class
# Windowed iteration inspired by : https://gis.stackexchange.com/questions/396891/parsing-through-a-sentinel-2-tile
## TODO Move create stack to utilities 
def create_stacked_image():
    file_list = [path_to_full_sen2_ete, path_to_full_sen2_print, path_to_full_sen1_ete, path_to_full_sen1_print, path_to_full_mhc, path_to_full_slopes, path_to_full_tpi, path_to_full_tri, path_to_full_twi]
    nb_of_layers = 35

    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    #meta.update(count = len(file_list))
    meta.update(count = nb_of_layers)

    # Read each layer and write it to stack
    # with rasterio.open('stack.tif', 'w', **meta) as dst:
    #     for id, layer in enumerate(file_list, start=1):
    #         print(id, layer)
    #         with rasterio.open(layer) as src1:
    #             dst.write_band(id, src1.read(1))

    with rasterio.open('/mnt/SN750/stack_with_STD_HM_v2.tif', 'w', **meta) as dst:
        id = 1
        for images in file_list:
            print("Writing image from : ", images)
            with rasterio.open(images) as src_tmp:
                array = src_tmp.read(out_dtype='float32')
                for bands in array:
                    print("Writing bands no : ", id)
                    dst.write_band(id, bands)
                    id += 1

def iter_windows(src_ds, width, height, boundless=False):
    offsets = product(range(0, src_ds.meta['width'], width), range(0, src_ds.meta['height'], height))
    big_window = windows.Window(col_off=0, row_off=0, width=src_ds.meta['width'], height=src_ds.meta['height'])
    for col_off, row_off in offsets:

        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

        if boundless:
            yield window
        else:
            yield window.intersection(big_window)

# def iter_windows_no_meta(src_ds, width, height, boundless=False):

#     offsets = product(range(0, src_ds.shape[1], width), range(0, src_ds.shape[2], height))
#     big_window = windows.Window(col_off=0, row_off=0, width=src_ds.shape[1], height=src_ds.shape[2])
#     for col_off, row_off in offsets:

#         window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

#         if boundless:
#             yield window
#         else:
#             yield window.intersection(big_window)

# From : https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
# Input image shape should be C x H x W

def iter_windows_no_meta(src_ds, stepsize, height, width):
    for y in range(0, src_ds.shape[1], stepsize):
        for x in range(0, src_ds.shape[2], stepsize):
            yield (x, y, src_ds[:, y:y + height, x:x + width])

def inference_from_ckpt(ckpt_path, bands):
    model = SemSegment.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    )

    #trainer = Trainer(accelerator='gpu', devices=1)
    #trainer.test(model)
    #trainer.predict(model, dataloaders=None)
    #img_sen2_print = 


    # Input need shape (B X C X H X W) where B is the batchsize, that is why we unsqueeze to have B = 1
    img   = torch.from_numpy(bands[0:24]).unsqueeze(0)
    radar = torch.from_numpy(bands[24:30]).unsqueeze(0)

    lidar = torch.from_numpy(bands[30:35])

    # Temporary std and means for lidar
    k_lidar_means = torch.tensor([13.348262, 13.45669, -0.006740755, -3.689763, 5.7766604])
    k_lidar_stds  = torch.tensor([7.7406297, 13.942361, 1.3129127, 241.4134, 5.6496654])

    estrie_lidar_mean = torch.tensor([7.798849, 5.5523205, 0.0029951811, 0.06429929, 6.7409873])
    estrie_lidar_std  = torch.tensor([7.033332, 5.196636, 1.0641352, 0.06102526, 3.182435])

    lidar = lidar.sub_(k_lidar_means[:, None, None]).div_(k_lidar_stds[:, None, None])
    #lidar = lidar.sub_(estrie_lidar_mean[:, None, None]).div_(estrie_lidar_std[:, None, None])

    lidar = lidar.unsqueeze(0)

    model.eval()
    pred = model(img, lidar, radar)

    #pred_temp   = pred.argmax(dim=1).unsqueeze(1)
    #pred_recast = pred_temp.type(torch.IntTensor).to(device=device)     

    predict_sig = pred[0].detach().argmax(dim=0) #.numpy().astype(np.int16)
    #predict_sig = pred.detach().argmax(dim=1) #.numpy() #.astype(np.int16)

    return predict_sig

# #TODO move to utils when clean
# def histo_matching(target_arr, ref_arr):


#     ar_k = match_histograms(target_arr, ref_arr channel_axis=0)


if __name__ == "__main__":
    # Activate logging
    logging.basicConfig()

    # Optional timer activation
    #start_time_glob = time.time()

    # Model parameters
    num_layers_main = 4
    lr_main = 0.001
    input_channel_main = 24
    input_channel_lidar = 5
    input_channel_radar = 6

    ## TODO Move create stack to utilities 
    # Paths to original data (no standardization or histogram matching)
    # path_to_full_sen2_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/ete/s2_kenauk_3m_ete_aout2020.tif'
    # path_to_full_sen2_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen2/print/S2_de_kenauk_3m_printemps_mai2020.tif' 
    # path_to_full_sen1_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/ete/s1_kenauk_3m_ete_aout2020.tif' 
    # path_to_full_sen1_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/sen1/print/S1_kenauk_3m_printemps_mai2020.tif' 
    # path_to_full_mhc        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mhc_kenauk_3m.tif' 
    # path_to_full_slopes     = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/pentes_kenauk_3m.tif' 
    # path_to_full_tpi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tpi_kenauk_3m.tif' 
    # path_to_full_tri        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif' 
    # path_to_full_twi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/twi_kenauk_3m.tif' 

    # Paths to standardize and/or histogram matched dataa
    path_to_full_sen2_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s2_kenauk_3m_ete_HMe_STD.tif' 
    path_to_full_sen2_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s2_kenauk_3m_print_HMe_STD.tif' 
    path_to_full_sen1_ete   = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s1_kenauk_3m_ete_STD.tif' 
    path_to_full_sen1_print = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/raw_standard/s1_kenauk_3m_print_STD.tif' 
    path_to_full_mhc        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/mhc_kenauk_3m.tif' 
    path_to_full_slopes     = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/pentes_kenauk_3m.tif' 
    path_to_full_tpi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tpi_kenauk_3m.tif' 
    path_to_full_tri        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/tri_kenauk_3m.tif' 
    path_to_full_twi        = '/mnt/Data/00_Donnees/02_maitrise/01_trainings/kenauk/processed_raw/lidar/twi_kenauk_3m.tif' 

    #create_stacked_image()

    # with rasterio.open(path_to_full_tri) as dst:
    #     array = dst.read(out_dtype='float32')
    #     for bands in array:
    #         print(bands)

    #     # id = 1
    #     # for images in file_list:
    #     #     print("Writing image from : ", images)
    #     #     with rasterio.open(images) as src_tmp:
    #     #         array = src_tmp.read(out_dtype='float32')
    #     #         for bands in array:
    #     #             print("Writing bands no : ", id)
    #     #             dst.write_band(id, bands)
    #     #             id += 1

    size = 256
    path_to_stack_img = '/mnt/SN750/stack_with_STD_HM_v2.tif'

    # #TODO automatiser les paths
    # # Evaluate 
    ckpt_path = "/mnt/Data/01_Codes/00_Github/Unet_lightning/lightning_logs/version_182/checkpoints/epoch=48-step=41797.ckpt"
    #ckpt_path = "/mnt/Data/01_Codes/00_Github/Unet_lightning/lightning_logs/version_182/checkpoints/epoch=46-step=40091.ckpt"
    log_version = "182_kenauk_2020_4"

    #With premade stack 
    logging.info('Starting prediction on full image')
    with rasterio.open(path_to_stack_img) as ds:
        profile = ds.profile
        profile['count'] = 1  # assume output is a single band raster
        with rasterio.open(path_to_stack_img.replace(".tif", "_clc_nometa_std_hm_sen2_tpi_correct_v4_estrie.tif"), "w", **profile) as out_ds:
            #for window in tqdm(iter_windows(ds, size, size), total=len(list(iter_windows(ds, size, size))), desc='Predicting and creating output'):
            for window in tqdm(iter_windows(ds, size, size), total=len(list(iter_windows(ds, size, size))), desc='Predicting and creating output'):
                bands = ds.read(window=window, out_dtype='float32')
                tile = inference_from_ckpt(ckpt_path, bands)
                out_ds.write(tile, 1, window=window)
    logging.info('Prediction finished')


    ## With loading data directly
    # profile_temp = {'driver': 'GTiff', 
    #                 'dtype': 'uint16', 
    #                 'nodata': None, 
    #                 'width': 7836, 
    #                 'height': 6101, 
    #                 'count': 1, 
    #                 'crs': CRS.from_epsg(32198), 
    #                 'transform': Affine(3.0, 0.0, -544994.0388, 0.0, -3.0, 234380.8231), 
    #                 'tiled': False, 'interleave': 'pixel'}


    # logging.info('Starting prediction on full image')

    # with rasterio.open(path_to_stack_img) as ds:
    #     # sentinel 2 images
    #         sen2_ete_img   = ds.read((1,2,3,4,5,6,7,8,9,10,11,12), out_dtype='float32')
    #         sen2_print_img = ds.read((13,14,15,16,17,18,19,20,21,22,23,24), out_dtype='float32')

    #         # sentinel-1 images
    #         sen1_ete_img   = ds.read((25,26,27), out_dtype='float32')
    #         sen1_print_img = ds.read((28,29,30), out_dtype='float32')

    #         # lidar images
    #         img_mhc    = np.expand_dims(ds.read((31), out_dtype='float32'), 0)
    #         img_slopes = np.expand_dims(ds.read((32), out_dtype='float32'), 0)
    #         img_tpi    = np.expand_dims(ds.read((33), out_dtype='float32'), 0)
    #         img_tri    = np.expand_dims(ds.read((34), out_dtype='float32'), 0)
    #         img_twi    = np.expand_dims(ds.read((35), out_dtype='float32'), 0)

    # full_array = np.concatenate((sen2_ete_img, sen2_print_img, sen1_ete_img, sen1_print_img, img_mhc, img_slopes, img_tpi, img_tri, img_twi))

    # profile = profile_temp
    # #profile['count'] = 1  # assume output is a single band raster

    # with rasterio.open(path_to_stack_img.replace(".tif", "_clc_nometa_std_hm_sen2_lidar.tif"), "w", **profile) as out_ds:
    #     #for window in tqdm(iter_windows(ds, size, size), total=len(list(iter_windows(ds, size, size))), desc='Predicting and creating output'):
    #     for window in tqdm(iter_windows_no_meta(full_array, size, size, size), total=len(list(iter_windows_no_meta(full_array, size, size, size))), desc='Predicting and creating output'):
    #         #bands = ds.read(window=window, out_dtype='float32')
    #         tile = inference_from_ckpt(ckpt_path, window[2])
    #         out_ds.write(tile, 1, window=Window(window[0], window[1], window[2].shape[2], window[2].shape[1]))

    # logging.info('Prediction finished')
