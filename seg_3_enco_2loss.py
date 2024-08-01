from argparse import ArgumentParser
import matplotlib
from matplotlib import colors
import sys
import gc
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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import rasterio
from tqdm import tqdm
import os
import datetime
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import hydra

#from pl_bolts.models.vision.unet import UNet
from unet_models.unet_3enco_concat_bin_mul import unet_3enco_concat_bin_mul
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, EarlyStopping

# Remove num_workers warning during training
# TODO test num_workers further
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Remove divide warning during training, should be turn back on when calculating indices on the fly
# or having loss as NaN
print()
print("WARNING, divide and invalid error from numpy are deactivated"
      "Activate in case of error such as loss=nan")
print()
np.seterr(divide='ignore', invalid='ignore')

# Custom LR
import utils.utils

# Custom loss
from utils.custom_loss import FocalLoss

# Add to specified some tensor to GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

logger = logging.getLogger(__name__)

class SemSegment2loss(LightningModule):
    def __init__(self, cfg, class_weights=None):
        super().__init__()
        self.cfg = cfg
        self.classif_mode = cfg.classif_mode
        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_class
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        self.calculate_input_channels()
        self.init_metrics()     

        self.binary_only_epochs = cfg.bin_warmup  # Number of epochs to train binary classifier only
        
        # Initiate value list for plotting
        self.epoch_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Model initialization
        if self.cfg.architecture == 'unet_3enco_concat':
            self.net = unet_3enco_concat_bin_mul(
                num_classes=self.num_classes,
                input_channels=self.input_channels_main,
                input_channels_lidar=self.input_channels_lidar,
                input_channels_radar=self.input_channels_radar,
                features_start=self.cfg.features_start,
                bilinear=self.cfg.bilinear,
            )

        else:
            raise ValueError(f"Unrecognized architecture defined : {self.cfg.architecture}")

    def calculate_input_channels(self):
        # Base input channels base on experiment
        self.input_channels_main = (len(self.cfg.opt_bands) + len(self.cfg.indices_lst)) * 2
        self.input_channels_lidar = len(self.cfg.lidar_bands)
        self.input_channels_radar = len(self.cfg.sar_bands) * 2


    def init_metrics(self):
        # Metrics 
        '''
            Notes about what I understand with 'average' and 'mdmc_average'

            Micro average takes into account how many samples there are per category

            mdmc = multidimensional-multiclass
            mdmc is not necessary because we use argmax before feeding the metrics, thus having only
            (N,) preds input shape (https://torchmetrics.readthedocs.io/en/stable/pages/classification.html#input-types)
        '''

        if self.classif_mode == 'bin':
                self.train_accuracy = torchmetrics.Accuracy(task="binary")
                self.val_accuracy = torchmetrics.Accuracy(task="binary")
                self.train_f1 = torchmetrics.F1Score(task="binary")
                self.val_f1 = torchmetrics.F1Score(task="binary")
        else:
            self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes = self.num_classes, average = "micro")
            self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes = self.num_classes, average = "micro")
            # self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = num_classes, average = "micro")
            # self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = num_classes, average = "micro")
            self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = self.num_classes, average = "micro", ignore_index = 7)
            self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = self.num_classes, average = "micro", ignore_index = 7)

    def configure_optimizers(self):
        if self.cfg.optim_main == 'Ad':
            opt = torch.optim.Adam(self.net.parameters(), weight_decay = self.cfg.weight_decay, lr=self.cfg.lr)
        else:
            opt = torch.optim.SGD(self.net.parameters(), momentum = self.cfg.momentum, weight_decay = self.cfg.weight_decay, lr=self.cfg.lr)

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=self.cfg.rl_factor, mode='min', patience=7, verbose=True)
        
        #return [opt], [sch]

        return {'optimizer': opt, "lr_scheduler" : {'scheduler': sch, 'interval':'epoch', 'frequency': 1, 'monitor': 'val_loss'}}
        #return {'optimizer': opt}

    def forward(self, x_sar, x_optical, x_lidar):
        return self.net(x_sar, x_optical, x_lidar)


    def on_train_start(self):
        tensorboard = self.logger.experiment
        tensorboard.add_text("epochs", str(self.trainer.max_epochs))
        tensorboard.add_text("batch_size", str(self.batch_size))                                       
        tensorboard.add_text("optimizer", self.cfg.optim_main)                                       
        #tensorboard.add_text("learning_rate", str(lr_main))  
        tensorboard.add_text("learning_rate", str(self.cfg.lr))                                         
        tensorboard.add_text("layers_unet", str(self.cfg.num_layers))
        tensorboard.add_text("input_ch", str(self.input_channels_main))

        # Save config #TODO put in utils as function     
        out_root = "lightning_logs/version_{version}".format(version = self.trainer.logger.version)
        config_save_path = os.path.join(out_root, 'config_param.out')
        with open(config_save_path, 'w') as f:
            f.write('# -----------------------------------#\n')
            f.write('# Complet architecture layout        #\n')
            f.write('# -----------------------------------#\n\n')
            f.write(f'Architecture : {self.net}\n\n')
            f.write('# -----------------------------------#\n')

            f.write('# Parameters of the training session #\n')
            f.write('# -----------------------------------#\n\n')

            # Hydra's output path
            f.write(f'Hydra output path : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"\n')

            # All parameters
            f.write(f'Epochs : {self.trainer.max_epochs}\n')
            f.write(f'Optimizer : {self.cfg.optim_main}\n')
            f.write(f'Learning rate : {self.cfg.lr}\n')
            f.write(f'Batch size : {self.batch_size}\n')
            f.write(f'Layers UNet : {self.cfg.num_layers}\n')
            f.write(f'Num of optical bands  : {self.input_channels_main}\n')
            f.write(f'Optical bands  : {self.cfg.opt_bands}\n')
            f.write(f'Indices included  : {self.cfg.indices_lst}\n')
            f.write(f'Num of LiDAR bands  : {self.input_channels_lidar}\n')
            f.write(f'LiDAR bands  : {self.input_channels_lidar}\n')
            f.write(f'Num of radar bands : {self.input_channels_radar}\n')
            f.write(f'Num_workers : {self.cfg.num_workers}\n')
            f.write(f'Pin memory : {self.cfg.pin_memory}\n\n')

            f.write('# -------------------#\n')
            f.write('# Dataset parameters #\n')
            f.write('# -------------------#\n\n')


            # Dataset parameters
            # f.write(f'Train region : {self.cfg.input_format}\n')
            # f.write(f'Test region : {self.cfg.test_region }\n')
            f.write(f'Classification mode : {self.cfg.classif_mode}\n')

            # Close file
            f.close()


    def filter_ignore_index(self, preds, targets, ignore_index=-1):
        preds = preds.squeeze(1)

        # Create a mask for values that should not be ignored
        valid_mask = targets != ignore_index

        # Filter out invalid targets and corresponding predictions
        filtered_targets = targets[valid_mask]
        filtered_preds = preds[valid_mask]

        return filtered_preds, filtered_targets

    # def binary_cross_entropy_with_ignore(pred, target, ignore_index=-1):
    #     mask = (target != ignore_index)
    #     masked_pred = pred[mask]
    #     masked_target = target[mask]
    #     return torch.nn.CrossEntropyLoss(masked_pred, masked_target)

    def training_step(self, batch, batch_nb):
	    # Switch to train mode
        self.trainer.model.train()

        # Initialize default values for logging
        train_accu = torch.tensor(0.0, device=self.device)
        train_f1 = torch.tensor(0.0, device=self.device)
        multiclass_loss = torch.tensor(0.0, device=self.device)

        img, lidar, mask_bin, mask_multi, radar, img_path = batch # img_path only needed in test, but still carried
                                                                  # Batch expected output are linked from the dataset.py file

        unique_values = torch.unique(mask_multi)
        logger.debug("Unique values in a multiclass mask tensor:", unique_values)

        img = img.float()   # x
        lidar = lidar.float()
        mask_multi = mask_multi.long()  # y
        radar = radar.float()
        mask_bin = mask_bin.long()

        binary_mask_logits, preds_bin, preds_multi = self(img, lidar, radar)

        binary_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(binary_mask_logits, mask_bin)
        #binary_loss = self.binary_cross_entropy_with_ignore(binary_mask_logits, mask_bin)

        if self.current_epoch < self.binary_only_epochs:
            # Train only binary classifier
            total_loss = binary_loss
            multiclass_loss = torch.tensor(0.0, device=self.device)  # Dummy value for logging
        else:
            multiclass_loss = FocalLoss(weight=self.class_weights, ignore_index=-1)(preds_multi, mask_multi)
            total_loss = binary_loss + multiclass_loss

            preds_accu = preds_multi.argmax(dim=1).float()

            #For augmentations ignore value -1 
            if self.cfg.train_transforms:
                filtered_preds, filtered_targets = self.filter_ignore_index(preds_accu, mask_multi, ignore_index=-1)
                train_accu = self.train_accuracy(filtered_preds, filtered_targets)
                train_f1   = self.train_f1(filtered_preds, filtered_targets)
            
            else:
                # Train metrics call
                train_accu = self.train_accuracy(preds_accu, mask_multi)
                train_f1   = self.train_f1(preds_accu, mask_multi)

            # Call to log
            self.log("train_f1", train_f1.detach(), batch_size=self.batch_size)
            self.log("train_accu", train_accu.detach(), batch_size=self.batch_size)
            self.log('multiclass_loss', multiclass_loss.detach(), on_step=True, prog_bar=True, batch_size=self.batch_size)
      
        self.log("train_loss", total_loss.detach(), on_epoch=True, on_step=True, batch_size=self.batch_size)
        self.log("binary_loss", binary_loss.detach(), on_epoch=True, on_step=True, prog_bar=True, batch_size=self.batch_size)

        # logger.info(f'Multiclass loss : {multiclass_loss.detach()}')
        # logger.info(f'Train accu : {train_accu.detach()}')                 
        # logger.info(f'Binary loss : {binary_loss.detach()}')
        # logger.info(f'Train f1 : {train_f1.detach()}')
        
        # Return can be called in corresponding *epoch_end() section
        return {"loss": total_loss, "bin_loss": binary_loss, "multi_loss": multiclass}

    def training_epoch_end(self, outputs):
        train_loss_avg = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu()
        # bin_loss = torch.stack([x['bin_loss'] for x in outputs]).mean().detach().cpu()
        # multi_loss = torch.stack([x['multi_loss'] for x in outputs]).mean().detach().cpu()

        self.log('train_loss_avg', train_loss_avg.detach(), prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        # self.log('bin_loss', bin_loss.detach(), prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        # self.log('multi_loss', multi_loss.detach(), prog_bar=True, on_epoch=True, batch_size=self.batch_size)

        self.logger.experiment.add_scalars('losses', {'loss_train': train_loss_avg}, self.current_epoch)
        self.train_accuracy.reset()
        self.train_f1.reset()


    def validation_step(self, batch, batch_idx):
        self.trainer.model.eval()
                
        # val_accu = torch.tensor(0.0, device=self.device)
        # val_f1 = torch.tensor(0.0, device=self.device)

        if batch is None:
            return None 

        with torch.no_grad():
            img, lidar, mask_bin, mask_multi, radar, img_path = batch

            img = img.float()   # x
            lidar = lidar.float()
            mask_multi = mask_multi.long()  # y
            radar = radar.float()
            mask_bin = mask_bin.long()

            binary_mask_logits, preds_bin, preds_multi = self(img, lidar, radar)

            binary_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(binary_mask_logits, mask_bin)

            # Replace values in preds_multi with -1 where the ignore_mask is True
            ignore_mask = (preds_bin == 0)
            preds_multi_masked = preds_multi.clone()
            preds_multi_masked[ignore_mask.expand_as(preds_multi_masked)] = -1

            multiclass_loss = FocalLoss(weight=self.class_weights, ignore_index=-1)(preds_multi_masked, mask_multi)
            
            total_loss = binary_loss + multiclass_loss

            preds_accu = preds_multi.argmax(dim=1).unsqueeze(1)

            filtered_preds, filtered_targets = self.filter_ignore_index(preds_accu, mask_multi, ignore_index=-1)
            val_accu = self.val_accuracy(filtered_preds, filtered_targets)
            val_f1 = self.val_f1(filtered_preds, filtered_targets)

            self.log("val_loss", total_loss.detach(), batch_size=self.batch_size)
            self.log("val_f1", val_f1.detach(), batch_size=self.batch_size)
            self.log("val_accu", val_accu.detach(), batch_size=self.batch_size)

            # logger.info(f'Val loss : {total_loss.detach()}')
            # logger.info(f'Val accu : {val_f1.detach()}')
            # logger.info(f'Val f1 : {val_accu.detach()}')

        return {"val_loss": total_loss}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
        self.log('val_loss_avg', loss_val.detach(), prog_bar=True, on_epoch=True, batch_size=self.batch_size)
        self.logger.experiment.add_scalars('losses', {'loss_val': loss_val}, self.current_epoch)
        self.val_accuracy.reset()
        self.val_f1.reset()

    @torch.no_grad()
    #def test_step(self, batch, batch_idx, dataloader_idx):
    def test_step(self, batch, batch_idx):

        if batch is None:
            return None 

        # gc.collect()
        # torch.cuda.empty_cache()
        self.trainer.model.eval()

        # Save config #TODO put in utils as function     
        out_root = "lightning_logs/version_{version}".format(version = self.trainer.logger.version)
        plt.savefig(os.path.join(out_root, "cm_{version}_{mask_vers}.png".format(version = self.trainer.logger.version, mask_vers = self.cfg.test_mask_dir)))
        config_save_path = os.path.join(out_root, 'config_test_param_{mask_vers}.out'.format(mask_vers = self.cfg.test_mask_dir))
        
        with open(config_save_path, 'w') as f:
            # Hydra's output path
            f.write(f'Hydra output path : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"\n')

            # Close file
            f.close()

        with torch.no_grad():
            img, lidar, mask_bin, mask_multi, radar, img_path = batch 

            img = img.float()   # x
            lidar = lidar.float()
            mask_multi = mask_multi.long()  # y
            radar = radar.float()
            mask_bin = mask_bin.long()

            # mask_4loss = mask_multi.float().unsqueeze(1)
            # mask_bin_4loss = mask_bin.float().unsqueeze(1)

            binary_mask_logits, preds_bin, preds_multi = self(img, lidar, radar)

            preds_temp = preds_multi.argmax(dim=1)
            preds_recast = preds_temp.type(torch.IntTensor).to(device=device)     

            # Filtre pour enlever le buffer
            # Filter out negative values in the mask
            valid = mask_multi >= 0
            filtered_preds = preds_recast[valid]
            filtered_mask = mask_multi[valid]

            # Create and compute confusion matrix
            if filtered_preds.numel() > 0 and filtered_mask.numel() > 0:  # Check if there are any elements to process
                confmat = ConfusionMatrix(task='multiclass', num_classes=self.num_classes).to(device=device)
                conf_print = confmat(filtered_preds, filtered_mask)
                conf_print = conf_print.detach().cpu()
            else: 
                conf_print = torch.zeros(self.num_classes, self.num_classes).to(device=device)   # or appropriate handling if no valid data
                conf_print = conf_print.detach().cpu()

            return {'conf matrice': conf_print.detach().cpu(), 'preds' : preds_multi.detach().cpu(), 'img' : img.detach().cpu(), 'mask' : mask_multi.detach().cpu()}
        
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        self.trainer.model.eval()

        full_matrice = np.zeros((self.num_classes,self.num_classes))
        full_preds   = []
        full_targets  = []

        # Define labels
        if self.classif_mode == 'bin':
            class_labels = ['0 (NH)', '1 (MH)']
        else:
            dict_labels = utils.utils.get_project_labels()
            class_num = dict_labels.keys()
            class_labels = dict_labels.values()

        # Setting output path
        out_root = "lightning_logs/version_{version}".format(version = self.trainer.logger.version)

        # Create output folders
        print()
        print("Creating output paths")
        folders_list = ['cms', 'figs', 'tifs']
        for f in folders_list:
            f_paths = os.path.join(out_root, f)
            if os.path.isdir(f_paths):
                continue
            else:
                print("Creating path : ", f_paths)
                os.makedirs(f_paths)

        print()
        print("Generating outputs")

        for x in tqdm(range(len(outputs))):
            cm = outputs[x]['conf matrice'] #.detach().cpu().numpy()

            # Get targets
            ori_target = outputs[x]['mask'][0] #.detach().cpu().numpy()
            predict_sig = outputs[x]['preds'][0].argmax(dim=0)

            # Feed full data for overall output
            # Update full matrix directly
            full_matrice += cm.numpy() if isinstance(cm, torch.Tensor) else cm  # Conditional handling based on type
            #full_matrice += cm # Feed full matrice

            # Store minimal required data
            full_preds.append(predict_sig.numpy() if isinstance(predict_sig, torch.Tensor) else predict_sig)
            full_targets.append(ori_target.numpy() if isinstance(ori_target, torch.Tensor) else ori_target)

            # full_preds.append(predict_sig)
            # full_targets.append(ori_target)

            # Generate other wanted outputs
            if self.cfg.generate_cm_sample:             
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
                disp.plot()
                plt.savefig(os.path.join(out_root, "cms/cm_{num}.png".format(num = x)))
                plt.clf() 
                plt.close()

            if self.cfg.generate_tif_sample: 
                # Extract CRS and transforms
                # Find a way to extract path without carrying it in the batch (dataset/module output)
                img_path = outputs[x]['img_path']
                with rasterio.open(img_path[0]) as src:
                    sample_crs = src.crs
                    transform_ori = src.transform

                # write predict image to file
                tiff_save_path =  os.path.join(out_root, "tifs/preds_{version}_{num}.tif".format(version = self.trainer.logger.version, num = x))

                with rasterio.open(tiff_save_path, 'w', driver='GTiff', height=self.cfg.input_tile_size, width=self.cfg.input_tile_size, count=1, dtype=str(predict_sig.dtype), crs=sample_crs, transform=transform_ori) as predict_img:
                    predict_img.write(predict_sig, 1)
                # predict_img = rasterio.open(tiff_save_path, 'w', driver='GTiff',
                #                 height = self.cfg.input_tile_size, width = self.cfg.input_tile_size,
                #                 count=1, dtype=str(predict_sig.dtype),
                #                 crs=sample_crs,
                #                 transform=transform_ori)

                # predict_img.write(predict_sig, 1)
                # predict_img.close()

            if self.cfg.generate_fig_sample: 
                # Generating figures
                fig = plt.figure(figsize=(15, 5))
                subfig = fig.subfigures(nrows=1, ncols=1)
                axes = subfig.subplots(nrows=1, ncols=3,sharey=True)
                cmap = plt.get_cmap('tab10', self.num_classes)

                # Get original optic input for ploting
                # Fetch maximum value to see if normalization is needed in cases 
                # where we decide to not pre-normalize images in dataset
                # Search any bands that would have value over 255 in RGB
                ori_input = outputs[x]['img'][0].detach().cpu().numpy()
                ori_rgb = ori_input[[3,2,1],:,:]
                for bands in ori_rgb:
                    if max(bands[0]) > 255:
                        ori_input = ori_rgb/10000
                        break
                    else: 
                        ori_input = ori_rgb

                # Generating images in axes 
                #im1 = axes[0].imshow(np.transpose(ori_input, (1,2,0))*3)
                #im1_temp = np.transpose(ori_input, (1,2,0))
                ori_input = (ori_input - np.min(ori_input)) / (np.max(ori_input) - np.min(ori_input))
                im1 = axes[0].imshow(np.transpose(ori_input, (1,2,0)))
                im2 = axes[1].imshow(predict_sig, cmap=cmap,vmin = -0.5, vmax = self.num_classes - 0.5)
                im3 = axes[2].imshow(ori_target,cmap=cmap,vmin = -0.5, vmax = self.num_classes - 0.5)

                # Adding colorbar to the right
                #TODO make ax.set_yticklabels automatic with getproject_labels() ?
                if self.classif_mode == 'bin':
                    cbar = subfig.colorbar(im2, shrink=0.7, ax=axes, ticks=np.arange(0,self.num_classes))
                    cbar.ax.set_yticklabels(['1 (NH)', '0 (MH)'])
                else:
                    cbar = subfig.colorbar(im2, shrink=0.7, ax=axes, ticks=np.arange(0,self.num_classes))
                    cbar.ax.set_yticklabels(['0 (EP)','1 (MS)','2 (PH)','3 (ME)','4 (BG)','5 (FN)','6 (TB)', '7 (NH)', '8 (SH)']) # Change colorbar labels
                
                cbar.ax.invert_yaxis() # Flip colorbar 

                # Set axes names
                axes[0].set_title('Sen2 Input')
                axes[1].set_title('Predicted')
                axes[2].set_title('Target')

                # Saving plot 
                plt.savefig(os.path.join(out_root, "figs/figs_{num}.png".format(num = x)))
                plt.clf()
                plt.close(fig)

        # Clear large temporary variables explicitly
        del cm, ori_target, predict_sig
        gc.collect()  # Invoke garbage collector manually

        # Creating compiled array
        full_preds_array = np.hstack([x.flatten() for x in full_preds])
        full_targets_array = np.hstack([x.flatten() for x in full_targets])
        
        ######## GENERER LES CM AVANT QUE CLASS LABELS SOIENT MODIFIE^^^
        ##### A REGLER ICI, class_labels a re-remplacer ??? ignored_labels devraient [etre] ignored_labels + missing labels
        print()
        print("Generating confusion matrix for whole dataset")

        if self.classif_mode == 'bin':
            class_labels = ['0 (NH)', '1 (MH)']
        else:
            class_labels = dict_labels.values()


        disp_full_cm = ConfusionMatrixDisplay(confusion_matrix=full_matrice, display_labels=class_labels)

        fig, ax = plt.subplots(figsize=(10, 10)) # ax is necessary to make large number fit in the output img
        disp_full_cm.plot(values_format = '.0f', ax=ax)

        plt.savefig(os.path.join(out_root, "cm_{version}_{mask_vers}.png".format(version = self.trainer.logger.version, mask_vers = self.cfg.test_mask_dir)))
        plt.close(fig)

        print("Finished")

        print()
        print("Generating normalized confusion matrix for whole dataset")
        disp_full_cm.confusion_matrix = np.nan_to_num(disp_full_cm.confusion_matrix/np.sum(disp_full_cm.confusion_matrix, axis=1)[:, np.newaxis], nan=0)
        #disp_full_cm.confusion_matrix = np.nan_to_num(disp_full_cm.confusion_matrix/np.sum(disp_full_cm.confusion_matrix, axis=0), nan=0, posinf=0, neginf=0)

        fig, ax = plt.subplots(figsize=(10, 10)) # ax is necessary to make large number fit in the output img
        #disp_full_cm.plot(values_format = '.2f', ax=ax)
        disp_full_cm.plot(values_format = '.2f', ax=ax)

        plt.savefig(os.path.join(out_root, "cm_normed_{version}_{mask_vers}.png".format(version = self.trainer.logger.version, mask_vers = self.cfg.test_mask_dir)))
        plt.close(fig)

        print("Finished")


        print()
        print("Generating classification report for whole dataset")
        print()

        # Error handling for when a class is missing from small subset

        targets_class_num = np.unique(full_targets_array)
        targets_class_len = len(np.unique(full_targets_array))

        if self.classif_mode == 'multiclass':
            missing_classes = [x for x in class_num if x not in targets_class_num]
            labels_cr1 = [i for i in class_num if i not in missing_classes]

            #classification_report(y_true=full_targets[0].flatten(), y_pred=full_preds[0].flatten())
            if targets_class_len < len(class_labels):
                print()
                print('*****************************************************************************')
                print('WARNING - The numbers class from this subset of images if lower then expected')
                print('The missing classes are : ', missing_classes)
                print('Setting number of class for classification report to : ', targets_class_len)
                missing_classes_labels = [dict_labels[x] for x in missing_classes]
                class_labels = [i for i in class_labels if i not in missing_classes_labels]
        else:
            labels_cr1 = [0, 1]

        #print(class_labels)
        cr = classification_report(y_true=full_targets_array, y_pred=full_preds_array, target_names=class_labels, labels=labels_cr1)
        cr_dict = classification_report(y_true=full_targets_array, y_pred=full_preds_array, target_names=class_labels, labels=labels_cr1, output_dict=True)

        cr_save_path = os.path.join(out_root, "class_report_{mask_vers}.out".format(version = self.trainer.logger.version, mask_vers = self.cfg.test_mask_dir))
        with open(cr_save_path, 'w') as f:
            f.write(cr)
            
              
        print(cr)

        # Extract macro F1 and accuracy from classification report
        macro_f1 = cr_dict["macro avg"]["f1-score"]
        if 'accuracy' not in cr_dict:
            accuracy = cr_dict["micro avg"]["f1-score"]
        else:
            accuracy = cr_dict["accuracy"]

        self.f1_test = macro_f1
        self.accu_test = accuracy

        # Remove NH class
        if self.classif_mode == 'multiclass':
            ignored_labels = [-1, 7]
            labels_cr2 = [i for i in class_num if i not in ignored_labels]
            ignored_classes_labels = [dict_labels[x] for x in ignored_labels if x != -1]
            class_labels_cr2 = [i for i in class_labels if i not in ignored_classes_labels]

            cr2 = classification_report(y_true=full_targets_array, y_pred=full_preds_array,  target_names=class_labels_cr2, labels=labels_cr2)
            cr_save_path = os.path.join(out_root, "class_report_noNH_{mask_vers}.out".format(version = self.trainer.logger.version, mask_vers = self.cfg.test_mask_dir))
            with open(cr_save_path, 'w') as f:
                f.write(cr2)
                
            print(cr2)

        # Set back to training mode
        self.trainer.model.train()

        # Explicitly delete large objects and run garbage collection
        del full_matrice, full_preds, full_targets, full_preds_array, full_targets_array, disp_full_cm
        gc.collect()

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(train_loss)

        if self.current_epoch > self.binary_only_epochs:
            train_acc = self.trainer.callback_metrics['train_accu'].item()
            self.train_accuracies.append(train_acc)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            val_loss = self.trainer.callback_metrics['val_loss'].item()
            self.val_losses.append(val_loss)
            val_acc = self.trainer.callback_metrics['val_accu'].item()
            self.val_accuracies.append(val_acc)
   
    def on_train_end(self):
        # Plotting training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt_path = os.path.join(self.logger.log_dir, 'training_validation_loss_plot.png')
        plt.savefig(plt_path)
        plt.close()

        # Plotting training and validation accuracies
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt_path = os.path.join(self.logger.log_dir, 'training_validation_accuracy_plot.png')
        plt.savefig(plt_path)
        plt.close()

if __name__ == "__main__":
    pass