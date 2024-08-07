from argparse import ArgumentParser
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
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor, EarlyStopping
import hydra
import segmentation_models_pytorch as smp
from utils.custom_loss import FocalLoss


# Custom LR
import utils.utils
#from utils.custom_loss import FocalLoss
from unet_models.unet_3enco_sum import unet_3enco_sum
from unet_models.unet_3enco_sum_droplayers import unet_3enco_sum_drop
from unet_models.unet_3enco_concat import unet_3enco_concat
from unet_models.unet_3enco_concat_attention import unet_3enco_concat_attention
from unet_models.unet_3enco_concat_dropout import unet_3enco_concat_dropout




# Remove num_workers warning during training
# TODO test num_workers further
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# Remove divide warning during training, should be turn back on when calculating indices on the fly
# or having loss as NaN
# print()
# print("WARNING, divide and invalid error from numpy are deactivated"
#       "Activate in case of error such as loss=nan")
# print()
np.seterr(divide='ignore', invalid='ignore')

# Set default precision for torch
torch.set_float32_matmul_precision('highest')
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") # Add to specified some tensor to GPU

logger = logging.getLogger(__name__)

# TODO Make a 'time' or debug variable that would activate or deactive time prints
#start_time_glob = time.time()


class SemSegment(LightningModule):
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
        
        # self.criterion = smp.losses.FocalLoss(
        #         'multiclass', alpha=0.5, ignore_index=-1, normalized=False
        #     )

        #self.criterion = FocalLoss(weight=self.class_weights, ignore_index=-1)

        #val_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(preds, mask)
        #val_loss = FocalLoss(weight=self.class_weights, ignore_index=-1)(preds, mask)
        #val_loss = FocalLoss(ignore_index=-1)(preds, mask)

        # Initiate value list for plotting
        self.epoch_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Model initialization
        # self.net = unet_3enco_sum(
        #     num_classes=self.num_classes,
        #     input_channels=self.input_channels_main,
        #     input_channels_lidar=self.input_channels_lidar,
        #     input_channels_radar=self.input_channels_radar,
        #     num_layers=self.cfg.num_layers,
        #     features_start=self.cfg.features_start,
        #     bilinear=self.cfg.bilinear,
        # )

        # self.net = unet_3enco_sum_drop(
        #     num_classes=self.num_classes,
        #     input_channels=self.input_channels_main,
        #     input_channels_lidar=self.input_channels_lidar,
        #     input_channels_radar=self.input_channels_radar,
        #     num_layers=self.cfg.num_layers,
        #     features_start=self.cfg.features_start,
        #     bilinear=self.cfg.bilinear,
        #     dropout_rate=self.cfg.dropout_rate
        # )

        # Model initialization
        if self.cfg.architecture == 'unet_3enco_concat':
            self.net = unet_3enco_concat(
                num_classes=self.num_classes,
                input_channels=self.input_channels_main,
                input_channels_lidar=self.input_channels_lidar,
                input_channels_radar=self.input_channels_radar,
                #num_layers=self.cfg.num_layers,
                features_start=self.cfg.features_start,
                bilinear=self.cfg.bilinear,
            )

        elif self.cfg.architecture == 'unet_3enco_concat_3_layers':
            self.net = unet_3enco_concat(
                num_classes=self.num_classes,
                input_channels=self.input_channels_main,
                input_channels_lidar=self.input_channels_lidar,
                input_channels_radar=self.input_channels_radar,
                features_start=self.cfg.features_start,
                bilinear=self.cfg.bilinear,
            )

        elif self.cfg.architecture == 'unet_3enco_concat_attention':
            self.net = unet_3enco_concat_attention(
                num_classes=self.num_classes,
                input_channels=self.input_channels_main,
                input_channels_lidar=self.input_channels_lidar,
                input_channels_radar=self.input_channels_radar,
                #num_layers=self.cfg.num_layers,
                features_start=self.cfg.features_start,
                bilinear=self.cfg.bilinear,
            )

        elif self.cfg.architecture == 'unet_3enco_concat_dropout':
            self.net = unet_3enco_concat_dropout(
                num_classes=self.num_classes,
                input_channels=self.input_channels_main,
                input_channels_lidar=self.input_channels_lidar,
                input_channels_radar=self.input_channels_radar,
                #num_layers=self.cfg.num_layers,
                features_start=self.cfg.features_start,
                bilinear=self.cfg.bilinear,
                dropout_rate=self.cfg.dropout_rate
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
            self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, average="micro")
            self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes = self.num_classes, average = "micro")
            # self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = num_classes, average = "micro")
            # self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = num_classes, average = "micro")
            self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = self.num_classes, average = "micro", ignore_index = 7)
            self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = self.num_classes, average = "micro", ignore_index = 7)

        # self.test_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes, average="micro")
        # self.test_f1 = torchmetrics.classification.MulticlassF1Score(num_classes=self.num_classes, average="micro", ignore_index = 7)      

    def configure_optimizers(self):
        if self.cfg.optim_main == 'Ad':
            opt = torch.optim.Adam(self.net.parameters(), weight_decay = self.cfg.weight_decay, lr=self.cfg.lr)
        else:
            opt = torch.optim.SGD(self.net.parameters(), momentum = self.cfg.momentum, weight_decay = self.cfg.weight_decay, lr=self.cfg.lr)

        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        #sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.95, mode='min', patience=7, verbose=True)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=self.cfg.rl_factor, mode='min', patience=7, verbose=True)
        
        #return [opt], [sch]

        return {'optimizer': opt, "lr_scheduler" : {'scheduler': sch, 'interval':'epoch', 'frequency': 1, 'monitor': 'val_loss'}}
        #return {'optimizer': opt}

    def forward(self, x, y, z):
        return self.net(x, y, z)


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
        # Create a mask for values that should not be ignored
        valid_mask = targets != ignore_index

        # Filter out invalid targets and corresponding predictions
        filtered_targets = targets[valid_mask]
        filtered_preds = preds[valid_mask]

        return filtered_preds, filtered_targets

    def training_step(self, batch, batch_nb):
	    # Switch to train mode
        self.trainer.model.train()

        img, lidar, mask, radar, img_path = batch # img_path only needed in test, but still carried
                                                  # Batch expected output are linked from the dataset.py file
        
        unique_values = torch.unique(mask)
        logger.debug("Unique values in a mask tensor:", unique_values)

        img = img.float()   # x
        lidar = lidar.float()
        mask = mask.long()  # y 
        radar = radar.float()

        #with torch.cuda.amp.autocast():
        preds = self(img, lidar, radar)

        mask_loss = mask.float().unsqueeze(1)
        
        #train_loss = torch.nn.CrossEntropyLoss()(preds, mask)
        if self.classif_mode == 'bin':
            train_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-1)(preds, mask)
            preds_accu = preds.argmax(dim=1).unsqueeze(1)
        else: 
            #train_loss = FocalLoss(weight=self.class_weights, ignore_index=-1)(preds, mask)
            train_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-1)(preds, mask)
            #train_loss = self.criterion(preds, mask)
            mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
            preds_accu = preds.argmax(dim=1).unsqueeze(1)

        # For augmentations ignore value -1 
        if self.cfg.train_transforms:
            filtered_preds, filtered_targets = self.filter_ignore_index(preds_accu, mask_loss, ignore_index=-1)

            train_accu = self.train_accuracy(filtered_preds, filtered_targets)
            train_f1   = self.train_f1(filtered_preds, filtered_targets)
        
        else:
            # Train metrics call
            train_accu = self.train_accuracy(preds_accu, mask_loss)
            train_f1   = self.train_f1(preds_accu, mask_loss)

        # Metric dictionnary
        #log_dict = {"train_loss": train_loss.detach(), "train_accu": train_accu.detach(), "train_f1": train_f1.detach()}

        # Call to log
        self.log("train_loss", train_loss.detach(), on_epoch=True, on_step=True, batch_size=self.batch_size)
        self.log("train_f1", train_f1.detach(), batch_size=self.batch_size)
        self.log("train_accu", train_accu.detach(), prog_bar=True, batch_size=self.batch_size)
           
        # Return can be called in corresponding *epoch_end() section
        return {"loss": train_loss}

    def training_epoch_end(self, outputs):
        train_loss_avg = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu()
        #self.log('train_loss_avg', avg_loss, prog_bar=True, logger=True, on_epoch=True, batch_size=BATCH_SIZE) 
        self.log('train_loss_avg', train_loss_avg.detach(), prog_bar=True, on_epoch=True, batch_size=self.batch_size) 

        # Add to combine graph with val loss
        self.logger.experiment.add_scalars('losses', {'loss_train': train_loss_avg}, self.current_epoch)

        # reset all metrics
        self.train_accuracy.reset()
        self.train_f1.reset()


    def validation_step(self, batch, batch_idx):
        # Switch to eval mode
        self.trainer.model.eval()

        if batch is None:
            return None 

        with torch.no_grad():
            img, lidar, mask, radar, img_path = batch  #img_path only needed in test

            unique_values = torch.unique(mask)
            logger.debug("Unique values in a mask tensor:", unique_values)

            img = img.float()   # x
            lidar = lidar.float()
            mask = mask.long()  # y
            radar = radar.float()

            preds = self(img, lidar, radar)   # predictions
            mask_loss = mask.float().unsqueeze(1) # Unsqueeze for BCE

            # Same for bin and multi class (since using 2 classes for bin)
            val_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(preds, mask)
            #val_loss = FocalLoss(weight=self.class_weights, ignore_index=-1)(preds, mask)
            #val_loss = FocalLoss(ignore_index=-1)(preds, mask)
            #val_loss = self.criterion(preds, mask)

            preds_accu = preds.argmax(dim=1).unsqueeze(1)

            # Filter out ignored index (-1) from predictions and targets (for val_accu, val_f1)
            filtered_preds, filtered_targets = self.filter_ignore_index(preds_accu, mask_loss, ignore_index=-1)

            mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
            # val_accu = self.val_accuracy(preds_accu, mask_loss)
            # val_f1   = self.val_f1(preds_accu, mask_loss)
            val_accu = self.val_accuracy(filtered_preds, filtered_targets)
            val_f1   = self.val_f1(filtered_preds, filtered_targets)

            self.log("val_loss", val_loss.detach(), batch_size=self.batch_size)
            self.log("val_f1", val_f1.detach(), batch_size=self.batch_size)
            self.log("val_accu", val_accu.detach(), prog_bar=True, batch_size=self.batch_size)

            # Combine curves output
            #self.logger.experiment.add_scalars('losses', {'loss_val': val_loss.detach()}, self.current_epoch)

        # Return can be called in corresponding *epoch_end() section
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
        self.log('val_loss_avg', loss_val.detach(), prog_bar=True, on_epoch=True, batch_size=self.batch_size)

        # Add to combine graph with train loss
        self.logger.experiment.add_scalars('losses', {'loss_val': loss_val}, self.current_epoch)

        #log_dict = {"val_loss": loss_val}

        #print(self.conf_print)

        # reset all metrics
        self.val_accuracy.reset()
        self.val_f1.reset()

        #return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

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
            img, lidar, mask, radar, img_path = batch 

            img = img.float()   # x
            lidar = lidar.float()
            mask = mask.long() #.detach().cpu()  # y 
            radar = radar.float() # z

            preds = self(img, lidar, radar)   # predictions

            if self.classif_mode == 'bin':
                #preds_temp = np.multiply((preds.sigmoid().cpu() > 0.5),1)
                #preds_temp = preds_temp.squeeze(1)
                preds_temp = preds.argmax(dim=1)
            else : 
                preds_temp = preds.argmax(dim=1)

            preds_recast = preds_temp.type(torch.IntTensor).to(device=device)     

            # Filtre pour enlever le buffer
            # Filter out negative values in the mask
            valid = mask >= 0
            filtered_preds = preds_recast[valid]
            filtered_mask = mask[valid]

            # if self.cfg.ckpts_dir is not None: # lazy check before implementing proper method
            #     test_accu = self.test_accuracy(filtered_preds, filtered_mask)
            #     test_f1 = self.test_f1(filtered_preds, filtered_mask)

            #     return {'test_accu': test_accu.detach().cpu(), 'test_f1': test_f1.detach().cpu(), 'preds': preds.detach().cpu(), 'mask': mask.detach().cpu()}

            #else:
            # Create and compute confusion matrix
            if filtered_preds.numel() > 0 and filtered_mask.numel() > 0:  # Check if there are any elements to process
                confmat = ConfusionMatrix(task='multiclass', num_classes=self.num_classes).to(device=device)
                conf_print = confmat(filtered_preds, filtered_mask)
                conf_print = conf_print.detach().cpu()
            else: 
                conf_print = torch.zeros(self.num_classes, self.num_classes).to(device=device)   # or appropriate handling if no valid data
                conf_print = conf_print.detach().cpu()

                # if self.classif_mode == 'bin':
                #     confmat = ConfusionMatrix(task='binary', num_classes=self.num_classes).to(device=device)
                # else:
                #     confmat = ConfusionMatrix(task='multiclass', num_classes=self.num_classes).to(device=device)

                # Ancienne partie 
                #confmat = ConfusionMatrix(task='multiclass', num_classes=self.num_classes).to(device=device)
                #conf_print = confmat(preds_recast, mask)
            


            #return {'conf matrice': conf_print.detach().cpu(), 'preds' : preds.detach().cpu(), 'img' : img.detach().cpu(), 'lidar' : lidar.detach().cpu(), 'mask' : mask.detach().cpu(), 'radar' : radar.detach().cpu(), 'img_path' : img_path}
            return {'conf matrice': conf_print.detach().cpu(), 'real_preds' : filtered_preds.detach().cpu(), 'preds' : preds.detach().cpu(), 'img' : img.detach().cpu(), 'mask' : mask.detach().cpu()}
            
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        self.trainer.model.eval()

        # if self.cfg.ckpts_dir is not None:
        #     # test_accuracies = [x['test_accu'] for x in outputs]
        #     # test_f1_scores = [x['test_f1'] for x in outputs]
        #     # avg_test_accu = torch.stack(test_accuracies).mean().item()
        #     # avg_test_f1 = torch.stack(test_f1_scores).mean().item()

        #     # # Debugging: Print individual and average metrics
        #     # # print("Individual Test Accuracies:", test_accuracies)
        #     # # print("Average Test Accuracy:", avg_test_accu)
        #     # # print("Individual Test F1 Scores:", test_f1_scores)
        #     # # print("Average Test F1 Score:", avg_test_f1)

        #     # self.avg_test_accu = avg_test_accu
        #     # self.avg_test_f1 = avg_test_f1

        #     # return avg_test_accu, avg_test_f1

        #     dict_labels = utils.utils.get_project_labels()
        #     class_num = dict_labels.keys()
        #     class_labels = dict_labels.values()

        #     # Setting output path
        #     out_root = "lightning_logs/version_{version}".format(version = self.trainer.logger.version)

        #     all_filtered_preds = torch.cat([x['real_preds'].flatten() for x in outputs])
        #     all_filtered_masks = torch.cat([x['mask'].flatten() for x in outputs])

        #     full_preds = all_filtered_preds.cpu().numpy()
        #     full_targets = all_filtered_masks.cpu().numpy()

        #     # Initialize confusion matrix
        #     full_matrice = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

        #     for pred, target in zip(full_preds, full_targets):
        #       full_matrice[target, pred] += 1

        #     class_labels = list(utils.utils.get_project_labels().values())

        #     cr = classification_report(full_targets, full_preds, target_names=class_labels, output_dict=True)
            
        #     with open(os.path.join(out_root, f"class_report_{self.cfg.test_mask_dir}.out"), 'w') as f:
        #         f.write(classification_report(full_targets, full_preds, target_names=class_labels))
            
        #     # Extract macro F1 and accuracy from classification report
        #     macro_f1 = cr["macro avg"]["f1-score"]
        #     accuracy = cr["accuracy"]
        #     print(classification_report(full_targets, full_preds, target_names=class_labels))
        #     print(f"Macro F1 Score: {macro_f1}")
        #     print(f"Accuracy: {accuracy}")

        #     self.f1_test = macro_f1
        #     self.accu_test = accuracy

        #     return {
        #         'macro_f1': macro_f1,
        #         'accuracy': accuracy,
        #     }

        # else:
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
            # Generate confusion matrix for each samples
            #fig = plt.figure()
            cm = outputs[x]['conf matrice'] #.detach().cpu().numpy()

            # Get targets
            ori_target = outputs[x]['mask'][0] #.detach().cpu().numpy()

            # Get predictions
            # if self.classif_mode == 'bin':
            #     predict_sig_temp = outputs[x]['preds'][0].detach().cpu().sigmoid().numpy()
            #     predict_sig = np.multiply((predict_sig_temp > 0.5), 1, dtype='int32')
            #     predict_sig = np.squeeze(predict_sig)
            #     #predict_sig = outputs[x]['preds'][0].detach().cpu().argmax(dim=0).numpy().astype(np.int32)
            # else:
            #     predict_sig = outputs[x]['preds'][0].detach().cpu().argmax(dim=0).numpy().astype(np.int32)

            #predict_sig = outputs[x]['preds'][0].detach().cpu().argmax(dim=0).numpy().astype(np.int32)
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
        #print(classification_report(full_targets, full_preds, target_names=class_labels))
        #print(f"Macro F1 Score: {macro_f1}")
        #print(f"Accuracy: {accuracy}")

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
        
        return {'f1_test': macro_f1, 'accu_test': accuracy}

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        self.train_losses.append(train_loss)
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