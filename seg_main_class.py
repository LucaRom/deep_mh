from argparse import ArgumentParser
from matplotlib import colors
import sys
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
from torchvision import transforms
from tqdm import tqdm
import os
import datetime
import matplotlib.pyplot as plt
import optuna
from torch.utils.data import DataLoader, Subset
from utils_folder.log_utils import save_config

#from pl_bolts.models.vision.unet import UNet
from unet_2enco import unet_2enco_sum
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
import utils

# Custom loss
from custom_loss import FocalLoss

# Add to specified some tensor to GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# TODO Make a 'time' or debug variable that would activate or deactive time prints
#start_time_glob = time.time()

class SemSegment(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.classif_mode = cfg.exp.classif_mode
        self.num_classes = cfg.model.num_classes
        self.determine_sensor_input_channels()
        self.init_metrics()        
        
        if cfg.mode == 'training':
            self.model = unet_2enco_sum(
                num_classes=cfg.model.num_classes,
                input_channels=self.input_channel_main,
                input_channels_support=self.input_channels_support,
                num_layers=cfg.exp.num_layers,
                features_start=cfg.model.features_start,
                bilinear=cfg.model.bilinear
            )
        elif cfg.mode == 'test_dataloader' or cfg.mode == 'inference':
            self.model = unet_2enco_sum(
                num_classes=cfg.model.num_classes,
                input_channels=self.input_channel_main,
                input_channels_support=self.input_channels_support,
                num_layers=cfg.trained_model.num_layers,
                features_start=cfg.model.features_start,
                bilinear=cfg.model.bilinear
            )

    def determine_sensor_input_channels(self):
        # Base input channels base on experiment
        self.input_channels = (len(self.cfg.exp.opt_bands) + len(self.cfg.exp.indices_lst)) * 2
        self.input_channels_lidar = len(self.cfg.exp.lidar_bands)
        self.input_channels_radar = 6

        # Channels input for 2 encoders
        sensor_configurations = {
            's2s1': {'input_channels_support': 6, 'input_channel_main': self.input_channels},
            's2lr': {'input_channels_support': 5, 'input_channel_main': self.input_channels},
            's1lr': {'input_channels_support': 5, 'input_channel_main': 6},
        }

        sensor_setting = self.cfg.exp.sensors
        if sensor_setting in sensor_configurations:
            self.input_channel_main = sensor_configurations[sensor_setting]['input_channel_main']
            self.input_channels_support = sensor_configurations[sensor_setting]['input_channels_support']
        else:
            raise ValueError(f"Error in sensors input: {sensor_setting} is not a valid sensor configuration.")
        
    def init_metrics(self):
        # Metrics initialization
        '''
            Notes about what I understand with 'average' and 'mdmc_average'

            Micro average takes into account how many samples there are per category

            mdmc = multidimensional-multiclass
            mdmc is not necessary because we use argmax before feeding the metrics, thus having only
            (N,) preds input shape (https://torchmetrics.readthedocs.io/en/stable/pages/classification.html#input-types)
        '''
        if self.classif_mode == 'bin':
                self.train_accuracy = torchmetrics.Accuracy()
                self.val_accuracy = torchmetrics.Accuracy()
                self.train_f1 = torchmetrics.F1Score()
                self.val_f1 = torchmetrics.F1Score()
        else:
            self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes = self.num_classes, average = "micro")
            self.val_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes = self.num_classes, average = "micro")
            # self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = num_classes, average = "micro")
            # self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = num_classes, average = "micro")
            self.train_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = self.num_classes, average = "micro", ignore_index = 7)
            self.val_f1 = torchmetrics.classification.MulticlassF1Score(num_classes = self.num_classes, average = "micro", ignore_index = 7)

    def forward(self, x, y):
        return self.net(x, y)

    def on_train_start(self):
        tensorboard = self.logger.experiment
        tensorboard.add_text("epochs", str(self.trainer.max_epochs))
        tensorboard.add_text("batch_size", str(BATCH_SIZE))                                       
        tensorboard.add_text("optimizer", optim_main)                                       
        #tensorboard.add_text("learning_rate", str(lr_main))  
        tensorboard.add_text("learning_rate", str(self.lr))                                         
        tensorboard.add_text("layers_unet", str(self.num_layers))
        tensorboard.add_text("input_ch", str(input_channel_main))

        # Save config #TODO put in utils as function     
        out_root = "lightning_logs/version_{version}".format(version = self.trainer.logger.version)
        config_save_path = os.path.join(out_root, 'config_param.out')
        with open(config_save_path, 'w') as f:
            f.write('# -----------------------------------#\n')
            f.write('# Parameters of the training session #\n')
            f.write('# -----------------------------------#\n\n')

            # All parameters
            f.write(f'Epochs : {self.trainer.max_epochs}\n')
            f.write(f'Optimizer : {optim_main}\n')
            f.write(f'Learning rate : {self.lr}\n')
            f.write(f'Batch size : {BATCH_SIZE}\n')
            f.write(f'Layers UNet : {self.num_layers}\n')
            f.write(f'Num of optical bands  : {input_channel_main}\n')
            f.write(f'Optical bands  : {opt_bands}\n')
            f.write(f'Indices included  : {indices_lst}\n')
            f.write(f'Num of LiDAR bands  : {input_channel_lidar}\n')
            f.write(f'LiDAR bands  : {lidar_bands}\n')
            f.write(f'Num of radar bands : {input_channel_radar}\n')
            f.write(f'Train mask  : {train_mask_dir}\n')
            f.write(f'Val mask  : {val_mask_dir}\n')
            f.write(f'Test mask : {test_mask_dir}\n')
            f.write(f'Num_workers : {NUM_WORKERS}\n')
            f.write(f'Pin memory : {PIN_MEMORY}\n\n')

            f.write('# -------------------#\n')
            f.write('# Dataset parameters #\n')
            f.write('# -------------------#\n\n')


            # Dataset parameters
            f.write(f'Train region : {input_format}\n')
            f.write(f'Test region : {test_region }\n')
            f.write(f'Classification mode : {classif_mode}\n')

            # Close file
            f.close()

    def training_step(self, batch, batch_nb):

        img, sensor_2, mask, img_path = batch # img_path only needed in test, but still carried
                                                  # Batch expected output are linked from the dataset.py file
                                                  # TODO fix with a if "test" in dataset...

	# Switch to train mode
        self.trainer.model.train()
        
        img = img.float() 
        sensor_2 = sensor_2.float()
        mask = mask.long() 

        #with torch.cuda.amp.autocast():
        preds = self(img, sensor_2)

        mask_loss = mask.float().unsqueeze(1)

        
        #train_loss = torch.nn.CrossEntropyLoss()(preds, mask)
        if classif_mode == 'bin':
            train_loss  = torch.nn.BCEWithLogitsLoss()(preds, mask_loss)
            mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
            preds_accu = torch.sigmoid(preds)
        else: 
            train_loss = FocalLoss()(preds, mask)
            mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
            preds_accu = preds.argmax(dim=1).unsqueeze(1)


        # Train metrics call
        train_accu = self.train_accuracy(preds_accu, mask_loss)
        train_f1   = self.train_f1(preds_accu, mask_loss)
        
        # Metric dictionnary
        log_dict = {"train_loss": train_loss.detach(), "train_accu": train_accu.detach(), "train_f1": train_f1.detach()}

        # Call to log
        self.log("train_loss", train_loss.detach(), batch_size=BATCH_SIZE)
        self.log("train_f1", train_f1.detach(), batch_size=BATCH_SIZE)
        self.log("train_accu", train_accu.detach(), prog_bar=True, batch_size=BATCH_SIZE)

        # Combine curves output
        self.logger.experiment.add_scalars('losses', {'loss_train': train_loss.detach()}, self.current_epoch)

        return {"loss": train_loss, "log": log_dict}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu()
        #self.log('train_loss_avg', avg_loss, prog_bar=True, logger=True, on_epoch=True, batch_size=BATCH_SIZE) 
        self.log('train_loss_avg', avg_loss, prog_bar=True, on_epoch=True, batch_size=BATCH_SIZE) 

        # TODO see start time line on top of code
        # current_train_time = time.time() - self.new_time
        # print("--- %s seconds (one train epoch) ---" % (current_train_time))
        # self.new_time = time.time()
        # self.train_time_list.append(current_train_time)

        # reset all metrics
        self.train_accuracy.reset()
        self.train_f1.reset()

    # def on_train_end(self):
    #     print("Average train time per epoch : ", sum(self.train_time_list) / len(self.train_time_list))

    def validation_step(self, batch, batch_idx):
        # Switch to eval mode
        self.trainer.model.eval()

        with torch.no_grad():
            img, sensor_2, mask, img_path = batch  #img_path only needed in test
            img = img.float()
            sensor_2 = sensor_2.float()
            mask = mask.long()

            preds = self(img, sensor_2)   # predictions

            mask_loss = mask.float().unsqueeze(1)
            if classif_mode == 'bin':
                val_loss  = torch.nn.BCEWithLogitsLoss()(preds, mask_loss)
                preds_accu = preds.sigmoid()
            else: 
                val_loss = FocalLoss()(preds, mask)
                preds_accu = preds.argmax(dim=1).unsqueeze(1)

            mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
            val_accu = self.val_accuracy(preds_accu, mask_loss)
            val_f1   = self.val_f1(preds_accu, mask_loss)


            self.log("val_loss", val_loss.detach(), batch_size=BATCH_SIZE)
            self.log("val_f1", val_f1.detach(), batch_size=BATCH_SIZE)
            self.log("val_accu", val_accu.detach(), prog_bar=True, batch_size=BATCH_SIZE)

            # Combine curves output
            self.logger.experiment.add_scalars('losses', {'loss_val': val_loss.detach()}, self.current_epoch)

        return {"val_loss": val_loss}
    
    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
        #self.log('val_loss_avg', loss_val, on_epoch=True, batch_size=BATCH_SIZE)
        self.log('val_loss_avg', loss_val.detach(), prog_bar=True, on_epoch=True, batch_size=BATCH_SIZE)

        #log_dict = {"val_loss": loss_val}

        #print(self.conf_print)

        # reset all metrics
        self.val_accuracy.reset()
        self.val_f1.reset()

        #return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.trainer.model.eval()

        img, sensor_2, mask, img_path = batch 

        img = img.float()   
        sensor_2 = sensor_2.float()
        mask = mask.long() 

        preds = self(img, sensor_2)   # predictions

        if classif_mode == 'bin':
            preds_temp   = np.multiply((preds.sigmoid().cpu() > 0.5),1)
        else : 
            preds_temp   = preds.argmax(dim=1)  #.unsqueeze(1)

        preds_recast = preds_temp.type(torch.IntTensor).to(device=device)     

        if classif_mode == 'bin':
            confmat = ConfusionMatrix(task='binary', num_classes=self.num_classes + 1).to(device=device)
        else:
            confmat = ConfusionMatrix(task='multiclass', num_classes=self.num_classes).to(device=device)

        conf_print = confmat(preds_recast, mask)
       
        return {'conf matrice': conf_print.detach().cpu(), 'preds' : preds.detach().cpu(), 'img' : img.detach().cpu(), 'sensor_2' : sensor_2.detach().cpu(), 'mask' : mask.detach().cpu(), 'img_path' : img_path}
        #return {'conf matrice': conf_print.detach(), 'preds' : preds.detach(), 'img' : img, 'img_path' : img_path}
        
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        # Define full test variable      
        if classif_mode == 'bin':
            self.num_classes = self.num_classes + 1

        full_matrice = np.zeros((self.num_classes,self.num_classes)) #, dtype=torch.float32) #.to(device=device)

        full_preds   = []
        full_targets  = []

        # Define labels
        if classif_mode == 'bin':
            class_labels = ['0 (NH)', '1 (MH)']
        else:
            dict_labels = utils.get_project_labels()
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
            cm = outputs[x]['conf matrice'].detach().cpu().numpy()

            # Get targets
            ori_target = outputs[x]['mask'][0].detach().cpu().numpy()

            # Get predictions
            if classif_mode == 'bin':
                predict_sig_temp = outputs[x]['preds'][0].detach().cpu().sigmoid().numpy()
                predict_sig = np.multiply((predict_sig_temp > 0.5), 1, dtype='int32')
                predict_sig = np.squeeze(predict_sig)
                #predict_sig = outputs[x]['preds'][0].detach().cpu().argmax(dim=0).numpy().astype(np.int32)
            else:
                predict_sig = outputs[x]['preds'][0].detach().cpu().argmax(dim=0).numpy().astype(np.int32)


            # Feed full data for overall output
            full_matrice += cm # Feed full matrice
            full_preds.append(predict_sig)
            full_targets.append(ori_target)

            # Generate other wanted outputs
            if generate_cm_sample:             
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
                disp.plot()
                plt.savefig(os.path.join(out_root, "cms/cm_{num}.png".format(num = x)))
                plt.clf() 
                plt.close()
                #plt.close(fig)

            if generate_tif_sample: 
                # Extract CRS and transforms
                img_path = outputs[x]['img_path']
                src = rasterio.open(img_path[0])
                sample_crs = src.crs
                transform_ori = src.transform
                src.close() # TODO Needed?

                # write predict image to file
                tiff_save_path =  os.path.join(out_root, "tifs/preds_{version}_{num}.tif".format(version = self.trainer.logger.version, num = x))

                predict_img = rasterio.open(tiff_save_path, 'w', driver='GTiff',
                                height = input_tile_size, width = input_tile_size,
                                count=1, dtype=str(predict_sig.dtype),
                                crs=sample_crs,
                                transform=transform_ori)

                predict_img.write(predict_sig, 1)
                predict_img.close()

            if generate_fig_sample: 
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
                if classif_mode == 'bin':
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

        # Creating compiled array
        full_preds_array = np.hstack([x.flatten() for x in full_preds])
        full_targets_array = np.hstack([x.flatten() for x in full_targets])
        
        ######## GENERER LES CM AVANT QUE CLASS LABELS SOIENT MODIFIE^^^
        ##### A REGLER ICI, class_labels a re-remplacer ??? ignored_labels devraient [etre] ignored_labels + missing labels
        print()
        print("Generating confusion matrix for whole dataset")

        if classif_mode == 'bin':
            class_labels = ['0 (NH)', '1 (MH)']
        else:
            class_labels = dict_labels.values()


        disp_full_cm = ConfusionMatrixDisplay(confusion_matrix=full_matrice, display_labels=class_labels)

        fig, ax = plt.subplots(figsize=(10, 10)) # ax is necessary to make large number fit in the output img
        disp_full_cm.plot(values_format = '.0f', ax=ax)

        plt.savefig(os.path.join(out_root, "cm_{version}_{mask_vers}.png".format(version = self.trainer.logger.version, mask_vers = out_root_mask)))
        plt.close(fig)

        print("Finished")

        print()
        print("Generating normalized confusion matrix for whole dataset")
        disp_full_cm.confusion_matrix = np.nan_to_num(disp_full_cm.confusion_matrix/np.sum(disp_full_cm.confusion_matrix, axis=1)[:, np.newaxis], nan=0)
        #disp_full_cm.confusion_matrix = np.nan_to_num(disp_full_cm.confusion_matrix/np.sum(disp_full_cm.confusion_matrix, axis=0), nan=0, posinf=0, neginf=0)

        fig, ax = plt.subplots(figsize=(10, 10)) # ax is necessary to make large number fit in the output img
        #disp_full_cm.plot(values_format = '.2f', ax=ax)
        disp_full_cm.plot(values_format = '.2f', ax=ax)

        plt.savefig(os.path.join(out_root, "cm_normed_{version}_{mask_vers}.png".format(version = self.trainer.logger.version, mask_vers = out_root_mask)))
        plt.close(fig)

        print("Finished")


        print()
        print("Generating classification report for whole dataset")
        print()

        # Error handling for when a class is missing from small subset

        targets_class_num = np.unique(full_targets_array)
        targets_class_len = len(np.unique(full_targets_array))

        if classif_mode == 'multiclass':
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
        cr = classification_report(y_true=full_targets_array, y_pred=full_preds_array,  target_names=class_labels, labels=labels_cr1)
        cr_save_path = os.path.join(out_root, "class_report_{mask_vers}.out".format(version = self.trainer.logger.version, mask_vers = out_root_mask))
        with open(cr_save_path, 'w') as f:
            f.write(cr)
              
        print(cr)

        # Remove NH class
        if classif_mode == 'multiclass':
            ignored_labels = [7]
            labels_cr2 = [i for i in class_num if i not in ignored_labels]
            ignored_classes_labels = [dict_labels[x] for x in ignored_labels]
            class_labels_cr2 = [i for i in class_labels if i not in ignored_classes_labels]

            cr2 = classification_report(y_true=full_targets_array, y_pred=full_preds_array,  target_names=class_labels_cr2, labels=labels_cr2)
            cr_save_path = os.path.join(out_root, "class_report_noNH_{mask_vers}.out".format(version = self.trainer.logger.version, mask_vers = out_root_mask))
            with open(cr_save_path, 'w') as f:
                f.write(cr2)
                
            print(cr2)

        # TODO find if its necessary to set back to training mode
        self.trainer.model.train()
        
    def configure_optimizers(self):
        if optim_main == 'Ad':
            opt = torch.optim.Adam(self.net.parameters(), weight_decay = 0.001, lr=self.lr)
        else:
            opt = torch.optim.SGD(self.net.parameters(), momentum = 0.9, weight_decay = 0.001, lr=self.lr)
            #opt = torch.optim.SGD(self.net.parameters(), momentum = 0.9, weight_decay = 0.01, lr=self.lr)

        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.95, mode='min', patience=7, verbose=True)
        
        #return [opt], [sch]

        return {'optimizer': opt, "lr_scheduler" : {'scheduler': sch, 'interval':'epoch', 'frequency': 1, 'monitor': 'val_loss'}}
        #return {'optimizer': opt}

#def objective(trial, model_class, train_dataloader, val_dataloader, test_dataloader_list):
def objective(trial, model_class, train_dataloader, val_dataloader):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    # Add other hyperparameters here, if needed

    # Create model using suggested hyperparameters
    model = model_class(lr=lr)

    # Define trainer
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor="val_loss", patience=7)
    trainer = Trainer(
        accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        max_epochs=num_epochs
    )

    # Train and evaluate the model
    trainer.fit(model, train_dataloader, val_dataloader)
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss

# def cli_main():
#     # Define seeds
#     # TODO look to activate 'true full seeding'
#     seed_everything(1234, workers=True)

#     # Define model and rule for model checkpoints
#     model = SemSegment()
#     checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
  
#     # LearningRateMonitor for automatic LR logging
#     lr_monitor = LearningRateMonitor(logging_interval='step')

#     # Define trainer
#     trainer = Trainer(accelerator='gpu', devices=1, 
#                       callbacks=[checkpoint_callback, lr_monitor], 
#                       max_epochs=num_epochs)

#     # Launch training and testing successively
#     trainer.fit(model, train_loader, val_loader)
#     trainer.test(model, test_loader) 

#     # TODO Produce a cleaner loop for all masks types or seperate function / file.py
#     # TODO trainer.test prend plusieurs dataloader en imput, est-ce possible de faire une loop direct?
#     # Testing on different masks

#     global generate_cm_sample, generate_tif_sample, generate_fig_sample, out_root_mask

#     generate_cm_sample = False
#     generate_tif_sample = False
#     generate_fig_sample = False

#     out_root_mask = '3223_buff'
#     trainer.test(model, test_loader_2)

#     out_root_mask = 'ori_9c'
#     trainer.test(model, test_loader_3)

def cli_main():
    # Define seeds
    seed_everything(1234, workers=True)

    # Prepare the test data loader list
    # test_dataloader_list = [
    #     (test_loader_2, '3223_buff'),
    #     (test_loader_3, 'ori_9c')
    # ]

    # Create an Optuna study and optimize the objective function
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage = "sqlite:///optuna.db"
    study = optuna.create_study(study_name="Optimization_gmq805", storage=storage, direction="minimize", load_if_exists=True)
    # study.optimize(
    #     lambda trial: objective(trial, SemSegment, train_loader, val_loader, test_dataloader_list),
    #     n_trials=100
    # )

    study.optimize(
        lambda trial: objective(trial, SemSegment, small_train_loader, small_val_loader),
        n_trials=5
    )

    # Retrieve the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Train and test the final model using the best hyperparameters
    best_model = SemSegment(**best_params)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        accelerator='gpu', devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=num_epochs
    )
    trainer.fit(best_model, small_train_loader, small_val_loader)

    # Test the final model on different masks
    # for test_loader, out_root_mask in test_dataloader_list:
    #     trainer.test(best_model, test_loader)

# TODO TEST in inference and then delete
# Laund evaluate on modell checkpoint

def evaluate_test_solo(ckpt_path):
    model = SemSegment.load_from_checkpoint(
    checkpoint_path=ckpt_path
    )

    seed_everything(1234, workers=True)

    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.set_float32_matmul_precision('medium')

    trainer = Trainer(accelerator='gpu', devices=1)

    trainer.test(model, dataloaders=test_loader)     # Test with default mask

    global generate_cm_sample, generate_tif_sample, generate_fig_sample, out_root_mask

    generate_cm_sample = False
    generate_tif_sample = False
    generate_fig_sample = False

    out_root_mask = '3223_buff'

    trainer.test(model, dataloaders=test_loader_2)   # Additionnal test with different mask

if __name__ == "__main__":
    #  # Import data with custom loader
    # train_region = "estrie_over50p"
    # input_format = train_region
    # test_region = "local_split"
    # classif_mode = "multiclass"          #multiclass or bin
    # dataset_size = 'full'         #small or full
    # test_mask_dir = None  # Choices : '3223_full', Change mask tiles used for test dataloader
    # PIN_MEMORY = True
    # NUM_WORKERS = 4
    # BATCH_SIZE = 6
    # num_epochs = 5
    # optim_main = "sg"  # 'Ad' ou 'sg'
    # #lr_main = 0.001
    # num_layers_main = 4
    # input_channel_main = 32 # TODO change automatically if possible
    # input_channel_lidar = 5 # 5 = pas de mnt, 6 = Full
    # input_channel_radar = 6
    # input_tile_size = 256 # Check size of output in test_epoch_end

    fast_dev_run = False
    input_format = "estrie_over50p" # "estrie_over50p" ou "estrie_over0p"
    test_region = "local_split"
    classif_mode = "multiclass"          #multiclass or bin
    dataset_size = 'full'         #small or full
    train_mask_dir = 'mask_multiclass_3223_9c'  # Choices : '3223_full', 'mask_multiclass_3223_9c'
    val_mask_dir = 'mask_multiclass_3223_9c'  # Choices : '3223_full', 'mask_multiclass_3223_9c'
    test_mask_dir = 'mask_multiclass_3223_9c'  # Choices : '3223_full', 'mask_multiclass_3223_9c'
    #opt_bands = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11] # Keep all bands except 'B1' and 'B9'
    opt_bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # Keep all bands
    lidar_bands = ['mhc', 'pentes', 'tpi', 'tri', 'twi'] # ['mnt', 'mhc', 'pentes', 'tpi', 'tri', 'twi']
    indices_lst = ['ndvi'] #  ['ndvi', 'ndwi', 'ndmi', 'mbwi']
    #class_weights_dict = {0: 0.9974770891260059, 1: 0.999314990078947, 2: 0.9957716837102707, 3: 0.9326251616735741, 4: 0.998735088954158, 5: 0.9850211180583281, 6: 0.9642186109416683, 7: 0.13410737472641465, 8: 0.9927288827306331}
    PIN_MEMORY = True
    NUM_WORKERS = 4
    BATCH_SIZE = 6
    sensors = 's2lr'
    #num_epochs = 5
    optim_main = "sg"  # 'Ad' ou 'sg'
    #lr_main = 0.001
    num_layers = 4

    input_channel_main = (len(opt_bands) + len(indices_lst))*2
    input_channel_lidar = len(lidar_bands)
    input_channel_radar = 6
    input_tile_size = 256 # Check size of output in test_epoch_end

    if sensors == 's2s1':
        input_channels_support = 6
    elif sensors == 's2lr':
        input_channels_support = 5
    elif sensors == 's1lr':
        input_channel_main = 6
        input_channels_support = 5
    else:
        print("Error in sensors input")

    if classif_mode == 'bin':
        num_classes = 1
    else:
        num_classes = 9

    # Set transforms
    #train_transform = transforms.RandomHorizontalFlipMultiChannel(p=0.5)
    # train_transform = Compose([
    #     transforms.RandomHorizontalFlipMultiChannel(p=0.5),
    #     transforms.RandomRotationMultiChannel(degrees=15, p=0.5),
    #     transforms.RandomBrightnessMultiChannel(factor_range=(0.8, 1.2), p=0.5),
    # ])

    train_transform = None

    # Set input_format for loaders
    if input_format in ['estrie_over0p', 'estrie_over50p']:

        test_mode = False
        generate_cm_sample = False
        generate_tif_sample = True
        generate_fig_sample = False
        out_root_mask = ''

        # train_loader, val_loader, test_loader = utils.get_tiled_datasets_estrie(
        # input_format,
        # classif_mode,
        # test_mask_dir,
        # BATCH_SIZE,
        # dataset_size,
        # # train_transform,
        # # val_transforms,
        # test_mode,
        # NUM_WORKERS, PIN_MEMORY,
        # )

        train_loader, val_loader, test_loader = utils.get_tiled_datasets_estrie(
        input_format,
        classif_mode,
        train_mask_dir,
        val_mask_dir,
        test_mask_dir,
        BATCH_SIZE,
        dataset_size,
        train_transform,
        # val_transforms,
        test_mode,
        sensors,
        opt_bands, 
        lidar_bands,
        indices_lst,
        NUM_WORKERS, PIN_MEMORY
        )

        # # Small dataloader
        # # train_loader
        # small_train_size = 100
        # indices = list(range(len(train_loader.dataset)))
        # np.random.shuffle(indices)
        # small_indices = indices[:small_train_size]
        # small_train_dataset = Subset(train_loader.dataset, small_indices)   
        # small_train_loader = DataLoader(small_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=custom_collate)

        # # val_loader
        # small_val_size = 10
        # indices = list(range(len(val_loader.dataset)))
        # np.random.shuffle(indices)
        # small_indices = indices[:small_val_size]
        # small_val_dataset = Subset(val_loader.dataset, small_indices)   
        # small_val_loader = DataLoader(small_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, collate_fn=custom_collate)



        #Generating dataset number 2
        test_mode = True
        test_mask_dir = 'mask_multiclass_3223_buff'

        test_loader_2 = utils.get_tiled_datasets_estrie(
        input_format,
        classif_mode,
        train_mask_dir,
        val_mask_dir,
        test_mask_dir,
        BATCH_SIZE,
        dataset_size,
        train_transform,
        # val_transforms,
        test_mode,
        sensors,
        opt_bands, 
        lidar_bands,
        indices_lst,
        NUM_WORKERS, PIN_MEMORY
        )
        
        # # Generating dataset number 3
        # test_mode = True
        # test_mask_dir = 'ori_9c'

        # test_loader_3 = utils.get_tiled_datasets_estrie(
        # input_format,
        # classif_mode,
        # test_mask_dir,
        # BATCH_SIZE,
        # dataset_size,
        # # train_transform,
        # # val_transforms,
        # test_mode,
        # NUM_WORKERS, PIN_MEMORY,
        # )

    else:
        print("Train region name unknown : ", input_format)

    # # Training + test (main function)
    # start_time_cli = datetime.datetime.now()
    # print("Start time : ", start_time_cli)
    # if dataset_size == 'small':
    #     answer = input('Use the small dataset for testing purpose? (y or n)\n')
    #     if answer == 'y':
    #         cli_main()
    #     else:
    #         exit()
    # else:
    #     cli_main()
    # print("Start time  : ", start_time_cli)
    # print("Finish time : ", datetime.datetime.now())
    
   #235 = s2s1, 236 = s2lr

    #evaluate_test_solo("lightning_logs/version_235/checkpoints/epoch=193-step=186046.ckpt")
    evaluate_test_solo("lightning_logs/version_236/checkpoints/epoch=196-step=188923.ckpt")
    #evaluate_test_solo("lightning_logs/version_222/checkpoints/epoch=193-step=176152.ckpt")
