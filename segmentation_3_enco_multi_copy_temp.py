from argparse import ArgumentParser
from matplotlib import colors

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
import os

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

#from pl_bolts.models.vision.unet import UNet
from unet_3enco_sum import unet_3enco_sum
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor

# Custom LR
from utils import get_datasets, get_project_labels

# Custom loss
from custom_loss import FocalLoss

# Add to specified some tensor to GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#start_time_glob = time.time()

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

        super().__init__()
    
        self.num_classes = num_classes
        self.num_layers = num_layers_main
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr_main
        self.new_time = time.time()
        self.train_time_list = []

        # Metrics 
        self.train_accuracy = torchmetrics.Accuracy(mdmc_average='samplewise')
        self.val_accuracy = torchmetrics.Accuracy(mdmc_average='samplewise')
        self.train_f1 = torchmetrics.F1Score(mdmc_average='samplewise')
        self.val_f1 = torchmetrics.F1Score(mdmc_average='samplewise')

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

    def on_train_start(self):
        tensorboard = self.logger.experiment
        tensorboard.add_text("epochs", str(num_epochs))
        tensorboard.add_text("batch_size", str(BATCH_SIZE))                                       
        tensorboard.add_text("optimizer", optim_main)                                       
        tensorboard.add_text("learning_rate", str(lr_main))                                       
        tensorboard.add_text("layers_unet", str(num_layers_main))
        tensorboard.add_text("input_ch", str(input_channel_main))

    # def on_train_epoch_start(self) -> None:
    #     if len(self.trainer.callbacks[1].lrs['lr-Adam']) == 0:
    #         current_lr = lr_main
    #     else:
    #         current_lr = self.trainer.callbacks[1].lrs['lr-Adam'][-1]
    #     print(f'The current learning rate is : {current_lr}')

    def training_step(self, batch, batch_nb):

        img, lidar, mask, radar, img_path = batch # img_path only needed in test, but still carried
                                                  # Batch expected output are linked from the dataset.py file
                                                  # TODO fix with a if "test" in dataset...

	# Switch to train mode
        self.trainer.model.train()
        
        img = img.float()   # x
        lidar = lidar.float()
        mask = mask.long()  # y 
        radar = radar.float()

        #with torch.cuda.amp.autocast():
        preds = self(img, lidar, radar)

        mask_loss = mask.float().unsqueeze(1)

        # Train metrics call
        #train_loss = torch.nn.CrossEntropyLoss()(preds, mask)
        train_loss = FocalLoss()(preds, mask)
        
        mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
        preds_accu = preds.argmax(dim=1).unsqueeze(1)

        train_accu = self.train_accuracy(preds_accu, mask_loss)
        train_f1   = self.train_f1(preds_accu, mask_loss)
        
        # Metric dictionnary
        log_dict = {"train_loss": train_loss.detach(), "train_accu": train_accu.detach(), "train_f1": train_f1.detach()}

        # Call to log
        self.log("train_loss", train_loss.detach(), batch_size=BATCH_SIZE)
        self.log("train_f1", train_f1.detach(), batch_size=BATCH_SIZE)
        self.log("train_accu", train_accu.detach(), prog_bar=True, batch_size=BATCH_SIZE)

        return {"loss": train_loss, "log": log_dict}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu()
        #self.log('train_loss_avg', avg_loss, prog_bar=True, logger=True, on_epoch=True, batch_size=BATCH_SIZE) 
        self.log('train_loss_avg', avg_loss, prog_bar=True, on_epoch=True, batch_size=BATCH_SIZE) 

        # current_train_time = time.time() - self.new_time
        # print("--- %s seconds (one train epoch) ---" % (current_train_time))
        # self.new_time = time.time()
        # self.train_time_list.append(current_train_time)

        # # compute metrics
        # train_accuracy = self.train_accuracy.compute()
        # train_f1 = self.train_f1.compute()

        # # log metrics
        # self.log("epoch_train_accuracy", train_accuracy)
        # self.log("epoch_train_f1", train_f1)

        # reset all metrics
        self.train_accuracy.reset()
        self.train_f1.reset()

    # def on_train_end(self):
    #     print("Average train time per epoch : ", sum(self.train_time_list) / len(self.train_time_list))

    def validation_step(self, batch, batch_idx):
        # Switch to eval mode
        self.trainer.model.eval()

        with torch.no_grad():
            img, lidar, mask, radar, img_path = batch  #img_path only needed in test
            img = img.float()   # x
            lidar = lidar.float()
            mask = mask.long()  # y
            radar = radar.float()

            preds = self(img, lidar, radar)   # predictions

            mask_loss = mask.float().unsqueeze(1)
            val_loss = FocalLoss()(preds, mask)

            preds_accu = preds.argmax(dim=1).unsqueeze(1)

            mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
            val_accu = self.val_accuracy(preds_accu, mask_loss)
            val_f1   = self.val_f1(preds_accu, mask_loss)


            self.log("val_loss", val_loss.detach(), batch_size=BATCH_SIZE)
            self.log("val_f1", val_f1.detach(), batch_size=BATCH_SIZE)
            self.log("val_accu", val_accu.detach(), prog_bar=True, batch_size=BATCH_SIZE)

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

        img, lidar, mask, radar, img_path = batch 

        img = img.float()   # x
        lidar = lidar.float()
        mask = mask.long()  # y 
        radar = radar.float() # z

        preds = self(img, lidar, radar)   # predictions

        preds_temp   = preds.argmax(dim=1).unsqueeze(1)
        preds_recast = preds_temp.type(torch.IntTensor).to(device=device)     

        confmat = ConfusionMatrix(num_classes=8).to(device=device)
        conf_print = confmat(preds_recast, mask)
       
        return {'conf matrice': conf_print, 'preds' : preds, 'img' : img, 'lidar' : lidar, 'mask' : mask, 'radar' : radar, 'img_path' : img_path}
        
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        # Define full test variables
        full_matrice = np.zeros((8,8)) #, dtype=torch.float32) #.to(device=device)
        full_preds   = []
        full_targets  = []

        # Define labels
        dict_labels = get_project_labels()
        class_num = dict_labels.keys()
        class_labels = dict_labels.values()

        # Define output path root
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
                os.mkdir(f_paths)

        print()
        print("Generating outputs by sample")
        for x in range(len(outputs)):
            # Generate confusin matrix for samples
            fig = plt.figure()
            cm = outputs[x]['conf matrice'].cpu().numpy()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_label=class_labels)
            disp.plot()
            plt.savefig(os.path.join(out_root, "cms/cm_{num}.png".format(num = x)))
            plt.close(fig)

            # Extract CRS and transforms
            img_path = outputs[x]['img_path']
            src = rasterio.open(img_path[0])
            sample_crs = src.crs
            transform_ori = src.transform
            src.close() # Needed?

            ori_input = outputs[x]['img'][0].cpu().numpy()
            ori_target = outputs[x]['mask'][0].cpu().numpy()
            #predict_sig = outputs[sample]['preds'][].cpu().squeeze().sigmoid().numpy()
            #predict_sig = outputs[x]['preds'][0].cpu().squeeze().sigmoid().numpy()
            #predict_sig = np.multiply((predict_sig > 0.5),1)
            #predict_sig = outputs[x]['preds'][0].softmax(dim=1).argmax(dim=1).unsqueeze(1)
            #predict_sig = outputs[x]['preds'][0].cpu().softmax(dim=1).argmax(dim=0).unsqueeze(0).numpy()
            predict_sig = outputs[x]['preds'][0].cpu().argmax(dim=0).numpy().astype(np.int32)



            # write predict image to file
            #tiff_save_path = "lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = self.trainer.logger.version, num = x)
            #tiff_save_path = "lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = log_version, num = x)

            tiff_save_path =  os.path.join(out_root, "tifs/preds_{version}_{num}.png".format(version = self.trainer.logger.version, num = x))

            predict_img = rasterio.open(tiff_save_path, 'w', driver='GTiff',
                            height = input_tile_size, width = input_tile_size,
                            count=1, dtype=str(predict_sig.dtype),
                            crs=sample_crs,
                            transform=transform_ori)

            predict_img.write(predict_sig, 1)
            predict_img.close()

            # trying to costomize colors


            # bounds = [0,1,2,3,4,5,6,7]
            # norm = colors.BoundaryNorm(bounds, cmap.N)
            # mat = plt.matshow(data,cmap=cmap,vmin = np.min(data)-.5, vmax = np.max(data)+.5)
            # cax = plt.colorbar(mat, ticks=np.arange(np.min(data),np.max(data)+1))

            fig = plt.figure(figsize=(15, 5))
            subfig = fig.subfigures(nrows=1, ncols=1)
            axes = subfig.subplots(nrows=1, ncols=3,sharey=True)

            cmap = plt.get_cmap('tab10', 8)

            # Generating images in axes 
            im1 = axes[0].imshow(np.transpose(ori_input[[3,2,1],:,:],(1,2,0))*3)
            im2 = axes[1].imshow(predict_sig, cmap=cmap,vmin = -0.5, vmax = 7.5)
            im3 = axes[2].imshow(ori_target,cmap=cmap,vmin = -0.5, vmax = 7.5)

            # Adding colorbar to the right
            cbar = subfig.colorbar(im2, shrink=0.7, ax=axes, ticks=np.arange(0,8))
            cbar.ax.set_yticklabels(['0 (EP)','1 (MS)','2 (PH)','3 (ME)','4 (BG)','5 (FN)','6 (TB)', '7 (NH)']) # Change colorbar labels
            cbar.ax.invert_yaxis() # Flip colorbar 

            # Set axes names
            axes[0].set_title('Sen2 Input')
            axes[1].set_title('Predicted')
            axes[2].set_title('Target')

            # Saving plot 
            plt.savefig(os.path.join(out_root, "figs/figs_{num}.png".format(num = x)))
            plt.close(fig)




            plt.title("Input")

            cmap = plt.get_cmap('tab10', 8)

            # Sen2 image
            fig = plt.figure()
            plt.subplot(1,3,1)
            #plt.imshow(np.transpose(ori_input[[3,2,1],:,:],(1,2,0))*3)
            im = plt.imshow(np.transpose(ori_input[[3,2,1],:,:],(1,2,0)))
            plt.title("Input")


            # Prediction image
            plt.subplot(1,3,2)
            #img = plt.imshow(predict_sig)
            #plt.colorbar(img, fraction=0.046, pad=0.04, boundaries=[0,1,2,3,4,5,6,7])

            im = plt.imshow(predict_sig, cmap=cmap,vmin = -0.5, vmax = 7.5)
            #cax = plt.colorbar(im, ticks=np.arange(0,8), fraction=0.046, pad=0.04)

            #plt.colorbar(img)
            plt.title("Predict")

            # Target image
            plt.subplot(1,3,3)
            plt.imshow(ori_target,cmap=cmap,vmin = -0.5, vmax = 7.5)
            cax = plt.colorbar(im, ticks=np.arange(0,8), fraction=0.046, pad=1)
            #plt.colorbar(fraction=0.046, pad=0.04, boundaries=[0,1,2,3,4,5,6,7])
            plt.title("Target")

            subfig = fig.subfigures(nrows=1, ncols=3)
            subfig.colorbar(im, location='bottom')

            plt.savefig(os.path.join(out_root, "figs/figs_{num}.png".format(num = x)))
            #plt.savefig("lightning_logs/version_{version}/fig_{num}.png".format(version = self.trainer.logger.version, num = x))
            #plt.savefig("lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = log_version, num = x))
            plt.close(fig)

            # Feed full data for overall output
            full_matrice += cm # Feed full matrice
            full_preds.append(predict_sig)
            full_targets.append(ori_target)


        # Creating compiled array 
        full_preds_array = np.hstack([x.flatten() for x in full_preds])
        full_targets_array = np.hstack([x.flatten() for x in full_targets])
        
        print()
        print("Generating classification report for whole dataset")
        #classification_report(y_true=full_targets[0].flatten(), y_pred=full_preds[0].flatten())
        cr = classification_report(y_true=full_targets_array, y_pred=full_preds_array,  target_names=class_labels)
        cr_save_path = os.path.join(out_root, 'class_report.out')
        with open(cr_save_path, 'w') as f:
            f.write(cr)
              
        print(cr)

        print()
        print("Generating confusion matrix for whole dataset")
        disp_full_cm = ConfusionMatrixDisplay(confusion_matrix=full_matrice, display_labels=class_labels)
        fig, ax = plt.subplots(figsize=(10, 10)) # ax is necessary to make large number fit in the output img
        disp_full_cm.plot(values_format = '.0f', ax=ax)

        plt.savefig(os.path.join(out_root, "cm_{version}.png".format(version = self.trainer.logger.version)))
        plt.close(fig)

        print()
        print("Finish")

        # TODO find if its necessary to set back to training mode
        self.trainer.model.train()
        
    def configure_optimizers(self):
        if optim_main == 'Ad':
            opt = torch.optim.Adam(self.net.parameters(), weight_decay = 0.001, lr=self.lr)
        else:
            opt = torch.optim.SGD(self.net.parameters(), momentum = 0.9, weight_decay = 0.001, lr=self.lr)

        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.95, mode='min', patience=7, verbose=True)
        
        #return [opt], [sch]

        return {'optimizer': opt, "lr_scheduler" : {'scheduler': sch, 'interval':'epoch', 'frequency': 1, 'monitor': 'val_loss'}}
        #return {'optimizer': opt}

 
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

# class MyPrintingCallback(Callback):
#     def on_train_end(self, trainer, pl_module):
#         #print('do something when training ends')
#         print("End of training")

def cli_main():
    seed_everything(1234, workers=True)

    model = SemSegment()
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss", mode="min")

    #lr_logger = LearningRateMonitor(logging_interval='step')

    # # train (custom)
    start_time = time.time()
    trainer = Trainer(accelerator='gpu', devices=1, 
                      #log_every_n_steps=1,
                      #callbacks=[checkpoint_callback, lr_logger],
                      callbacks=[checkpoint_callback], 
                      max_epochs=num_epochs)
    #print("--- %s seconds (training) ---" % (time.time() - start_time))

    trainer.fit(
        model,
        train_loader,
        val_loader,
    )
    
    trainer.test(model, test_loader) # TODO
    #print("--- %s seconds (After test) ---" % (time.time() - start_time))


def evaluate_test_solo(ckpt_path, log_version):
    model = SemSegment.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    #hparams_file="/path/to/experiment/version/hparams.yaml",
    #map_location=None,
    )

    #checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss", mode="min")
    #lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(accelerator='gpu', devices=1)

    trainer.test(model, dataloaders=test_loader)
    

    #debug
    #print("stop")

    # test_dataloader = val_loader
    # trainer.test(dataloaders=test_dataloader, ckpt_path="best")

if __name__ == "__main__":
    #cli_main()

     # Import data with custom loader
    train_region = "estrie"
    test_region = "local_split"   # "local_split", "kenauk_2016", "kenauk_full" (old) 
    classif_mode = "multiclass"
    PIN_MEMORY = True
    NUM_WORKERS = 4
    BATCH_SIZE = 6
    num_epochs = 4
    optim_main = "sg"  # 'Ad' ou 'sg'
    lr_main = 0.001
    num_layers_main = 4
    input_channel_main = 24
    input_channel_lidar = 5 # 5 = pas de mnt, 6 = Full
    input_channel_radar = 6
    input_tile_size = 256 # Check size of output in test_epoch_end


    # all_mean = [259.971087045696, 277.3490067676725, 520.4650232890134, 342.23574780553645, 906.7611488412249, 2656.3582951694643, 3203.3543093369944, 3389.6250611778078, 3487.079600166239, 3555.416409200909, 1714.2260907527316, 828.2768740555728, 457.4229830346009, 501.79759875320303, 694.4711397083421, 835.1158882308216, 1219.9447441650816, 1823.0661322180392, 2064.6505317461747, 2316.1887302003915, 2363.5869859139643, 2359.4662122932396, 2390.6124116260303, 1586.6126304451745, -15.479797, -9.211855, 6.267961, -15.0310545, -9.519093, 5.5120163])
    # all_std  = [525.5551122108338, 526.4768589585602, 515.8903727938966, 527.3656790023017, 561.5222503677404, 836.1454714836563, 984.9190349745415, 1067.0420278801334, 1026.7569263359944, 1066.123618103052, 630.0584359871733, 505.2076063419134, 169.44646075504082, 249.03030944938908, 293.96819726121373, 408.20429488371605, 392.1811051266158, 492.36521601358254, 550.8773405439316, 623.9017038640061, 590.0457818993959, 540.556974947324, 740.4564895487368, 581.7629650224691])
    
    # # Sentinel 1
    # s1_e_p_mean = torch.tensor([-15.479797, -9.211855, 6.267961, -15.0310545, -9.519093, 5.5120163])
    # s1_e_p_std  = torch.tensor([1.622046, 1.8651232, 1.2285297, 2.1044014, 1.9065734, 1.37706]) 

    # # Lidar
    # estrie_lidar_mean = torch.tensor([7.798849, 5.5523205, 0.0029951811, 0.06429929, 6.7409873])
    # estrie_lidar_std  = torch.tensor([7.033332, 5.196636, 1.0641352, 0.06102526, 3.182435])

    # train_transform = transforms.Compose([
    #     transforms.Normalize()
    # ])

    # Call the loaders
    train_loader, val_loader, test_loader = get_datasets(
    train_region,
    test_region,
    classif_mode,
    BATCH_SIZE,
    # train_transform,
    # val_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
    )

    # Training + test (main function)
    cli_main()

    # Evaluate #TODO automatiser les paths
    # ckpt_path = "/mnt/Data/01_Codes/00_Github/Unet_lightning/lightning_logs/version_133/checkpoints/epoch=21-step=20856.ckpt"
    # log_version = 133
    # evaluate_test_solo(ckpt_path, log_version)
