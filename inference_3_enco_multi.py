from argparse import ArgumentParser

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

import matplotlib.pyplot as plt

#from pl_bolts.models.vision.unet import UNet
from unet_3enco_sum import unet_3enco_sum
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor

# Custom LR
from utils import get_datasets_inference, get_datasets

# Custom loss
from custom_loss import FocalLoss

# Add to specified some tensor to GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

start_time_glob = time.time()

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
        """Basic model for semantic segmentation. Uses UNet architecture by default.

        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

        Implemented by:

            - `Annika Brundyn <https://github.com/annikabrundyn>`_

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
        log_dict = {"train_loss": train_loss.detach(), "train_accu": train_accu, "train_f1": train_f1}

        # Call to log
        self.log("train_loss", train_loss, batch_size=BATCH_SIZE)
        self.log("train_f1", train_f1, batch_size=BATCH_SIZE)
        self.log("train_accu", train_accu, prog_bar=True, batch_size=BATCH_SIZE)

        #return {"loss": train_loss, "log": log_dict, "progress_bar": log_dict}
        #return {"log": log_dict, "progress_bar": log_dict}
        return {"loss": train_loss, "log": log_dict}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu()
        self.log('train_loss_avg', avg_loss, prog_bar=True, logger=True, on_epoch=True, batch_size=BATCH_SIZE) 

        # # Print learning rate monitor (maybe debug)
        # if len(self.trainer.callbacks[1].lrs['lr-Adam']) == 0:
        #     current_lr = lr_main
        # else:
        #     current_lr = self.trainer.callbacks[1].lrs['lr-Adam'][-1]
        # print(f'The current learning rate is : {current_lr}')

        current_train_time = time.time() - self.new_time
        print("--- %s seconds (one train epoch) ---" % (current_train_time))
        self.new_time = time.time()
        self.train_time_list.append(current_train_time)

        # # compute metrics
        # train_accuracy = self.train_accuracy.compute()
        # train_f1 = self.train_f1.compute()

        # # log metrics
        # self.log("epoch_train_accuracy", train_accuracy)
        # self.log("epoch_train_f1", train_f1)

        # # reset all metrics
        # self.train_accuracy.reset()
        # self.train_f1.reset()

    def on_train_end(self):
        print("Average train time per epoch : ", sum(self.train_time_list) / len(self.train_time_list))

    def validation_step(self, batch, batch_idx):
        # img, mask = batch
        # img = img.float()
        # mask = mask.long()
        # out = self(img)

        img, lidar, mask, radar, img_path = batch  #img_path only needed in test
        img = img.float()   # x
        lidar = lidar.float()
        mask = mask.long()  # y
        radar = radar.float()

        preds = self(img, lidar, radar)   # predictions

        #confmat = ConfusionMatrix(num_classes=2).to(device=device)
        #self.conf_print = confmat(preds, mask)
        #print(conf_print)

        #mask_loss = mask.float().unsqueeze(1)
        mask_loss = mask.float().unsqueeze(1)
        #preds_loss = preds.argmax(dim=1)

        #loss_val = F.cross_entropy(out, mask, ignore_index=250)
        #loss_val = F.binary_cross_entropy_with_logits(preds, mask_loss)
        #val_loss  = torch.nn.NLLLoss()(preds_sig, mask_loss)
        #val_loss = torch.nn.CrossEntropyLoss()(preds, mask)
        val_loss = FocalLoss()(preds, mask)

        preds_accu = preds.argmax(dim=1).unsqueeze(1)

        mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
        val_accu = self.val_accuracy(preds_accu, mask_loss)
        val_f1   = self.val_f1(preds_accu, mask_loss)


        self.log("val_loss", val_loss, batch_size=BATCH_SIZE)
        self.log("val_f1", val_f1, batch_size=BATCH_SIZE)
        self.log("val_accu", val_accu, prog_bar=True,batch_size=BATCH_SIZE)

        return {"val_loss": val_loss, "val_acc": val_accu}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
        #self.log('val_loss_avg', loss_val, on_epoch=True, batch_size=BATCH_SIZE)
        self.log('val_loss_avg', loss_val, prog_bar=True, logger=True, on_epoch=True, batch_size=BATCH_SIZE) 


        #log_dict = {"val_loss": loss_val}

        #print(self.conf_print)

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
        # TODO Add logs to test aswell?

        for x in range(len(outputs)):
            fig = plt.figure()
            cm = outputs[x]['conf matrice'].cpu().numpy()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig("lightning_logs/inference_{version}/cm_{num}.png".format(version = log_version, num = x))
            #plt.savefig("lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = log_version, num = x))
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
            tiff_save_path = "lightning_logs/inference_{version}/predict_geo_{num}.tif".format(version = log_version, num = x)
            #tiff_save_path = "lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = log_version, num = x)

            predict_img = rasterio.open(tiff_save_path, 'w', driver='GTiff',
                            height = input_tile_size, width = input_tile_size,
                            count=1, dtype=str(predict_sig.dtype),
                            crs=sample_crs,
                            transform=transform_ori)

            predict_img.write(predict_sig, 1)
            predict_img.close()

            fig = plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(np.transpose(ori_input[[3,2,1],:,:],(1,2,0))*3)
            plt.title("Input")
            plt.subplot(1,3,2)
            plt.imshow(predict_sig)
            plt.title("Predict")
            plt.subplot(1,3,3)
            plt.imshow(ori_target)
            plt.title("Target")

            plt.savefig("lightning_logs/inference_{version}/fig_{num}.png".format(version = log_version, num = x))
            #plt.savefig("lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = log_version, num = x))
            plt.close(fig)

        self.trainer.model.train()

    def configure_optimizers(self):
        if optim_main == 'Ad':
            opt = torch.optim.Adam(self.net.parameters(), weight_decay = 0.001, lr=self.lr)
        else:
            opt = torch.optim.SGD(self.net.parameters(), momentum = 0.9, weight_decay = 0.001, lr=self.lr)

        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.95, mode='min', patience=3, verbose=True)
        
        #return [opt], [sch]

        return {'optimizer': opt, "lr_scheduler" : {'scheduler': sch, 'interval':'epoch', 'frequency': 1, 'monitor': 'val_loss'}}
        #return {'optimizer': opt}

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    #     return {'optimizer': optimizer, 'frequency': 5, 'monitor': 'val_loss',
    #             'strict': True}  # 'lr_scheduler':scheduler,


    # implémentation optimiser basée sur julien
    # def configure_optimizers(self):
    #     # optimizer = Adam(self.resnet_model.parameters(), lr=1e-3)
    #     optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
    #     #sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #     # scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.05, last_epoch=-1,verbose=True)
    #     # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode= 'min', factor = 0.5,
    #     #                                           patience=10, min_lr=1e-8, verbose=False)

    #     return {'optimizer': optimizer, 'frequency': 5, 'monitor': 'val_loss',
    #             'strict': True}  


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

class MyPrintingCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        #print('do something when training ends')
        print("End of training")

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
    NUM_WORKERS = 8
    BATCH_SIZE = 6
    num_epochs = 10
    optim_main = "sg"  # 'Ad' ou 'sg'
    lr_main = 0.001
    num_layers_main = 4
    input_channel_main = 24
    input_channel_lidar = 5
    input_channel_radar = 6
    input_tile_size = 256 # Check size of output in test_epoch_end

    # # Call the loaders
    # train_loader, val_loader, test_loader = get_datasets_inference(
    # train_region,
    # test_region,
    # classif_mode,
    # BATCH_SIZE,
    # # train_transform,
    # # val_transforms,
    # NUM_WORKERS,
    # PIN_MEMORY,
    # )

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

    # Evaluate #TODO automatiser les paths
    #ckpt_path = "/mnt/Data/01_Codes/00_Github/Unet_lightning/lightning_logs/version_145/checkpoints/epoch=49-step=47400.ckpt"
    #ckpt_path = "/mnt/Data/01_Codes/00_Github/Unet_lightning/lightning_logs/version_151/checkpoints/epoch=9-step=9480.ckpt"
    ckpt_path = "/mnt/Data/01_Codes/00_Github/Unet_lightning/lightning_logs/version_182/checkpoints/epoch=46-step=40091.ckpt"
    log_version = "182_kenauk_2020_4"

    evaluate_test_solo(ckpt_path, log_version)