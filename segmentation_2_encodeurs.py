from argparse import ArgumentParser

import time
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
from unet_multi import UNet_multi
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# Custom LR
from utils import get_loaders, get_loaders_kenauk_multi_enco

# Add to specified some tensor to GPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

start_time_glob = time.time()

class SemSegment(LightningModule):
    def __init__(
        self,
        #lr: float = 0.001,
        #num_classes: int = 19,
        num_classes: int = 1,
        #num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = True,

    # ## Kitty test parameters
    #     lr: float = 0.01,
    #     num_classes: int = 19,
    #     num_layers: int = 5,
    #     features_start: int = 64,
    #     bilinear: bool = False,

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
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score()
        self.val_f1 = torchmetrics.F1Score()

        # Model
        self.net = UNet_multi(
            num_classes=num_classes,
            input_channels=input_channel_main,
            input_channels_lidar=input_channel_lidar,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    # def forward(self, x):
    #     return self.net(x)

    def forward(self, x, y):
        return self.net(x, y)

    def on_train_start(self):
        tensorboard = self.logger.experiment
        tensorboard.add_text("epochs", str(num_epochs))
        tensorboard.add_text("batch_size", str(BATCH_SIZE))                                       
        tensorboard.add_text("optimizer", optim_main)                                       
        tensorboard.add_text("learning_rate", str(lr_main))                                       
        tensorboard.add_text("layers_unet", str(num_layers_main))
        tensorboard.add_text("input_ch", str(input_channel_main))

    def training_step(self, batch, batch_nb):
        # img, mask = batch
        # img = img.float()
        # mask = mask.long()
        # out = self(img)

        # img, mask = batch   # x, y
        # #img = img.float()   # x
        # #mask = mask.long()
        # preds = self(img)   # predictions

        img, lidar, mask, img_path = batch # img_poth only needed in test, but still carried 
                                           # TODO fix with a if "test" in dataset...

        #with torch.cuda.amp.autocast():
        preds = self(img, lidar)

        mask_loss = mask.float().unsqueeze(1)
        preds_loss = preds.float()
        preds_sig = torch.sigmoid(preds)

        # x, y = batch
        # preds = self(x)

        # Train metrics call
        #loss_val = F.cross_entropy(out, mask, ignore_index=250)
        #train_loss = F.binary_cross_entropy_with_logits(preds_loss, mask_loss)

        train_loss  = torch.nn.BCEWithLogitsLoss()(preds, mask_loss)
        mask_loss = mask_loss.type(torch.IntTensor).to(device=device)

        train_accu = self.train_accuracy(preds_sig, mask_loss)
        train_f1   = self.train_f1(preds_sig, mask_loss)
        
        # Metric dictionnary
        log_dict = {"train_loss": train_loss, "train_accu": train_accu, "train_f1": train_f1}

        # Call to log
        self.log("train_loss", train_loss)
        self.log("train_f1", train_f1)
        self.log("train_accu", train_accu, prog_bar=True)

        #return {"loss": train_loss, "log": log_dict, "progress_bar": log_dict}
        #return {"log": log_dict, "progress_bar": log_dict}
        return {"loss": train_loss, "log": log_dict}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().detach().cpu()
        self.log('train_loss_avg', avg_loss, prog_bar=True, logger=True, on_epoch=True) 

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

        img, lidar, mask, img_path = batch  #img_path only needed in test
        img = img.float()   # x
        lidar = lidar.float()
        mask = mask.long()  # y 
        preds = self(img, lidar)   # predictions

        #confmat = ConfusionMatrix(num_classes=2).to(device=device)
        #self.conf_print = confmat(preds, mask)
        #print(conf_print)

        mask_loss = mask.float().unsqueeze(1)
        preds_sig = preds.sigmoid()

        #loss_val = F.cross_entropy(out, mask, ignore_index=250)
        #loss_val = F.binary_cross_entropy_with_logits(preds, mask_loss)
        val_loss  = torch.nn.BCEWithLogitsLoss()(preds, mask_loss)

        mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
        val_accu = self.val_accuracy(preds_sig, mask_loss)
        val_f1   = self.val_f1(preds_sig, mask_loss)


        self.log("val_loss", val_loss)
        self.log("val_f1", val_f1)
        self.log("val_accu", val_accu, prog_bar=True)

        return {"val_loss": val_loss, "val_acc": val_accu}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean().detach().cpu()
        self.log('val_loss_avg', loss_val, on_epoch=True) 

        log_dict = {"val_loss": loss_val}

        #print(self.conf_print)

        return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}


    def test_step(self, batch, batch_idx):
        self.trainer.model.eval()
        x, y, z, img_path = batch
        x = x.float()   # x
        y = y.float()   # y 
        z = z.long()    # y 

        preds = self(x, y)   # predictions
        self.trainer.model.train()

        preds_temp   = np.multiply((preds.sigmoid().cpu() > 0.5),1)
        preds_recast = preds_temp.type(torch.IntTensor).to(device=device)     

        confmat = ConfusionMatrix(num_classes=2).to(device=device)
        conf_print = confmat(preds_recast, z)

        # mask_loss = mask.float().unsqueeze(1) # Unsqueeze for BCE

        # test_loss  = torch.nn.BCEWithLogitsLoss()(preds, mask_loss)

        # mask_loss = mask.float().unsqueeze(1)
        # preds_sig = preds.sigmoid()

        # #loss_val = F.cross_entropy(out, mask, ignore_index=250)
        # #loss_val = F.binary_cross_entropy_with_logits(preds, mask_loss)
        # val_loss  = torch.nn.BCEWithLogitsLoss()(preds, mask_loss)

        # mask_loss = mask_loss.type(torch.IntTensor).to(device=device)
        # val_accu = self.val_accuracy(preds_sig, mask_loss)
        # val_f1   = self.val_f1(preds_sig, mask_loss)


        # loss = nn.CrossEntropyLoss(weight=self.weights.to(device=self.device))(preds, y)
        # # self.log('test_loss', loss, prog_bar=True, logger=True,on_step=True, on_epoch=True) # , sync_dist=True
        # loss = loss.detach().cpu()
        # # preds = preds.cpu().detach()
        # preds = preds.detach().cpu()
        # # y = y.cpu().detach()
        # y = y.detach().cpu()
        # y = y.type(torch.IntTensor)
    
        # # prec_score = precision_score(y.numpy().flatten(), preds.softmax(dim=1).argmax(dim=1).numpy().flatten(), average='weighted' )
        # # rec_score = recall_score(y.numpy().flatten(), preds.softmax().argmax(dim=1).numpy().flatten(), average='weighted')
    
        # self.pr = np.append(self.pr.flatten(), preds.softmax(dim=1).argmax(dim=1).numpy().flatten())
        # self.yr = np.append(self.yr.flatten(), y.numpy().astype(np.int32).flatten())
        # # gc.collect()
        # #        print (y_reel)
    
        #return {'test_loss': loss, 'test_preds': preds, 'test_target': y}
        return {'conf matrice': conf_print, 'preds' : preds, 'x' : x, 'y' : y, 'z' : z, 'img_path' : img_path}

    def test_epoch_end(self, outputs):
        # TODO Add logs to test aswell?

        # get confusion matrix
        # for x in range(len(outputs)):
        #     outputs_confmat = outputs[x]['conf matrice'].cpu().numpy()
        #     row_to_add = np.array([0,1])
        #     results = np.vstack(( row_to_add, outputs_confmat))
        #     print("\nConfusion matrix")
        #     print(results)

        # print outputs
        # predict_sig = outputs[0]['preds'][2].cpu().squeeze().sigmoid().numpy()
        # predict_sig = np.multiply((predict_sig > 0.5),1)
        # plt.imshow(predict_sig)

        for x in range(len(outputs)):
            fig = plt.figure()
            cm = outputs[x]['conf matrice'].cpu().numpy()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig("lightning_logs/version_{version}/cm_{num}.png".format(version = self.trainer.logger.version, num = x))
            plt.close(fig)

            # Extract CRS and transforms
            img_path = outputs[x]['img_path']
            src = rasterio.open(img_path[0])
            sample_crs = src.crs
            transform_ori = src.transform
            src.close() # Needed?

            ori_input = outputs[x]['x'][0].cpu().numpy()
            ori_target = outputs[x]['z'][0].cpu().numpy()
            #predict_sig = outputs[sample]['preds'][].cpu().squeeze().sigmoid().numpy()
            predict_sig = outputs[x]['preds'][0].cpu().squeeze().sigmoid().numpy()
            predict_sig = np.multiply((predict_sig > 0.5),1)

            # write predict image to file
            tiff_save_path = "lightning_logs/version_{version}/predict_geo_{num}.tif".format(version = self.trainer.logger.version, num = x)
            predict_img = rasterio.open(tiff_save_path, 'w', driver='GTiff',
                            height = 512, width = 512,
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

            # fig = plt.figure()
            # plt.subplot(1,4,1)
            # plt.imshow(np.transpose(ori_input[[3,2,1],:,:],(1,2,0))*3)
            # plt.title("Input")
            # plt.subplot(1,4,2)
            # plt.imshow(predict_sig)
            # plt.title("Predict")
            # plt.subplot(1,4,3)
            # plt.imshow(ori_target)
            # plt.title("Target")
            # plt.subplot(1,4,3)
            # plt.show()
            # plt.title("Confusion Matrix")

            plt.savefig("lightning_logs/version_{version}/fig_{num}.png".format(version = self.trainer.logger.version, num = x))
            plt.close(fig)

        
        #plt.imshow(np.transpose(ori_input[[3,2,1],:,:],(1,2,0))*3) # Show source
        #plt.imshow(predict_sig) # show predicted value
        #plt.imshow(ori_taget) # show target


        #plt.show()

        #print("debug")

        # for batch in outputs:
        #     for sample in range(len(batch)): 
        #         print(batch)
        #         print(sample)
        #         print("next")
        #         ori_input = outputs[sample]['x'].cpu().numpy()
        #         ori_taget = outputs[sample]['y'].cpu().numpy()
        #         #predict_sig = outputs[sample]['preds'][].cpu().squeeze().sigmoid().numpy()
        #         predict_sig = outputs[sample]['preds'].cpu().squeeze().sigmoid().numpy()

        #unique, counts = np.unique(np.multiply((predict_sig > 0.5),1), return_counts=True)

        #print('debug')
        #print(outputs)
    
    #     # avg_loss = torch.stack([x['test_loss'] for x in outputs]).detach().mean()
    #     preds = torch.cat([tmp['test_preds'] for tmp in outputs]).detach()
    #     targets = torch.cat([tmp['test_target'] for tmp in outputs]).detach()

    #     # ------ Confusion Matrix SKLEARN / Pandas ------
    #     sklearn_cm_display = cfg.new_plot_cm(targets.numpy().astype(int).flatten(),
    #                                          preds.softmax(dim=1).argmax(dim=1).numpy().flatten())
    #     plt.close(sklearn_cm_display)

    def configure_optimizers(self):
        if optim_main == 'Ad':
            opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

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

def cli_main():
    #from pl_bolts.datamodules import KittiDataModule

    seed_everything(1234)

    # parser = ArgumentParser()
    # # trainer args
    # parser = Trainer.add_argparse_args(parser)
    # # model args
    # parser = SemSegment.add_model_specific_args(parser)
    # # datamodule args
    # parser = KittiDataModule.add_argparse_args(parser)

    #args = parser.parse_args()

    # model
    #model = SemSegment(**args.__dict__)
    model = SemSegment()
    #model = UNet(num_classes=2)

    """ 

    # Sanity check
    inputs, labels = next(iter(train_loader))
    #print(inputs, labels)

    #print(img_path, mask_path)
    
    print(inputs.shape)
    print(inputs.squeeze().shape)

    print(type(inputs))

    inputs = torch.squeeze(inputs)

    print(type(inputs))

    print("stop")

    #f, axarr = plt.subplots(3,2, figsize=(15,15))
    f, axarr = plt.subplots(1,2, figsize=(15,15))
    #axarr[0].imshow(inputs[:,[3,2,1]][0].numpy().transpose([2, 1, 0])*3)
    axarr[0].imshow(inputs[[3,2,1],:,:,][0].numpy().transpose()*2)
    axarr[1].imshow(labels[0])
    # axarr[0,0].imshow(inputs[:,[3,2,1]][0].permute(1,2,0).numpy()*3)
    # axarr[0,1].imshow(labels[0])
    # axarr[1,0].imshow(inputs[:,[3,2,1]][1].permute(1,2,0).numpy()*3)
    # axarr[1,1].imshow(labels[0])
    # axarr[2,0].imshow(inputs[:,[3,2,1]][2].permute(1,2,0).numpy()*3)
    # axarr[2,1].imshow(labels[2])
    plt.show()

    """


    # for i in range(3):
    #     # plt.subplot(1, 3, i+1)
    #     # plt.axis('off')
    #     plt.imshow(labels[i])
    #     plt.figure(i+1)

    #plt.imshow(inputs[:,[3,2,1]][0].permute(1,2,0).numpy()*3)

    #plt.show()

    # #imgplot=plt.imshow(mask)
    # for i in range(3):
    #     return plt.imshow(labels[i])
    
    print("stop")

    # train (kitti)
    #F:\00_Donnees_SSD\03_existing_datasets\data_semantics
    # data_dir='F:/00_Donnees_SSD/03_existing_datasets/data_semantics'
    # dm = KittiDataModule(data_dir)

    # trainer = Trainer(accelerator='gpu', devices=1, log_every_n_steps=1, max_epochs=50)
    # trainer.fit(model, datamodule=dm)

    # # Name of experiment for logs # TODO
    # logger = TensorBoardLogger("tb_logs", name="my_model")

    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss", mode="min")

    # # train (custom)
    start_time = time.time()
    trainer = Trainer(accelerator='gpu', devices=1, 
                      log_every_n_steps=1,
                      callbacks=[MyPrintingCallback(), checkpoint_callback], 
                      max_epochs=num_epochs)
    print("--- %s seconds (training) ---" % (time.time() - start_time))

    trainer.fit(
        model,
        train_loader,
        val_loader,
    )
    
    trainer.test(model, test_loader) # TODO
    print("--- %s seconds (After test) ---" % (time.time() - start_time))

    #debug
    #print("stop")

    # test_dataloader = val_loader
    # trainer.test(dataloaders=test_dataloader, ckpt_path="best")

if __name__ == "__main__":
    #cli_main()

     # Import data with custom loader
    TRAIN_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/train"
    TRAIN_MASK_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/train"
    TRAIN_MNT_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/train"
    VAL_IMG_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/sen2_print/val"
    VAL_MASK_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/mask_bin/val"
    VAL_MNT_DIR = "D:/00_Donnees/01_trainings/mh_sentinel_2/lidar_mnt/val"
    PIN_MEMORY = True
    NUM_WORKERS = 6
    BATCH_SIZE = 2
    num_epochs = 50
    optim_main = "Ad"  # 'Ad' ou 'sg'
    lr_main = 0.0001
    num_layers_main = 5
    input_channel_main = 13
    input_channel_lidar = 1

    # train_loader, val_loader, test_loader = get_loaders(
    # TRAIN_IMG_DIR,
    # TRAIN_MASK_DIR,
    # VAL_IMG_DIR,
    # VAL_MASK_DIR,
    # TRAIN_MNT_DIR,
    # VAL_MNT_DIR,
    # BATCH_SIZE,
    # # train_transform,
    # # val_transforms,
    # NUM_WORKERS,
    # PIN_MEMORY,
    # )
    
    # Paths estrie
    e_img_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/sen2"
    e_mask_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_bin"
    #e_mask_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/mask_multi" # Multiclass
    e_lidar_dir = "D:/00_Donnees/01_trainings/02_mh_double_stack/estrie/lidar_mnt"

    # Path Kenauk Full (test)
    k_test_img = "D:/00_Donnees/01_trainings/03_kenauk_test_full/sen2_print"
    k_test_mask = "D:/00_Donnees/01_trainings/03_kenauk_test_full/mask_bin"
    k_test_lid = "D:/00_Donnees/01_trainings/03_kenauk_test_full/lidar_mnt"

    train_loader, val_loader, test_loader = get_loaders_kenauk_multi_enco(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    TRAIN_MNT_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    VAL_MNT_DIR,
    k_test_img,
    k_test_mask,
    k_test_lid,
    BATCH_SIZE,
    # train_transform,
    # val_transforms,
    NUM_WORKERS,
    PIN_MEMORY,
    )

    cli_main()