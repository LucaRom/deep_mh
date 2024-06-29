import gc
import logging
import sys

import hydra
import matplotlib.pyplot as plt
import optuna
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from datamodule_bin_mul import EstrieDataModule
from seg_3_enco_2loss import SemSegment2loss

logger = logging.getLogger(__name__)

# Set a seed for reproducibility
SEED = 1234
seed_everything(SEED, workers=True)

def set_test_mask(cfg, mask_name, generate_outputs=False):
    cfg.model.test_mask_dir = mask_name
    cfg.model.test_mode=True

    cfg.model.generate_cm_sample = generate_outputs
    cfg.model.generate_tif_sample = generate_outputs
    cfg.model.generate_fig_sample = generate_outputs

    new_data_module = EstrieDataModule(
        input_format=cfg.model.input_format,
        classif_mode=cfg.model.classif_mode,
        train_mask_dir=cfg.model.train_mask_dir,
        val_mask_dir=cfg.model.val_mask_dir,
        test_mask_dir=cfg.model.test_mask_dir,
        batch_size=cfg.model.batch_size,
        dataset_size=cfg.model.dataset_size,
        train_transforms=cfg.model.train_transforms,
        test_mode=cfg.model.test_mode,
        sensors=cfg.model.sensors,
        opt_bands=cfg.model.opt_bands,
        lidar_bands=cfg.model.lidar_bands,
        sar_bands=cfg.model.sar_bands,
        indices_lst=cfg.model.indices_lst,
        num_workers=cfg.model.num_workers,
        pin_memory=False,
        debug_mode=cfg.debug
        )

    new_test_loader = new_data_module.create_dataloaders()

    return new_data_module, new_test_loader

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    # Print command line used to run training
    command_line = ' '.join(sys.argv)
    logger.info(f"Launched with command line: {command_line}")

    print(cfg)  # Optionally print the full config
    
    if cfg.get('optimize', False):
        print('outch!')
    else:
        # load_dataloders
        print('Loading data module...')
        data_module = EstrieDataModule(
        input_format=cfg.model.input_format,
        classif_mode=cfg.model.classif_mode,
        train_mask_dir=cfg.model.train_mask_dir,
        val_mask_dir=cfg.model.val_mask_dir,
        test_mask_dir=cfg.model.test_mask_dir,
        batch_size=cfg.model.batch_size,
        dataset_size=cfg.model.dataset_size,
        train_transforms=cfg.model.train_transforms,              
        test_mode=cfg.model.test_mode,
        sensors=cfg.model.sensors,
        opt_bands=cfg.model.opt_bands,
        lidar_bands=cfg.model.lidar_bands,
        sar_bands=cfg.model.sar_bands,
        indices_lst=cfg.model.indices_lst,
        num_workers=cfg.model.num_workers,
        pin_memory=cfg.model.pin_memory,
        #debug_mode=cfg.debug
        )
        print('Data module built!')

        train_loader, val_loader, test_loader = data_module.create_dataloaders()

        checkpoint_callback_top = ModelCheckpoint(save_top_k=4, monitor="val_loss", mode="min", filename="topk_{epoch:02d}-{val_loss:.2f}", save_on_train_epoch_end=True)
        checkpoint_callback_epoch = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=-1, every_n_epochs=5, save_on_train_epoch_end=True)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stopping = EarlyStopping(monitor="val_loss", patience=30, verbose=False, mode="min")

        trainer = Trainer(
            accelerator='gpu', devices=1,
            callbacks=[checkpoint_callback_top, checkpoint_callback_epoch, lr_monitor, early_stopping],
            max_epochs=cfg.model.num_epochs,
            min_epochs=30
        )

        lightning_log_num = trainer.logger.version
        logger.info(f"Lightning log version number : {lightning_log_num}")

        train_weights = data_module.train_class_weights
        print('WEIGHTS:', train_weights)
        model = SemSegment2loss(cfg=cfg.model, class_weights=train_weights)

        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, test_loader)

        # Load M2 mask (buffer) for test set only
        mask_name = "mask_multiclass_3223_buff"
        _, buff_test_loader = set_test_mask(cfg, mask_name, generate_outputs=False)
        trainer.test(model, buff_test_loader)

if __name__ == "__main__":
    main()
