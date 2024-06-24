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

from datamodule import EstrieDataModule
from datamodule_pn import PortneufNordDataModule
from seg_3_enco_multi import SemSegment

logger = logging.getLogger(__name__)

# Set a seed for reproducibility
SEED = 1234
seed_everything(SEED, workers=True)

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    # Print command line used to run training
    command_line = ' '.join(sys.argv)
    logger.info(f"Launched with command line: {command_line}")

    print(cfg)  # Optionally print the full config

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

    data_module = PortneufNordDataModule(train_config_path=cfg)
    data_module.setup()

    train_weights = data_module.train_class_weights
    print('WEIGHTS:', train_weights)

    model = SemSegment(cfg=cfg.model, class_weights=train_weights)

    if cfg.mode == 'test':
        model_path = cfg.model_path
        model = model.load_from_checkpoint(model_path, cfg=cfg.model, class_weights=None, strict=False)
        test_loader = data_module.test_dataloader()
        trainer.test(model, test_loader)

    else:
        trainer.fit(model, data_module)
        test_loader = data_module.test_dataloader()
        trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
