import gc
import logging
import sys
import os
import csv

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from sklearn.metrics import accuracy_score, f1_score

#from datamodule import EstrieDataModule
from datamodule_new import PortneufNordDataModule
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

    data_module = PortneufNordDataModule(train_config_path=cfg, dataset_config_path=cfg.model.aoi_conf)
    data_module.setup()

    #train_weights = data_module.train_class_weights
    #print('WEIGHTS:', train_weights)

    model = SemSegment(cfg=cfg.model, class_weights=None)

    if cfg.mode == 'test_ckpts':
        checkpoint_dir = cfg.model.ckpts_dir  # Add this to your config
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]

        test_loader = data_module.test_dataloader()

        accuracy_scores = []
        f1_scores = []

        for checkpoint_file in checkpoint_files:
            model = model.load_from_checkpoint(checkpoint_file, cfg=cfg.model, class_weights=None, strict=False)
            logger.info(f"Testing with checkpoint: {checkpoint_file}")
            trainer.test(model, test_loader, verbose=False)

            avg_test_accu = model.accu_test
            avg_test_f1 = model.f1_test

            accuracy_scores.append(avg_test_accu)
            f1_scores.append(avg_test_f1)

            logger.info(f"Checkpoint: {checkpoint_file} - Accuracy: {avg_test_accu}, F1-Score: {avg_test_f1}")

        # Save the results to a CSV file
        csv_file = os.path.join(checkpoint_dir, 'checkpoint_scores.csv')
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Checkpoint', 'Accuracy', 'F1-Score'])
            for checkpoint_file, acc, f1 in zip(checkpoint_files, accuracy_scores, f1_scores):
                writer.writerow([os.path.basename(checkpoint_file), acc, f1])

        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.plot([os.path.basename(f) for f in checkpoint_files], accuracy_scores, label='Accuracy')
        plt.plot([os.path.basename(f) for f in checkpoint_files], f1_scores, label='F1-Score')
        plt.xlabel('Checkpoints')
        plt.ylabel('Scores')
        plt.title('Checkpoint Evaluation')
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot without displaying it
        plot_file = os.path.join(checkpoint_dir, 'checkpoint_evaluation.png')
        plt.savefig(plot_file)

    else:
        print('No mode specified')

if __name__ == "__main__":
    main()
