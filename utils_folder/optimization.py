import optuna
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from datamodule import EstrieDataModule
from seg_3_enco_multi import SemSegment


def optuna_objective(cfg: DictConfig, trial: optuna.Trial):
    # Suggest hyperparameters
    cfg.model.learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    cfg.model.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Create the data module and model with the updated configuration
    data_module = EstrieDataModule(cfg=cfg.model)
    model = SemSegment(cfg=cfg.model)

    # Trainer setup
    trainer = Trainer(
        accelerator='gpu', devices=1,
        callbacks=[
            ModelCheckpoint(dirpath="checkpoints", save_top_k=5, monitor="val_loss", mode="min"),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(monitor="val_loss", patience=10)
        ],
        max_epochs=cfg.model.num_epochs
    )

    # Training
    train_loader, val_loader, _ = data_module.create_dataloaders()
    trainer.fit(model, train_loader, val_loader)

    # Return the metric to minimize/maximize
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss