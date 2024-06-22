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
from seg_3_enco_multi import SemSegment

logger = logging.getLogger(__name__)

# Set a seed for reproducibility
SEED = 1234
seed_everything(SEED, workers=True)

def plot_band_distributions_and_count(dataloaders, num_bands):
    """
    Plots the distribution of values for each band in the dataset and prints the count of processed items.

    Args:
    - dataloaders (dict): A dictionary containing 'train', 'val', and 'test' DataLoader objects.
    - num_bands (int): The number of bands in the dataset images.
    """
    # Initialize a dictionary to hold the aggregated data and item counts
    aggregated_data = {phase: {band: [] for band in range(num_bands)} for phase in dataloaders}
    item_counts = {phase: 0 for phase in dataloaders}

    # Aggregate data for each band and count items
    for phase, loader in dataloaders.items():
        for img, lidar, mask, radar, img_path in loader:  # Assuming the second element is the target
            item_counts[phase] += img.shape[0]  # Increment count by batch size
            for band in range(num_bands):
                # Flatten the band data and convert to numpy for easier processing
                band_data = img[:, band, :, :].flatten().cpu().numpy()
                aggregated_data[phase][band].extend(band_data)

    # Print item counts
    for phase, count in item_counts.items():
        print(f"Total items processed in {phase} dataset: {count}")

    # Plot distribution for each band
    for band in range(num_bands):
        plt.figure(figsize=(10, 6))
        for phase in dataloaders:
            data = aggregated_data[phase][band]
            plt.hist(data, bins=50, alpha=0.5, label=f'{phase} - Band {band + 1}')
        
        plt.legend(loc='upper right')
        plt.title(f'Distribution of Band {band + 1} Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

# def load_model(cfg, data_module=None):
#     model_path = cfg.model_path
#     class_weights = None if cfg.mode == 'test' else data_module.class_weights if data_module else None
#     return model.load_from_checkpoint(model_path, cfg=cfg.model, class_weights=class_weights, strict=False)

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

# def test_model(trainer, model, test_loader, new_test_loader=None):
#     trainer.test(model, test_loader)
#     if new_test_loader:
#         trainer.test(model, new_test_loader)

# def optuna_objective(cfg: DictConfig, trial: optuna.Trial):

#     total_opt_bands = len(cfg.model.opt_bands)
#     band_to_exclude = trial.number % total_opt_bands
#     cfg.model.opt_bands = [band for i, band in enumerate(cfg.model.opt_bands) if i != band_to_exclude]

#     # CHECK YOUR NOTES, NO TRANSFORMS

#     # Create the data module and model with the updated configuration
#     print('Loading data module...')
#     data_module = EstrieDataModule(
#         input_format=cfg.model.input_format,
#         classif_mode=cfg.model.classif_mode,
#         train_mask_dir=cfg.model.train_mask_dir,
#         val_mask_dir=cfg.model.val_mask_dir,
#         test_mask_dir=cfg.model.test_mask_dir,
#         batch_size=cfg.model.batch_size,
#         dataset_size=cfg.model.dataset_size,
#         train_transform=None,              
#         test_mode=cfg.model.test_mode,
#         sensors=cfg.model.sensors,
#         opt_bands=cfg.model.opt_bands,
#         lidar_bands=cfg.model.lidar_bands,
#         sar_bands=cfg.model.sar_bands,
#         indices_lst=cfg.model.indices_lst,
#         num_workers=cfg.model.num_workers,
#         pin_memory=cfg.model.pin_memory,
#         debug_mode=cfg.debug
#         )
#     print('Data module built!')
    
#     model = SemSegment(cfg=cfg.model, class_weights=None)

#     # Trainer setup
#     trainer = Trainer(
#         accelerator='gpu', devices=1,
#         callbacks=[
#             ModelCheckpoint(dirpath="checkpoints", save_top_k=5, monitor="val_loss", mode="min"),
#             LearningRateMonitor(logging_interval='step'),
#             EarlyStopping(monitor="val_loss", patience=7, verbose=False, mode="min")
#         ],
#         max_epochs=cfg.model.num_epochs
#     )

#     # Training
#     train_loader, val_loader, _ = data_module.create_dataloaders()
#     trainer.fit(model, train_loader, val_loader)

#     # Return the metric to minimize/maximize
#     val_loss = trainer.callback_metrics["val_loss"].item()

#     return val_loss

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    # Print command line used to run training
    command_line = ' '.join(sys.argv)
    logger.info(f"Launched with command line: {command_line}")

    print(cfg)  # Optionally print the full config
    
    if cfg.get('optimize', False):
        study = optuna.create_study(direction='minimize', study_name='Optical bands', storage='sqlite:///optical_bands.db')
        study.optimize(lambda trial: optuna_objective(cfg, trial), n_trials=len(cfg.model.opt_bands))
        print("Best trial:", study.best_trial.params)

    else:

        # CHECK YOUR NOTES, NO TRANSFORMS # Recheck your notes for transform? or the code at least

        #train_transforms = None

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
        debug_mode=cfg.debug
        )
        print('Data module built!')

        train_loader, val_loader, test_loader = data_module.create_dataloaders()

        # # Use your provided DataLoaders
        # dataloaders = {
        #     'train': train_loader,
        #     'val': val_loader,
        #     'test': test_loader
        # }

        # # Specify the number of bands in your images
        # num_bands = 3  # Update this to match the actual number of bands in your dataset

        # # Plot the distributions and print the counts
        # plot_band_distributions_and_count(dataloaders, num_bands)

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

        print('WEIGHTS:', data_module.class_weights)
        model = SemSegment(cfg=cfg.model, class_weights=data_module.class_weights)
        #model = SemSegment(cfg=cfg.model, class_weights=None)

        if cfg.mode == 'test':
            model_path = cfg.model_path
            model = model.load_from_checkpoint(model_path, cfg=cfg.model, class_weights=None, strict=False)
            trainer.test(model, test_loader)

            # Load M2 mask (buffer) for test set only
            mask_name = "mask_multiclass_3223_buff"
            _, buff_test_loader = set_test_mask(cfg, mask_name, generate_outputs=False)
            trainer.test(model, buff_test_loader)

            # 2 dataloaders = problème d'impression des CM avec le nom du masque
            #dataloaders = [test_loader, new_test_loader]

            # needs to add argument dataloader_idx in test_step()
            #trainer.test(model, dataloaders)

            # for loader in dataloaders:
            #     gc.collect()
            #     trainer.test(model, loader)

        elif cfg.mode == 'resume':
            #model_path = cfg.model_path
            #model = model.load_from_checkpoint(model_path, cfg=cfg.model, class_weights=data_module.class_weights, strict=False)
            model_path = cfg.model_path
            new_trainer = Trainer(
                resume_from_checkpoint=model_path,
                accelerator='gpu', devices=1,
                callbacks=[checkpoint_callback_top, checkpoint_callback_epoch, lr_monitor, early_stopping],
                max_epochs=cfg.model.num_epochs,
                min_epochs=30
             )
            new_trainer.fit(model, train_loader, val_loader, ckpt_path=cfg.model_path)
            new_trainer.test(model, test_loader)

            # Load M2 mask (buffer) for test set only
            mask_name = "mask_multiclass_3223_buff"
            _, buff_test_loader = set_test_mask(cfg, mask_name, generate_outputs=False)
            trainer.test(model, buff_test_loader)

        elif cfg.mode == 'finetune':
            model_path = cfg.model_path
            model = model.load_from_checkpoint(model_path, cfg=cfg.model, class_weights=data_module.class_weights, strict=False)
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, test_loader)

            # Load M2 mask for test set only
            # Load M2 mask (buffer) for test set only
            mask_name = "mask_multiclass_3223_buff"
            _, buff_test_loader = set_test_mask(cfg, mask_name, generate_outputs=False)
            trainer.test(model, buff_test_loader)

        else:
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, test_loader)

            # Load M2 mask (buffer) for test set only
            mask_name = "mask_multiclass_3223_buff"
            _, buff_test_loader = set_test_mask(cfg, mask_name, generate_outputs=False)
            trainer.test(model, buff_test_loader)

if __name__ == "__main__":
    main()
