import gc
import logging
import sys
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from torch.utils.data import DataLoader, Dataset
import torch

from datamodule_new import PortneufNordDataModule
from seg_3_enco_multi import SemSegment

# Set the sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)

# Set a seed for reproducibility
SEED = 1234
seed_everything(SEED, workers=True)

def plot_importance(accuracies, f1_scores, feature_names):
    x = range(len(feature_names))
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(x, accuracies)
    plt.xlabel('Feature')
    plt.ylabel('Decrease in Accuracy')
    plt.title('Permutation Importance (Accuracy)')
    plt.xticks(x, feature_names, rotation=90)
    
    plt.subplot(1, 2, 2)
    plt.bar(x, f1_scores)
    plt.xlabel('Feature')
    plt.ylabel('Decrease in F1-Score')
    plt.title('Permutation Importance (F1-Score)')
    plt.xticks(x, feature_names, rotation=90)
    
    plt.tight_layout()
    plt.show()

def permute_band(batch, band_index, data_type):
    data, lidar, mask, radar, img_path = batch

    # Print shapes for debugging
    # print(f"Data shape: {data.shape}")
    # print(f"Lidar shape: {lidar.shape}")
    # print(f"Radar shape: {radar.shape}")
    # print(f"Mask shape: {mask.shape}")

    if data_type == "optical":
        original = data[band_index, :, :].clone()
        data[band_index, :, :] = data[band_index, torch.randperm(data.size(1)), :]
        permuted = data[band_index, :, :]

    elif data_type == "lidar":
        original = lidar[band_index, :, :].clone()
        lidar[band_index, :, :] = lidar[band_index, torch.randperm(lidar.size(1)), :]
        permuted = lidar[band_index, :, :]

    elif data_type == "sar":
        original = radar[band_index, :, :].clone()
        radar[band_index, :, :] = radar[band_index, torch.randperm(radar.size(1)), :]
        permuted = radar[band_index, :, :]

    # print(f"Original {data_type} band {band_index}:", original)
    # print(f"Permuted {data_type} band {band_index}:", permuted)

    return data, lidar, mask, radar, img_path

class PermutedDataset(Dataset):
    def __init__(self, original_dataset, band_index, data_type):
        self.original_dataset = original_dataset
        self.band_index = band_index
        self.data_type = data_type

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        batch = self.original_dataset[idx]
        return permute_band(batch, self.band_index, self.data_type)

def run_permutation_test(cfg, band_type, band_index, band_label, accuracies, f1_scores, feature_names):
    logger.info(f"Running permutation test for {band_label} {band_index}")

    data_module = PortneufNordDataModule(train_config_path=cfg, dataset_config_path=cfg.model.aoi_conf)
    data_module.setup(stage='test')
    
    model = SemSegment(cfg=cfg.model, class_weights=None)
    checkpoint_path = cfg.model_path

    # Create test loader
    test_loader = data_module.test_dataloader()

    # Create permuted dataset and loader
    permuted_dataset = PermutedDataset(test_loader.dataset, band_index, band_type)
    permuted_loader = DataLoader(permuted_dataset, batch_size=test_loader.batch_size, pin_memory=True, num_workers=0, shuffle=False)

    # Test model with permuted data
    model = model.load_from_checkpoint(checkpoint_path, cfg=cfg.model, class_weights=None, strict=False)

    model.accu_test = 0  # Resetting for accurate measurement
    model.f1_test = 0  # Resetting for accurate measurement

    trainer = Trainer(
        accelerator='gpu', devices=1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=30, verbose=False, mode="min")],
        max_epochs=cfg.model.num_epochs,
        min_epochs=30
    )

    # permuted_accuracies = []
    # permuted_f1_scores = []

    # for batch in test_loader:
    #     permuted_batch = permute_band(batch, band_index, band_type)
    #     data, lidar, mask, radar, img_path = permuted_batch
    with torch.no_grad():
        trainer.test(model, permuted_loader, verbose=False)

        permuted_accuracy = model.accu_test
        permuted_f1 = model.f1_test

    accuracies.append(permuted_accuracy)
    f1_scores.append(permuted_f1)
    feature_names.append(f'{band_label} {band_index}')

    logger.info(f"{band_label} {band_index} - Permuted Accuracy: {permuted_accuracy}, Permuted F1-Score: {permuted_f1}")

    # Clear memory and clean up
    del permuted_loader
    del permuted_dataset
    del data_module
    del model
    torch.cuda.empty_cache()
    gc.collect()

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    # Print command line used to run training
    command_line = ' '.join(sys.argv)
    logger.info(f"Launched with command line: {command_line}")

    print(cfg)  # Optionally print the full config

    accuracies = []
    f1_scores = []
    feature_names = []

    # Calculate band ranges from configuration
    optical_bands_count = len(cfg.model.opt_bands) * 2 + len(cfg.model.indices_lst) * 2  # Two seasons for each
    lidar_bands_count = len(cfg.model.lidar_bands)
    sar_bands_count = len(cfg.model.sar_bands) * 2  # Two seasons for SAR

    for band_type, band_count, band_label in zip(
            ["optical", "lidar", "sar"],
            [optical_bands_count, lidar_bands_count, sar_bands_count],
            ["Optical Band", "Lidar Band", "SAR Band"]
    ):
        for i in range(band_count):
            run_permutation_test(cfg, band_type, i, band_label, accuracies, f1_scores, feature_names)

    plot_importance(accuracies, f1_scores, feature_names)

if __name__ == "__main__":
    main()
