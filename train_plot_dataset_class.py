import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from datamodule import EstrieDataModule
from seg_3_enco_multi import SemSegment

# Set a seed for reproducibility
SEED = 1234
#SEED = 363254675374532
seed_everything(SEED, workers=True)

# def aggregate_segmentation_class_counts(dataloader, num_classes, ignore_class=7):
#     """Aggregate class counts from segmentation labels."""
#     class_counts = np.zeros((num_classes,))
#     for _, _, mask, _, _ in dataloader:
#         # Assuming labels are of shape [batch_size, height, width]
#         # and contain class indices for segmentation
#         for label in mask:
#             unique, counts = np.unique(label.numpy(), return_counts=True)
#             unique = unique.astype(int)

#             for u, c in zip(unique, counts):
#                 if u != ignore_class:  # Check if the class is not the one to ignore
#                     class_counts[u] += c
#     return class_counts

def calculate_segmentation_class_ratios_and_counts(dataloader, num_classes, ignore_class=None):
    """Calculate class ratios and return class counts from segmentation labels, optionally ignoring a specific class."""
    class_counts = np.zeros((num_classes,))
    for _, _, mask, _, _ in dataloader:
        for label in mask:
            unique, counts = np.unique(label, return_counts=True)
            unique = unique.astype(int)  # Convert unique class labels to integers

            for u, c in zip(unique, counts):
                if ignore_class is None or u != ignore_class:  # Check if the class is not the one to ignore
                    class_counts[u] += c

    total_counts = class_counts.sum()
    if ignore_class is not None:
        # Adjust total count by removing the ignored class count
        total_counts -= class_counts[ignore_class]

    # Calculate ratios
    class_ratios = class_counts / total_counts
    return class_ratios, class_counts

def plot_class_distributions(train_counts, val_counts, train_total_counts, val_total_counts, class_names=None):
    """Plot class distributions for training and validation datasets, including total counts."""
    indices = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(12, 8))
    train_bars = plt.bar(indices - width/2, train_counts, width, label='Training')
    val_bars = plt.bar(indices + width/2, val_counts, width, label='Validation')

    plt.ylabel('Ratios')
    plt.title('Class Distribution in Training and Validation Datasets')
    plt.xticks(indices, class_names)
    plt.legend()

    # Annotate bars with the total counts
    for bar, count in zip(train_bars, train_total_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(count)}', ha='center', va='bottom')
    for bar, count in zip(val_bars, val_total_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(count)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def calculate_segmentation_class_ratios(dataloader, num_classes, ignore_class=None):
    """Calculate class ratios from segmentation labels, optionally ignoring a specific class."""
    class_counts = np.zeros((num_classes,))
    for _, _, mask, _, _ in dataloader:

        for label in mask:
            unique, counts = np.unique(label, return_counts=True)
            unique = unique.astype(int)  # Convert unique class labels to integers

            for u, c in zip(unique, counts):
                if ignore_class is None or u != ignore_class:  # Check if the class is not the one to ignore
                    class_counts[u] += c

    total_counts = class_counts.sum()
    if ignore_class is not None:
        # Adjust total count by removing the ignored class count
        total_counts -= class_counts[ignore_class]

    # Calculate ratios
    class_ratios = class_counts / total_counts
    return class_ratios

# def plot_class_distributions(train_counts, val_counts, class_names=None):
#     """Plot class distributions for training and validation datasets."""
#     indices = np.arange(len(class_names))
#     width = 0.35

#     plt.figure(figsize=(12, 8))
#     plt.bar(indices - width/2, train_counts, width, label='Training')
#     plt.bar(indices + width/2, val_counts, width, label='Validation')

#     plt.ylabel('Counts')
#     plt.title('Class Distribution in Training and Validation Datasets')
#     plt.xticks(indices, class_names)
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

@hydra.main(config_path="conf", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    print(cfg)  # Optionally print the full config

    data_module = EstrieDataModule(
        input_format=cfg.model.input_format,
        classif_mode=cfg.model.classif_mode,
        train_mask_dir=cfg.model.train_mask_dir,
        val_mask_dir=cfg.model.val_mask_dir,
        test_mask_dir=cfg.model.test_mask_dir,
        batch_size=cfg.model.batch_size,
        dataset_size=cfg.model.dataset_size,
        train_transform=None,              
        test_mode=cfg.model.test_mode,
        sensors=cfg.model.sensors,
        opt_bands=cfg.model.opt_bands,
        lidar_bands=cfg.model.lidar_bands,
        indices_lst=cfg.model.indices_lst,
        num_workers=cfg.model.num_workers,
        pin_memory=cfg.model.pin_memory,
        debug_mode=False
    )

    # model = SemSegment(cfg=cfg.model)
    # trainer = Trainer(
    #     accelerator='gpu', 
    #     devices=1,
    #     callbacks=[
    #         ModelCheckpoint(save_top_k=5, monitor="val_loss", mode="min"),
    #         LearningRateMonitor(logging_interval='step')
    #     ],
    #     max_epochs=cfg.model.num_epochs
    # )

    train_loader, val_loader, _ = data_module.create_dataloaders()


    # Aggregate class counts
    num_classes = cfg.model.num_class  # Ensure this is defined in your config
    # train_class_counts = aggregate_segmentation_class_counts(train_loader, num_classes)
    # val_class_counts = aggregate_segmentation_class_counts(val_loader, num_classes)

    # train_class_counts = calculate_segmentation_class_ratios(train_loader, num_classes, ignore_class=7)
    # val_class_counts = calculate_segmentation_class_ratios(val_loader, num_classes, ignore_class=7)

    train_class_ratios, train_class_counts = calculate_segmentation_class_ratios_and_counts(train_loader, num_classes, ignore_class=7)
    val_class_ratios, val_class_counts = calculate_segmentation_class_ratios_and_counts(val_loader, num_classes, ignore_class=7)


    # Plotting class distributions
    class_names = [f'Class {i}' for i in range(num_classes)]  # Adjust as per your class names
    #plot_class_distributions(train_class_counts, val_class_counts, class_names=class_names)
    plot_class_distributions(train_class_ratios, val_class_ratios, train_class_counts, val_class_counts, class_names=class_names)

    # Continue with training and testing
    # trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)

if __name__ == "__main__":
    main()
