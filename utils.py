import torch
import torchvision
import wandb
from dataset import KenaukDataset
from torch.utils.data import DataLoader

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    #train_transform,
    #val_transform,
    num_workers=1,
    pin_memory=True,
):
    train_ds = KenaukDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        #transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = KenaukDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        #transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = KenaukDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader