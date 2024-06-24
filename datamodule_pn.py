import os
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from natsort import natsorted
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from utils.transformations import geo_transform
from utils.debug_functions import visualize_sample

def custom_collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class PortneufNordDataModule(LightningDataModule):
    def __init__(self, train_config_path, dataset_config_path='conf/datasets/portneuf_nord.yaml', extra_test_mask_dirs=None):
        super().__init__()
        self.dataset_config = OmegaConf.load(dataset_config_path)
        self.train_config = train_config_path # usually cfg from Hydra
        self.extra_test_mask_dirs = extra_test_mask_dirs
        self.train_class_weights = None
        self.load_paths()
        self.load_stats()
        self.load_indexes()
        self.load_config_params()

    def load_config_params(self):
        self.indices_lst = self.train_config.model.indices_lst
        self.batch_size = self.train_config.model.batch_size
        self.num_workers = self.train_config.model.num_workers
        self.pin_memory = self.train_config.model.pin_memory
        self.transforms = self.train_config.model.train_transforms

    def load_paths(self):
        # Resolving all paths using omegaconf's resolver
        self.paths = OmegaConf.to_container(self.dataset_config.data_sources, resolve=True)

        # Printing resolved paths for verification
        #print("Resolved paths:", self.paths)

    def load_stats(self):
        stats = OmegaConf.to_container(self.dataset_config.stats, resolve=True)

        sen2_means = [item for sublist in (stats['means']['sen2_ete'] + stats['means']['sen2_pri']) for item in sublist]
        sen1_means = [item for sublist in (stats['means']['sen1_ete'] + stats['means']['sen1_pri']) for item in sublist]
        lidar_means = [item for sublist in stats['means']['lidar'] for item in sublist]

        sen2_stdevs = [item for sublist in (stats['stdevs']['sen2_ete'] + stats['stdevs']['sen2_pri']) for item in sublist]
        sen1_stdevs = [item for sublist in (stats['stdevs']['sen1_ete'] + stats['stdevs']['sen1_pri']) for item in sublist]
        lidar_stdevs = [item for sublist in stats['stdevs']['lidar'] for item in sublist]

        self.stats = {
            'sen2_means': sen2_means,
            'sen1_means': sen1_means,
            'lidar_means': lidar_means,
            'sen2_stdevs': sen2_stdevs,
            'sen1_stdevs': sen1_stdevs,
            'lidar_stdevs': lidar_stdevs
        }

        # Printing resolved paths for verification
        #print("Loaded paths:", self.stats)
        #print("Loaded stats:", self.sen2_means, self.sen1_means, self.lidar_means, self.sen2_stdevs, self.sen1_stdevs, self.lidar_stdevs)

    def load_indexes(self):
        valid_idx = OmegaConf.to_container(self.dataset_config.indexes.idx_range.valid)
        trainval_idx_range = OmegaConf.to_container(self.dataset_config.indexes.idx_range.trainval)
        test1_idx_range = OmegaConf.to_container(self.dataset_config.indexes.idx_range.test1)
        test2_idx_range = OmegaConf.to_container(self.dataset_config.indexes.idx_range.test2)
        test3_idx_range = OmegaConf.to_container(self.dataset_config.indexes.idx_range.test3)

        all_test_idx = (list(range(test1_idx_range[0], test1_idx_range[1])) +
                        list(range(test2_idx_range[0], test2_idx_range[1])) +
                        list(range(test3_idx_range[0], test3_idx_range[1])))

        all_trainval_idx = list(range(trainval_idx_range[0], trainval_idx_range[1]))

        # Filter to include only those within the valid indexes and remove test idx from trainval idx
        valid_idx_set = set(valid_idx)
        test_idx_set = set(all_test_idx)

        self.trainval_idx = [idx for idx in all_trainval_idx if idx in valid_idx_set and idx not in test_idx_set]
        self.test_idx = [idx for idx in all_test_idx if idx in valid_idx_set]

        # print("Train/Val indexes within valid range:", self.trainval_idx)
        # print("Test indexes within valid range:", self.test_idx)

    def setup(self, stage=None):
        if self.train_config.debug_mode:
            self.trainval_idx = self.trainval_idx[:self.train_config.debug_size]
            self.test_idx = self.test_idx[:self.train_config.debug_size]

        main_mask_dir = os.path.join(self.paths['pre_path_tiled'], 'mask_multiclass_3223')

        # Use the initial dataset instance to calculate class distribution and weights
        temp_dataset = PortneufNordDataset(
            dataset_config=self.dataset_config,
            train_config=self.train_config,
            img_train_dir=self.paths['pre_path_tiled'],
            mask_dir=main_mask_dir,
            transforms=False,
            indices_lst=self.indices_lst,
            stats=self.stats,
            idx_list=self.trainval_idx
        )

        # Calculate class distribution and weights
        class_dist = temp_dataset.calculate_class_distribution(main_mask_dir, self.trainval_idx)
        train_idx, val_idx = temp_dataset.stratified_split(class_dist)
        
        train_class_distribution = {k: class_dist[k] for k in train_idx}
        aggregated_counts = temp_dataset.aggregate_class_counts(train_class_distribution)
        self.train_class_weights = temp_dataset.calculate_class_weights(aggregated_counts)

        train_sample_weights = temp_dataset.calculate_sample_weights(train_idx, class_dist, self.train_class_weights)
        sampler = WeightedRandomSampler(train_sample_weights, num_samples=int(len(train_sample_weights)*1.5), replacement=True)

        # Create the actual training and validation datasets
        train_dataset = PortneufNordDataset(
            dataset_config=self.dataset_config,
            train_config=self.train_config,
            img_train_dir=self.paths['pre_path_tiled'],
            mask_dir=main_mask_dir,
            transforms=self.transforms,
            indices_lst=self.indices_lst,
            stats=self.stats,
            idx_list=train_idx
        )

        val_dataset = PortneufNordDataset(
            dataset_config=self.dataset_config,
            train_config=self.train_config,
            img_train_dir=self.paths['pre_path_tiled'],
            mask_dir=main_mask_dir,
            transforms=False,
            indices_lst=self.indices_lst,
            stats=self.stats,
            idx_list=val_idx
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.sampler = sampler

        # Create the main test dataset
        self.main_test_dataset = PortneufNordDataset(
            dataset_config=self.dataset_config,
            train_config=self.train_config,
            img_train_dir=self.paths['pre_path_tiled'],
            mask_dir=main_mask_dir,
            transforms=False,
            indices_lst=self.indices_lst,
            stats=self.stats,
            idx_list=self.test_idx
        )

        # Create an additional test dataset if an additional mask directory is provided
        self.additional_test_dataset = None
        if self.extra_test_mask_dirs:
            self.additional_test_dataset = PortneufNordDataset(
                dataset_config=self.dataset_config,
                train_config=self.train_config,
                img_train_dir=self.paths['pre_path_tiled'],
                mask_dir=self.extra_test_mask_dirs,
                transforms=False,
                indices_lst=self.indices_lst,
                stats=self.stats,
                idx_list=self.test_idx
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.main_test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=False, collate_fn=custom_collate_fn)


class PortneufNordDataset(Dataset):
    def __init__(self, dataset_config, train_config, img_train_dir, mask_dir, transforms, indices_lst, stats, idx_list=None):
        self.dataset_config = dataset_config
        self.train_config = train_config
        self.image_dir = img_train_dir
        self.classif_mode = train_config.model.classif_mode
        self.mask_dir = mask_dir
        self.sensors = train_config.model.sensors
        self.transforms = transforms
        self.opt_bands = train_config.model.opt_bands
        self.lidar_bands = train_config.model.lidar_bands
        self.sar_bands = train_config.model.sar_bands
        self.indices_lst = indices_lst
        self.idx_list = idx_list
        self.stats = stats

        # Stats
        self.sen2_means = torch.tensor(stats['sen2_means'])
        self.sen1_means = torch.tensor(stats['sen1_means'])
        self.lidar_means = torch.tensor(stats['lidar_means'])
        self.sen2_stdevs = torch.tensor(stats['sen2_stdevs'])
        self.sen1_stdevs = torch.tensor(stats['sen1_stdevs'])
        self.lidar_stdevs = torch.tensor(stats['lidar_stdevs'])

        #self.images = self.filter_no_data_images()

        self.images = natsorted([x for x in os.listdir(os.path.join(img_train_dir, "sen2_ete")) if x.endswith('.tif')])
        if self.idx_list is not None:
            self.images = [self.images[i] for i in self.idx_list]

    def __len__(self):
        return len(self.images)

    def read_image(self, path, expand_dims=False, dtype=np.float32):
        img = np.array(tiff.imread(path), dtype=dtype)
        if expand_dims:
            img = np.expand_dims(img, axis=2)
        return img

    def get_mask_path(self, base_filename):
        if not self.mask_dir:
            raise ValueError("mask_dir must be specified for multiclass classification mode.")
        mask_base_filename = base_filename.replace("sen2_ete", os.path.basename(self.mask_dir))
        mask_path = os.path.join(self.mask_dir, mask_base_filename)
        return mask_path
    
    @staticmethod
    def pad_image(img, target_height, target_width, fill_value=-1):
        _, orig_height, orig_width = img.shape
        pad_height = target_height - orig_height
        pad_width = target_width - orig_width

        padding = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
        padded_img = F.pad(img, padding, mode='constant', value=fill_value)
        #print(f"Padding applied: {padding}")
        return padded_img

    def convert_multiclass_mask_to_binary(self, mask_path):
        mask_array = self.read_image(mask_path, expand_dims=False)
        binary_mask_array = np.where(mask_array != 7, 1, 0).astype(np.float32)
        return binary_mask_array

    def calculate_indices(self, image):
        band_names = {
            "B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, "B7": 6, "B8": 7,
            "B8A": 8, "B9": 9, "B11": 10, "B12": 11
        }

        indices = {}

        # NDVI (NIR - R) / (NIR + R) | (8 - 4) / (8 + 4)
        ndvi = (image[band_names['B8']] - image[band_names['B4']]) / (image[band_names['B8']] + image[band_names['B4']])
        indices['ndvi'] = np.nan_to_num(ndvi, nan=-1)

        # NDWI (GREEN - NIR) / (Green + NIR) | (3 - 8) / (3 + 8)
        ndwi = (image[band_names['B3']] - image[band_names['B8']]) / (image[band_names['B3']] + image[band_names['B8']])
        indices['ndwi'] = np.nan_to_num(ndwi, nan=-1)

        # NDMI (NIR - SWIR) / (NIR + SWIR) | (8 - 11) / (8 + 11)
        ndmi = (image[band_names['B8']] - image[band_names['B11']]) / (image[band_names['B8']] + image[band_names['B11']])
        indices['ndmi'] = np.nan_to_num(ndmi, nan=-1)

        # MBWI (omega * G) - R - N - S1 - S2 | (omega * 3) - (4 + 8 + 11 + 12)  # omega == 2
        mbwi = (2 * image[band_names['B3']]) - (image[band_names['B4']] + image[band_names['B8']] + image[band_names['B11']] + image[band_names['B12']])
        indices['mbwi'] = np.nan_to_num(mbwi, nan=-1)

        return indices

    def calculate_class_distribution(self, mask_dir, idx_list, no_data_value=-1):
        """
        Calculate the class distribution for each tile in the given directory.
        
        This method reads each tile (mask) file from the specified directory and computes
        the class distribution, ignoring tiles that contain only no data values. If a tile
        contains extreme float values indicating no data, they are replaced with the 
        specified no_data_value.

        Args:
        mask_dir (str): Path to the directory containing the mask files.
        idx_list (list): List of indices representing the tiles.
        no_data_value (int, optional): Value representing no data. Defaults to -1.

        Returns:
        dict: A dictionary with tile indices as keys and class distributions as values.
              Tiles containing only no data values are skipped and not included in the
              returned dictionary.
        """
        class_distribution = defaultdict(lambda: defaultdict(int))
        mask_name = os.path.basename(self.mask_dir)
        
        for idx in idx_list:
            mask_path = os.path.join(mask_dir, f"{mask_name}.{idx}.tif")
            mask = np.array(tiff.imread(mask_path))
                    
            # Replace any large/small float placeholders with the no_data_value
            mask[(mask <= -1e+38) | (mask >= 1e+38)] = no_data_value

            unique, counts = np.unique(mask, return_counts=True)
            class_counts = dict(zip(unique, counts))
            for class_id, count in class_counts.items():
                if class_id != -1:
                    class_distribution[idx][class_id] += count
        return class_distribution
    
    def aggregate_class_counts(self, class_distribution, background_class=7):
        total_distribution = defaultdict(int)
        for idx, distribution in class_distribution.items():
            if self.classif_mode == 'bin':
                for class_id, count in distribution.items():
                    if class_id == background_class:
                        total_distribution[0] += count
                    else:
                        total_distribution[1] += count
            else:
                for class_id, count in distribution.items():
                    total_distribution[class_id] += count
        return total_distribution

    def calculate_class_weights(self, class_counts):
        total_counts = sum(class_counts.values())
        class_weights = {class_id: total_counts / (len(class_counts) * count) for class_id, count in class_counts.items() if class_id != -1}
        sorted_class_weights = sorted(class_weights.items(), key=lambda item: item[0])
        sorted_weights = [weight for class_id, weight in sorted_class_weights]
        weights_tensor = torch.tensor(sorted_weights, dtype=torch.float)
        return weights_tensor    

    def stratified_split(self, class_distribution, val_ratio=0.1):
        total_distribution = Counter()
        for dist in class_distribution.values():
            total_distribution.update(dist)
        
        total_instances = sum(total_distribution.values())
        val_instances = int(total_instances * val_ratio)
        
        sorted_tiles = sorted(class_distribution.items(), key=lambda x: min(x[1].values()))
        
        val_set = set()
        current_val_instances = 0
        
        for idx, distribution in sorted_tiles:
            if current_val_instances >= val_instances:
                break
            val_set.add(idx)
            current_val_instances += sum(distribution.values())
        
        train_idx = [idx for idx in class_distribution if idx not in val_set]
        val_idx = list(val_set)
        
        return train_idx, val_idx
    
    def calculate_sample_weights(self, idx_list, class_dist, train_class_weights, background_class=7):
        weights = []
        for idx in idx_list:
            class_distribution = class_dist[idx]
            if len(class_distribution) == 1 and background_class in class_distribution:
                weight = 1
            else:
                weight = sum(train_class_weights[int(class_id)] * count 
                             for class_id, count in class_distribution.items() if class_id != -1 and class_id != background_class) / \
                         sum(count for class_id, count in class_distribution.items() if class_id != -1 and class_id != background_class)
            weights.append(weight)
        return weights
    
    def __getitem__(self, index):
        base_filename = self.images[index]
        img_opt, img_rad, img_lidar, mask = None, None, None, None

        sen2_ete_path = os.path.join(self.image_dir, 'sen2_ete', base_filename)
        sen2_pri_path = os.path.join(self.image_dir, 'sen2_pri', base_filename.replace("ete", "pri"))
        sen2_ete_img = self.read_image(sen2_ete_path)
        sen2_pri_img = self.read_image(sen2_pri_path)
        img_opt = np.dstack((sen2_ete_img, sen2_pri_img))

        sen1_ete_path = os.path.join(self.image_dir, 'sen1_ete', base_filename).replace("sen2_ete", "sen1_ete")
        sen1_pri_path = os.path.join(self.image_dir, 'sen1_pri', base_filename).replace("sen2_ete", "sen1_pri")
        sen1_ete_img = self.read_image(sen1_ete_path)
        sen1_pri_img = self.read_image(sen1_pri_path)
        img_rad = np.dstack((sen1_ete_img, sen1_pri_img))

        lidar_bands_all = ['mnt', 'mhc', 'slo', 'tpi', 'tri', 'twi']
        lidar_img_list = [self.read_image(os.path.join(self.image_dir, band, base_filename.replace("sen2_ete", band)), expand_dims=True) for band in lidar_bands_all]
        img_lidar = np.dstack(lidar_img_list).astype(np.float32)

        mask_path = self.get_mask_path(base_filename)
        if self.classif_mode == "bin":
            mask = self.convert_multiclass_mask_to_binary(mask_path)
        else:
            mask = self.read_image(mask_path)

        # Replace any large/small float placeholders with the no_data_value in mask
        no_data_value = -1
        mask[(mask <= -1e+38) | (mask >= 1e+38)] = no_data_value

        # Check if the entire mask is no-data
        if np.all(mask == no_data_value):
            return None

        img_opt = torch.from_numpy(img_opt).permute(2, 0, 1)
        img_rad = torch.from_numpy(img_rad).permute(2, 0, 1)
        img_lidar = torch.from_numpy(img_lidar).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        # # Calculate indices if needed
        # if len(self.indices_lst) > 0:
        #     indices = self.calculate_indices(img_opt)
        #     indices_tensors = [torch.tensor(indices[idx]).unsqueeze(0) for idx in self.indices_lst]

        # Calculate indices for summer and spring separately
        if len(self.indices_lst) > 0:
            img_summer = img_opt[:12, :, :]
            img_spring = img_opt[12:, :, :]

            summer_indices = self.calculate_indices(img_summer)
            spring_indices = self.calculate_indices(img_spring)

            summer_indices_tensors = [torch.tensor(summer_indices[idx]).unsqueeze(0) for idx in self.indices_lst]
            spring_indices_tensors = [torch.tensor(spring_indices[idx]).unsqueeze(0) for idx in self.indices_lst]

        # Standardization
        if self.stats is not None:
            img_opt = img_opt.sub_(self.sen2_means[:, None, None]).div_(self.sen2_stdevs[:, None, None])
            img_rad = img_rad.sub_(self.sen1_means[:, None, None]).div_(self.sen1_stdevs[:, None, None])
            img_lidar = img_lidar.sub_(self.lidar_means[:, None, None]).div_(self.lidar_stdevs[:, None, None])

        # Select bands for optical images after indices concatenation
        opt_bands_s2es2p = sorted(self.opt_bands + [x + 12 for x in self.opt_bands])
        img_opt = img_opt[opt_bands_s2es2p, :, :]

        # Select bands for SAR images
        sar_bands_s1es1p = sorted(self.sar_bands + [x + 3 for x in self.sar_bands])
        img_rad = img_rad[sar_bands_s1es1p, :, :]

        # Select bands for LiDAR images
        lidar_band_indices = {'mnt': 0, 'mhc': 1, 'slo': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
        lidar_band_indices_selected = [lidar_band_indices[band] for band in self.lidar_bands]
        img_lidar = img_lidar[lidar_band_indices_selected, :, :]

        # Determine the target size for padding
        target_height, target_width = 256, 256  # Adjust to the desired target size

        # Pad the images to the target size if they are smaller
        if img_opt.shape[1] < target_height or img_opt.shape[2] < target_width:
            #print("Padding required")
            img_opt = self.pad_image(img_opt, target_height, target_width)
            img_rad = self.pad_image(img_rad, target_height, target_width)
            img_lidar = self.pad_image(img_lidar, target_height, target_width)
            mask = self.pad_image(mask.unsqueeze(0), target_height, target_width).squeeze(0) # Because mask has dim=2
            
            summer_indices_tensors = [self.pad_image(indices_idx, target_height, target_width) for indices_idx in summer_indices_tensors]
            spring_indices_tensors = [self.pad_image(indices_idx, target_height, target_width) for indices_idx in spring_indices_tensors]

            #indices_tensors = [self.pad_image(indices_idx, target_height, target_width) for indices_idx in indices_tensors]

        # # Concatenate indices tensors after standardization
        # if len(self.indices_lst) > 0:
        #     img_opt = torch.cat((img_opt, *indices_tensors), dim=0)

        # Concatenate indices tensors after standardization
        if len(self.indices_lst) > 0:
            img_opt = torch.cat([img_opt, *summer_indices_tensors, *spring_indices_tensors], dim=0)

        if self.transforms:
            img_opt, img_rad, img_lidar, mask = geo_transform(img_opt, img_rad, img_lidar, mask)

        return img_opt, img_lidar, mask, img_rad, sen2_ete_path
    
if __name__ == "__main__":
    dataset_config_path = 'conf/datasets/portneuf_nord.yaml'
    train_config_path = 'conf/model/dummy_train.yaml'

    module = PortneufNordDataModule(
        dataset_config_path=dataset_config_path,
        train_config_path=train_config_path
    )

    module.setup()
    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()

    # for batch in train_loader:
    #     img_opt, img_rad, img_lidar, mask = batch
    #     print("Batch loaded")
    #     break

    # for batch in train_loader:
    #     img_opt, img_rad, img_lidar, mask = batch
    #     batch_size = img_opt.shape[0]
    #     for i in range(batch_size):
    #         visualize_sample(img_opt[i], img_rad[i], img_lidar[i], mask[i])
    #     break

    # Visualize the bands in the first batch
    for batch in train_loader:
        img_opt, img_rad, img_lidar, mask = batch
        batch_size = img_opt.shape[0]
        for i in range(batch_size):
            # print(f"Sample {i+1}:")
            # print(f"Optical Image Bands: {img_opt[i].shape[0]}")
            # print(f"Radar Image Bands: {img_rad[i].shape[0]}")
            # print(f"Lidar Image Bands: {img_lidar[i].shape[0]}")
            visualize_sample(img_opt[i], img_rad[i], img_lidar[i], mask[i])