import os
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
import torchvision.transforms.functional as TF
from hydra.utils import to_absolute_path
from natsort import natsorted
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, WeightedRandomSampler

import utils_folder.img_paths as img_paths


class EstrieDataModule:
    def __init__(self, input_format, classif_mode, train_mask_dir, val_mask_dir, test_mask_dir, batch_size, dataset_size, train_transforms, test_mode=False, sensors='all', opt_bands=None, lidar_bands=None, sar_bands=None, indices_lst=None, num_workers=1, pin_memory=True, debug_mode=False, debug_size=100):
        self.input_format = input_format
        self.classif_mode = classif_mode
        self.train_mask_dir = train_mask_dir
        self.val_mask_dir = val_mask_dir
        self.test_mask_dir = test_mask_dir
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.train_transforms = train_transforms
        self.test_mode = test_mode
        self.sensors = sensors
        self.opt_bands = opt_bands
        self.lidar_bands = lidar_bands
        self.sar_bands = sar_bands
        self.indices_lst = indices_lst
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.debug_mode = debug_mode
        self.debug_size = debug_size
        self.paths_list, self.trainval_idx_lst, self.test_idx_lst = self.load_paths_and_indexes()
        self.mean_lst, self.stdev_lst = self.load_means_and_stdevs()

        # Calculate class weights to feed loss
        temp_mask_dir = os.path.join(self.paths_list[0], self.test_mask_dir)
        class_distribution = self.calculate_class_distribution(temp_mask_dir, self.trainval_idx_lst)
        aggregated_counts = self.aggregate_class_counts(class_distribution)
        self.class_weights = self.calculate_class_weights(aggregated_counts)

    def load_paths_and_indexes(self):
        # Logic to load paths and indexes based on input_format
        if self.input_format == 'estrie_over0p':
            paths_list = img_paths.estrie_256over0p_paths_lst
            trainval_idx_lst = np.load('results/estrie_trainval_idx_v16_v0p.npy')
            test_idx_lst = np.load('results/estrie_test_idx_v16_v0p.npy')
        elif self.input_format == 'estrie_over50p':
            paths_list = img_paths.estrie_256over50p_paths_lst
            trainval_idx_lst = np.load(to_absolute_path('results/estrie_trainval_idx_v15.npy'))
            test_idx_lst = np.load(to_absolute_path('results/estrie_test_idx_v15.npy'))
        return paths_list, trainval_idx_lst, test_idx_lst

    def load_means_and_stdevs(self):
        # Loading means and stdevs
        sen2_means = np.concatenate((np.load(to_absolute_path('stats/estrie/estrie_sen2_ete_means.npy')), np.load(to_absolute_path('stats/estrie/estrie_sen2_pri_means.npy'))))
        sen1_means = np.concatenate((np.load(to_absolute_path('stats/estrie/estrie_sen1_ete_means.npy')), np.load(to_absolute_path('stats/estrie/estrie_sen1_pri_means.npy'))))
        lidar_means = np.load(to_absolute_path('stats/estrie/estrie_lidar_means_v2.npy'))

        sen2_stdevs = np.concatenate((np.load(to_absolute_path('stats/estrie/estrie_sen2_ete_stds.npy')), np.load(to_absolute_path('stats/estrie/estrie_sen2_pri_stds.npy'))))
        sen1_stdevs = np.concatenate((np.load(to_absolute_path('stats/estrie/estrie_sen1_ete_stds.npy')), np.load(to_absolute_path('stats/estrie/estrie_sen1_pri_stds.npy'))))
        lidar_stdevs = np.load(to_absolute_path('stats/estrie/estrie_lidar_stds_v2.npy'))

        return [sen2_means, sen1_means, lidar_means], [sen2_stdevs, sen1_stdevs, lidar_stdevs]
    
    def read_image(self, path, expand_dims=False, dtype=np.float32):
        img = np.array(tiff.imread(path), dtype=dtype)
        if expand_dims:
            img = np.expand_dims(img, axis=2)
        return img     

    def convert_multiclass_mask_to_binary(self, mask_path):
        # Load the multiclass mask image
        mask_array = self.read_image(mask_path, expand_dims=False)
        
        # Convert all classes except 7 (background) to 1 (foreground) and class 7 to 0
        binary_mask_array = np.where(mask_array != 7, 1, 0).astype(np.float32)
               
        return binary_mask_array

    def calculate_class_distribution(self, mask_dir, idx_list):
        """
        Calculates the class distribution for each tile based on its mask.
        :param mask_dir: Directory containing mask files.
        :param idx_list: List of indices for which to calculate class distributions.
        :return: Dictionary mapping tile index to its class distribution.
        """
        class_distribution = defaultdict(lambda: defaultdict(int))
        for idx in idx_list:
            mask_path = os.path.join(mask_dir, f"{self.test_mask_dir}.{idx}.tif")
            mask = np.array(tiff.imread(mask_path))
            unique, counts = np.unique(mask, return_counts=True)
            class_counts = dict(zip(unique, counts))
            for class_id, count in class_counts.items():
                class_distribution[idx][class_id] += count
        return class_distribution

    def aggregate_class_counts(self, class_distribution, background_class=7):
        total_distribution = defaultdict(int)
        for idx, distribution in class_distribution.items():
            if self.classif_mode == 'bin':
                # For binary classification, switch all class IDs to 1 except for class #7 (background)
                for class_id, count in distribution.items():
                    if class_id == background_class:
                        total_distribution[0] += count  # Combine all non-background classes into class ID 1
                    else:
                        total_distribution[1] += count  # Keep background class as is
            else:
                #raise('outch')
                # Multiclass 
                for class_id, count in distribution.items():
                    total_distribution[class_id] += count
        return total_distribution

    def calculate_class_weights(self, class_counts):
        total_counts = sum(class_counts.values())
        class_weights = {class_id: total_counts / (len(class_counts) * count) for class_id, count in class_counts.items()}
        sorted_class_weights = sorted(class_weights.items(), key=lambda item: item[0])
        sorted_weights = [weight for class_id, weight in sorted_class_weights]
        weights_tensor = torch.tensor(sorted_weights, dtype=torch.float)
        return weights_tensor

    def stratified_split(self, class_distribution, val_ratio=0.1):
        # Aggregate class distributions across tiles
        total_distribution = Counter()
        for dist in class_distribution.values():
            total_distribution.update(dist)
        
        # Calculate total number of instances and the desired number of validation instances
        total_instances = sum(total_distribution.values())
        val_instances = int(total_instances * val_ratio)
        
        # Sort tiles by their contribution to the least represented class to prioritize diversity
        sorted_tiles = sorted(class_distribution.items(), key=lambda x: min(x[1].values()))
        
        val_set = set()
        current_val_instances = 0
        
        # Iteratively add tiles to the validation set
        for idx, distribution in sorted_tiles:
            if current_val_instances >= val_instances:
                break
            val_set.add(idx)
            current_val_instances += sum(distribution.values())
        
        # Split indices into training and validation sets
        train_idx = [idx for idx in class_distribution if idx not in val_set]
        val_idx = list(val_set)
        
        return train_idx, val_idx

    # def create_dataloaders(self):
    #     if self.test_mode:
    #         print('This is test mode only')
    #         test_dataset = EstrieDataset(
    #             img_train_dir=self.paths_list[0], 
    #             classif_mode=self.classif_mode, 
    #             mask_dir=self.test_mask_dir, 
    #             sensors=self.sensors,
    #             transforms=None, 
    #             mean_lst=self.mean_lst, 
    #             stdev_lst=self.stdev_lst, 
    #             opt_bands=self.opt_bands, 
    #             lidar_bands=self.lidar_bands,
    #             sar_bands=self.sar_bands, 
    #             indices_lst=self.indices_lst, 
    #             idx_list=self.test_idx_lst
    #         )
    #         test_loader = DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=False)
    #         return test_loader

    #     else:
    #         if self.debug_mode:
    #             # Reduce the size of the index lists for debug mode
    #             self.trainval_idx_lst = self.trainval_idx_lst[:self.debug_size]
    #             self.test_idx_lst = self.test_idx_lst[:self.debug_size]

    #         # Split the trainval index list for training and validation
    #         # shuffled_trainval = np.random.permutation(self.trainval_idx_lst)
    #         # val_size = round(len(self.trainval_idx_lst)*0.1) #10%

    #         # val_idx = shuffled_trainval[:val_size]  
    #         # train_idx = [x for x in shuffled_trainval if x not in val_idx]

    #         # Train val split with stratified by class
    #         mask_dir = os.path.join(self.paths_list[0], self.test_mask_dir)
    #         class_dist = self.calculate_class_distribution(mask_dir, self.trainval_idx_lst)
    #         train_idx, val_idx = self.stratified_split(class_dist, val_ratio=0.1)

    #         # Initialize datasets with the respective idx lists corrected
    #         train_dataset = EstrieDataset(
    #             img_train_dir=self.paths_list[0], 
    #             classif_mode=self.classif_mode, 
    #             mask_dir=self.train_mask_dir, 
    #             sensors=self.sensors,
    #             transforms=self.train_transforms,
    #             mean_lst=self.mean_lst, 
    #             stdev_lst=self.stdev_lst, 
    #             opt_bands=self.opt_bands, 
    #             lidar_bands=self.lidar_bands, 
    #             sar_bands=self.sar_bands, 
    #             indices_lst=self.indices_lst, 
    #             idx_list=train_idx 
    #         )
    #         val_dataset = EstrieDataset(
    #             img_train_dir=self.paths_list[0], 
    #             classif_mode=self.classif_mode, 
    #             mask_dir=self.val_mask_dir, 
    #             sensors=self.sensors,
    #             transforms=None,
    #             mean_lst=self.mean_lst, 
    #             stdev_lst=self.stdev_lst, 
    #             opt_bands=self.opt_bands, 
    #             lidar_bands=self.lidar_bands,
    #             sar_bands=self.sar_bands, 
    #             indices_lst=self.indices_lst, 
    #             idx_list=val_idx 
    #         )
    #         test_dataset = EstrieDataset(
    #             img_train_dir=self.paths_list[0], 
    #             classif_mode=self.classif_mode, 
    #             mask_dir=self.test_mask_dir, 
    #             sensors=self.sensors,
    #             transforms=None, 
    #             mean_lst=self.mean_lst, 
    #             stdev_lst=self.stdev_lst, 
    #             opt_bands=self.opt_bands, 
    #             lidar_bands=self.lidar_bands,
    #             sar_bands=self.sar_bands, 
    #             indices_lst=self.indices_lst, 
    #             idx_list=self.test_idx_lst
    #         )

    #         # Initialize DataLoaders without samplers
    #         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
    #         val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
    #         test_loader = DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=False)
  
    #         return train_loader, val_loader, test_loader
        
    def create_dataloaders(self):
            if self.test_mode:
                print('This is test mode only')
                test_dataset = EstrieDataset(
                    img_train_dir=self.paths_list[0], 
                    classif_mode=self.classif_mode, 
                    mask_dir=self.test_mask_dir, 
                    sensors=self.sensors,
                    transforms=None, 
                    mean_lst=self.mean_lst, 
                    stdev_lst=self.stdev_lst, 
                    opt_bands=self.opt_bands, 
                    lidar_bands=self.lidar_bands,
                    sar_bands=self.sar_bands, 
                    indices_lst=self.indices_lst, 
                    idx_list=self.test_idx_lst
                )
                test_loader = DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=False)
                return test_loader

            else:
                if self.debug_mode:
                    # Reduce the size of the index lists for debug mode
                    self.trainval_idx_lst = self.trainval_idx_lst[:self.debug_size]
                    self.test_idx_lst = self.test_idx_lst[:self.debug_size]

                # Split the trainval index list for training and validation
                # shuffled_trainval = np.random.permutation(self.trainval_idx_lst)
                # val_size = round(len(self.trainval_idx_lst)*0.1) #10%

                # val_idx = shuffled_trainval[:val_size]  
                # train_idx = [x for x in shuffled_trainval if x not in val_idx]

                # Train val split with stratified by class
                mask_dir = os.path.join(self.paths_list[0], self.test_mask_dir)
                class_dist = self.calculate_class_distribution(mask_dir, self.trainval_idx_lst)
                train_idx, val_idx = self.stratified_split(class_dist, val_ratio=0.1)

                # Calculate weights for each sample
                weights = []
                
                for idx in train_idx:
                    class_distribution = class_dist[idx]
                    if len(class_distribution) == 1 and 7.0 in class_distribution:
                        # If class 7 is the only class in the distribution
                        weight = 1
                    else:
                        # Compute a weighted sum or average based on the class distribution
                        #weight_1 = sum(self.class_weights[int(class_id)] * count for class_id, count in class_distribution.items()) / sum(class_distribution.values())
                        weight = sum(self.class_weights[int(class_id)] * count for class_id, count in class_distribution.items() if class_id != 7) /  sum(count for class_id, count in class_distribution.items() if class_id != 7)
                        
                    weights.append(weight)

                sampler = WeightedRandomSampler(weights, num_samples=int(len(weights)*1.5), replacement=True)

                # Initialize datasets with the respective idx lists corrected
                train_dataset = EstrieDataset(
                    img_train_dir=self.paths_list[0], 
                    classif_mode=self.classif_mode, 
                    mask_dir=self.train_mask_dir, 
                    sensors=self.sensors,
                    transforms=self.train_transforms,
                    mean_lst=self.mean_lst, 
                    stdev_lst=self.stdev_lst, 
                    opt_bands=self.opt_bands, 
                    lidar_bands=self.lidar_bands, 
                    sar_bands=self.sar_bands, 
                    indices_lst=self.indices_lst, 
                    idx_list=train_idx 
                )
                val_dataset = EstrieDataset(
                    img_train_dir=self.paths_list[0], 
                    classif_mode=self.classif_mode, 
                    mask_dir=self.val_mask_dir, 
                    sensors=self.sensors,
                    transforms=None,
                    mean_lst=self.mean_lst, 
                    stdev_lst=self.stdev_lst, 
                    opt_bands=self.opt_bands, 
                    lidar_bands=self.lidar_bands,
                    sar_bands=self.sar_bands, 
                    indices_lst=self.indices_lst, 
                    idx_list=val_idx 
                )

                test_dataset = EstrieDataset(
                    img_train_dir=self.paths_list[0], 
                    classif_mode=self.classif_mode, 
                    mask_dir=self.test_mask_dir, 
                    sensors=self.sensors,
                    transforms=None, 
                    mean_lst=self.mean_lst, 
                    stdev_lst=self.stdev_lst, 
                    opt_bands=self.opt_bands, 
                    lidar_bands=self.lidar_bands,
                    sar_bands=self.sar_bands, 
                    indices_lst=self.indices_lst, 
                    idx_list=self.test_idx_lst
                )

                # Initialize DataLoaders without samplers
                # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
                # val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, pin_memory=self.pin_memory)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)


                test_loader = DataLoader(test_dataset, batch_size=1, num_workers=self.num_workers, pin_memory=False)
    
                return train_loader, val_loader, test_loader
            

    def get_test_loaders(self, test_mask_dirs):
        """Generate a test DataLoader for each mask directory."""
        test_loaders = []
        for test_mask_dir in test_mask_dirs:
            test_dataset = EstrieDataset(
                img_train_dir=self.paths_list[0], 
                classif_mode=self.classif_mode, 
                mask_dir=test_mask_dir,  # Use the current mask directory from the list
                sensors=self.sensors, 
                mean_lst=self.mean_lst, 
                stdev_lst=self.stdev_lst, 
                opt_bands=self.opt_bands, 
                lidar_bands=self.lidar_bands, 
                indices_lst=self.indices_lst, 
                idx_list=self.test_idx_lst
            )
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
            test_loaders.append(test_loader)
        return test_loaders

class EstrieDataset(Dataset):
    def __init__(self, img_train_dir, classif_mode, mask_dir, sensors, transforms, mean_lst, stdev_lst, opt_bands, lidar_bands, sar_bands, indices_lst, idx_list=None):
        self.image_dir = img_train_dir
        self.classif_mode = classif_mode
        self.mask_dir = mask_dir
        self.sensors = sensors
        self.transforms = transforms
        self.mean_lst = mean_lst
        self.stdev_lst = stdev_lst
        self.opt_bands = opt_bands
        self.lidar_bands = lidar_bands
        self.sar_bands = sar_bands
        self.indices_lst = indices_lst
        self.idx_list = idx_list

        self.images = natsorted([x for x in os.listdir(os.path.join(img_train_dir, "sen2_ete")) if x.endswith('.tif')])
        if self.idx_list is not None:
            self.images = [self.images[i] for i in self.idx_list]

        #Loads means and stds for standardization according to selected bands
        if self.mean_lst is not None and self.stdev_lst is not None:
            # if self.opt_bands is not None:
            #     # Extract means and stds for selected bands (and add 12 to get the corresponding spring set bands)
            #     opt_bands_s1s2 = sorted(self.opt_bands + [x+12 for x in self.opt_bands])
            #     self.sen2_means = torch.tensor([self.mean_lst[0][i] for i in opt_bands_s1s2])
            #     self.sen2_stdevs = torch.tensor([self.stdev_lst[0][i] for i in opt_bands_s1s2])
            # else:
            self.sen2_means = torch.tensor(self.mean_lst[0])
            self.sen2_stdevs = torch.tensor(self.stdev_lst[0])

            if self.lidar_bands is not None:
                # Extract means and stds for selected LiDAR bands
                lidar_band_indices = {'mnt': 0, 'mhc': 1, 'pentes': 2, 'tpi': 3, 'tri': 4, 'twi': 5}
                self.lidar_means = torch.tensor([self.mean_lst[2][lidar_band_indices[band]] for band in self.lidar_bands])
                self.lidar_stdevs = torch.tensor([self.stdev_lst[2][lidar_band_indices[band]] for band in self.lidar_bands])
            else:
                self.lidar_means = torch.tensor(self.mean_lst[2])
                self.lidar_stdevs = torch.tensor(self.stdev_lst[2])

            # No bands selection for radar
            self.sen1_means = torch.tensor(self.mean_lst[1])
            self.sen1_stdevs= torch.tensor(self.stdev_lst[1])

        else:
            print('No means or stds available for standardization')

    def __len__(self):
        return len(self.images)

    def read_image(self, path, expand_dims=False, dtype=np.float32):
        img = np.array(tiff.imread(path), dtype=dtype)
        if expand_dims:
            img = np.expand_dims(img, axis=2)
        return img

    def get_mask_path(self, base_filename):
        if not self.mask_dir:  # Ensure that a mask directory is specified for multiclass mode
            raise ValueError("mask_dir must be specified for multiclass classification mode.")
        # Construct the mask path based on the provided mask_dir
        # Note: Adjust the replacement and directory structure as needed to fit your dataset
        mask_base_filename = base_filename.replace("sen2_ete", self.mask_dir)
        if self.mask_dir.endswith('mask_multiclass_3223_9c') or self.mask_dir.endswith('mask_multiclass_3223_buff') or self.mask_dir.endswith('mask_multiclass_9c'):
            mask_path = os.path.join(self.image_dir, self.mask_dir, mask_base_filename)
        else:
            # Fallback or default mask directory for multiclass
            mask_path = os.path.join(self.image_dir, mask_base_filename)
        return mask_path
    
    def convert_multiclass_mask_to_binary(self, mask_path):
        # Load the multiclass mask image
        mask_array = self.read_image(mask_path, expand_dims=False)
        
        # Convert all classes except 7 (background) to 1 (foreground) and class 7 to 0
        binary_mask_array = np.where(mask_array != 7, 1, 0).astype(np.float32)
               
        return binary_mask_array

    def calculate_indices(self, image, selected_bands):

        band_names = {
            "B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, "B7": 6, "B8": 7,
            "B8A": 8, "B9": 9, "B11": 10, "B12": 11
        }

        #selected_band_indices = {band: index for band, index in band_names.items() if index in selected_bands}

        selected_band_indices = {band: index for band, index in band_names.items() if index in selected_bands}
        opt_band_indices = {band: idx for idx, band in enumerate(selected_bands)}

        indices = {}

        # NDVI (NIR - R) / (NIR + R) | (8 - 4) / (8 + 4)
        ndvi = (image[opt_band_indices[selected_band_indices['B8']]] - image[opt_band_indices[selected_band_indices['B4']]]) / (image[opt_band_indices[selected_band_indices['B8']]] + image[opt_band_indices[selected_band_indices['B4']]])
        indices['ndvi'] = np.nan_to_num(ndvi, -1)

        # NDWI (GREEN - NIR) / (Green + NIR) | (3 - 8) / (3 + 8)
        ndwi = (image[opt_band_indices[selected_band_indices['B3']]] - image[opt_band_indices[selected_band_indices['B8']]]) / (image[opt_band_indices[selected_band_indices['B3']]] + image[opt_band_indices[selected_band_indices['B8']]])
        indices['ndwi'] = np.nan_to_num(ndwi, -1)

        # NDMI (NIR - SWIR) / (NIR + SWIR) | (8 - 11) / (8 + 11)
        ndmi = (image[opt_band_indices[selected_band_indices['B8']]] - image[opt_band_indices[selected_band_indices['B11']]]) / (image[opt_band_indices[selected_band_indices['B8']]] + image[opt_band_indices[selected_band_indices['B11']]])
        indices['ndmi'] = np.nan_to_num(ndmi, -1)

        # MBWI (omega * G) - R - N - S1 - S2 | (omega * 3) - (4 + 8 + 11 + 12)  # omega == 2
        mbwi = (2 * image[opt_band_indices[selected_band_indices['B3']]]) - (image[opt_band_indices[selected_band_indices['B4']]] + image[opt_band_indices[selected_band_indices['B8']]] + image[opt_band_indices[selected_band_indices['B11']]] + image[opt_band_indices[selected_band_indices['B12']]])
        indices['mbwi'] = np.nan_to_num(mbwi, -1)

        return indices
    

    def visualize_transformations(self, base_filename, img_idx, img_opt, img_rad, img_lidar, mask, title_prefix=""):
        """
        Visualizes transformations for optical, radar (SAR), LiDAR, and mask images side by side.
        img_opt, img_rad, img_lidar: PyTorch tensors of shape [C, H, W].
        mask: PyTorch tensor of shape [H, W].
        title_prefix: String to prefix the title for before/after differentiation.
        """
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        # Assuming the first three bands of optical images can be visualized as RGB
        if img_opt.shape[0] >= 3:
            img_opt_np = img_opt[:3].cpu().numpy()
            img_opt_np = np.transpose(img_opt_np, (1, 2, 0))  # Convert to HWC for plotting
            img_opt_np = (img_opt_np - img_opt_np.min()) / (img_opt_np.max() - img_opt_np.min())  # Normalize
            axs[0].imshow(img_opt_np)
        axs[0].set_title(f"{title_prefix}Optical")
        axs[0].axis('off')
        
        # For SAR, visualize the first channel (or any specific channel/intensity image)
        if img_rad.shape[0] >= 1:
            img_rad_np = img_rad[0].cpu().numpy()
            img_rad_np = (img_rad_np - img_rad_np.min()) / (img_rad_np.max() - img_rad_np.min())  # Normalize
            axs[1].imshow(img_rad_np, cmap='gray')
        axs[1].set_title(f"{title_prefix}SAR")
        axs[1].axis('off')
        
        # For LiDAR, visualize the first channel as an example
        if img_lidar.shape[0] >= 1:
            img_lidar_np = img_lidar[0].cpu().numpy()
            img_lidar_np = (img_lidar_np - img_lidar_np.min()) / (img_lidar_np.max() - img_lidar_np.min())  # Normalize
            axs[2].imshow(img_lidar_np, cmap='gray')
        axs[2].set_title(f"{title_prefix}LiDAR")
        axs[2].axis('off')
        
        # For mask, assuming single channel
        mask_np = mask.cpu().numpy()
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())  # Normalize
        axs[3].imshow(mask_np, cmap='gray')
        axs[3].set_title(f"{title_prefix}Mask")
        axs[3].axis('off')

        axs[0].set_title(f"{title_prefix}Optical (Image #{img_idx})")
        axs[1].set_title(f"{title_prefix}SAR (Image #{img_idx})")
        axs[2].set_title(f"{title_prefix}LiDAR (Image #{img_idx})")
        axs[3].set_title(f"{title_prefix}Mask (Image #{img_idx})")

        # Explicitly adjust spacing to make room for the overall figure title
        fig.subplots_adjust(top=0.8)  # You may need to adjust this value
        
        # Add the overall title
        fig.suptitle(f"Sentinel 2 path : {base_filename}", fontsize=16)

        plt.show()


    def __getitem__(self, index):
        base_filename = self.images[index]
        img_opt, img_rad, img_lidar, mask = None, None, None, None  # Initialize to handle different sensor configurations

        # Loading images
        # Sentinel-2 images
        sen2_ete_path = os.path.join(self.image_dir, 'sen2_ete', base_filename)
        sen2_print_path = os.path.join(self.image_dir, 'sen2_print', base_filename.replace("ete", "print"))
        sen2_ete_img = self.read_image(sen2_ete_path)
        sen2_print_img = self.read_image(sen2_print_path)

        # Select bands for Sen2
        # if self.opt_bands is not None:
        #     sen2_ete_img = sen2_ete_img[:,:,self.opt_bands]
        #     sen2_print_img = sen2_print_img[:,:,self.opt_bands]

        img_opt = np.dstack((sen2_ete_img, sen2_print_img))

        # Sentinel-1 images
        sen1_ete_path = os.path.join(self.image_dir, 'sen1_ete', base_filename).replace("sen2_ete", "sen1_ete")
        sen1_print_path = os.path.join(self.image_dir, 'sen1_print', base_filename).replace("sen2_ete", "sen1_print")
        sen1_ete_img = self.read_image(sen1_ete_path)
        sen1_print_img = self.read_image(sen1_print_path)

        # Load and preprocess SAR data
        if self.sar_bands is not None:
            sen1_ete_img = sen1_ete_img[:,:,self.sar_bands]  # Adjusted for SAR band selection
            sen1_print_img = sen1_print_img[:,:,self.sar_bands]

        img_rad = np.dstack((sen1_ete_img, sen1_print_img))

        # LiDAR images
        lidar_img_list = [self.read_image(os.path.join(self.image_dir, lidar_file, base_filename.replace("sen2_ete", lidar_file)), expand_dims=True) for lidar_file in self.lidar_bands]
        img_lidar = np.dstack(lidar_img_list)
        img_lidar = img_lidar.astype(np.float32)

        # Mask images
        mask_path = self.get_mask_path(base_filename)
        if self.classif_mode == "bin":
            mask = self.convert_multiclass_mask_to_binary(mask_path)
        else:
            mask = self.read_image(mask_path)

        # Cast to tensor for better permute
        img_opt = torch.from_numpy(img_opt).permute(2, 0, 1)
        img_rad = torch.from_numpy(img_rad).permute(2, 0, 1)
        img_lidar = torch.from_numpy(img_lidar).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        # Apply standardization (Need to be C x H X W)
        if self.mean_lst is not None and self.stdev_lst is not None:
            img_opt = img_opt.sub_(self.sen2_means[:, None, None]).div_(self.sen2_stdevs[:, None, None])
            img_rad = img_rad.sub_(self.sen1_means[:, None, None]).div_(self.sen1_stdevs[:, None, None])
            #img_rad = img_rad.sub_(self.sar_means[:, None, None]).div_(self.sar_stdevs[:, None, None])
            img_lidar = img_lidar.sub_(self.lidar_means[:, None, None]).div_(self.lidar_stdevs[:, None, None])

        # # Add indices after standardizationimg_
        # if len(self.indices_lst) > 0 and (sen2_ete_img.shape[2] != 12 or sen2_print_img.shape[2] != 12):
        #     raise ValueError("Number of bands must be 12 to calculate indices (not implemented for other number of bands)")
        if len(self.indices_lst) > 0:
            sen2_ete_img = torch.from_numpy(sen2_ete_img).permute(2, 0, 1)
            sen2_print_img = torch.from_numpy(sen2_print_img).permute(2, 0, 1)

            # Calculate indices here
            ete_indices = {key: value for key, value in self.calculate_indices(sen2_ete_img, self.opt_bands).items() if key in self.indices_lst}
            print_indices = {key: value for key, value in self.calculate_indices(sen2_print_img, self.opt_bands).items() if key in self.indices_lst}

            # Concatenate images and their indices
            ete_img_indices = torch.cat([torch.from_numpy(ete_indices[key]).unsqueeze(0) for key in ete_indices], dim=0)
            print_img_indices = torch.cat([torch.from_numpy(print_indices[key]).unsqueeze(0) for key in print_indices], dim=0)
            #img_opt = torch.cat((img_opt, ete_img_indices, print_img_indices), dim=0)

        # Select band after indices in case we remove a band that is used for indice
        opt_bands_s1s2 = sorted(self.opt_bands + [x+12 for x in self.opt_bands])
        img_opt = img_opt[opt_bands_s1s2,:,:]

        # Concatenate s2 selected bands with created indices
        img_opt = torch.cat((img_opt, ete_img_indices, print_img_indices), dim=0)

        # Calculate the percentage of non-class 7 elements
        # non_class_7_elements = (mask != 7).sum().item()
        # total_elements = mask.numel()
        # non_class_7_percentage = (non_class_7_elements / total_elements) * 100

        # if self.transform and non_class_7_percentage > 10:
        #         transformed = self.transform({'img_opt': img_opt, 'img_rad': img_rad, 'img_lidar': img_lidar, 'mask': mask})
        #         img_opt, img_rad, img_lidar, mask = transformed['img_opt'], transformed['img_rad'], transformed['img_lidar'], transformed['mask']

        # Apply transform for training if true
        if self.transforms:
            def random_masked_crop_individual(img_opt, img_rad, img_lidar, mask, min_output_size=(150,150), max_output_size=(225,225), fill_value=-1):
                def mask_and_crop(img, fill_value, top, left, output_height, output_width, orig_height, orig_width):
                    masked_img = torch.full_like(img, fill_value)

                    masked_img[:, top:top + output_height, left:left + output_width] = img[:, top:top + output_height, left:left + output_width]
                    return masked_img

                # Crop size
                _, orig_height, orig_width = img_opt.shape
                output_height = random.randint(min_output_size[0], min(max_output_size[0], orig_height))
                output_width = random.randint(min_output_size[1], min(max_output_size[1], orig_width))
                
                # Crop corner
                top = random.randint(0, orig_height - output_height)
                left = random.randint(0, orig_width - output_width)

                cropped_images = []
                for img in [img_opt, img_rad, img_lidar, mask]:
                    cropped_image = mask_and_crop(img, fill_value, top, left, output_height, output_width, orig_height, orig_width)
                    cropped_images.append(cropped_image)

                return tuple(cropped_images)

            def geo_transform(img_opt, img_rad, img_lidar, mask):
                """
                Apply the same random geometric transformations to all inputs (images and mask).
                img_opt, img_rad, img_lidar: Tensors of shape [C, H, W].
                mask: Tensor of shape [H, W].
                """
                # Seed for ensuring the same transformation across all inputs
                seed = torch.Generator().manual_seed(int(random.random() * 2**32))

                # Random horizontal flipping
                if random.random() > 0.5:
                    img_opt = TF.hflip(img_opt)
                    img_rad = TF.hflip(img_rad)
                    img_lidar = TF.hflip(img_lidar)
                    mask = TF.hflip(mask)

                # Random vertical flipping
                if random.random() > 0.5:
                    img_opt = TF.vflip(img_opt)
                    img_rad = TF.vflip(img_rad)
                    img_lidar = TF.vflip(img_lidar)
                    mask = TF.vflip(mask)

                # Random rotation
                # TODO See how to implement rotation because of the created no data value
                mask = mask.unsqueeze(0)
                if random.random() > 0.5:
                    angle = random.randint(-30, 30)
                    img_opt = TF.rotate(img_opt, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False, fill=-1)
                    img_rad = TF.rotate(img_rad, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False, fill=-1)
                    img_lidar = TF.rotate(img_lidar, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False, fill=-1)
                    mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, expand=False, fill=-1)

                # Assuming img_opt, img_rad, img_lidar, and mask are your input tensors
                if random.random() > 0.5:
                    img_opt, img_rad, img_lidar, mask = random_masked_crop_individual(img_opt, img_rad, img_lidar, mask)

                mask = mask.squeeze(0)

                return img_opt, img_rad, img_lidar, mask

            # Example usage in your __getitem__ method:

            # Before transformations # Debug
            #self.visualize_transformations(base_filename, index, img_opt, img_rad, img_lidar, mask, title_prefix="Before: ")

            # Apply the transformations
            # print("Before transforms")
            # print(img_opt.shape, img_rad.shape, img_lidar.shape, mask.shape)
            img_opt, img_rad, img_lidar, mask = geo_transform(img_opt, img_rad, img_lidar, mask)
            # print("After transforms")
            # print(img_opt.shape, img_rad.shape, img_lidar.shape, mask.shape)

            # After transformations # Debug
            #self.visualize_transformations(base_filename, index, img_opt, img_rad, img_lidar, mask, title_prefix="After: ")

            # DEBUG FOR INTERMEDIATE OUTPUT BEFORE MODEL
            # import rasterio 
            # from rasterio.transform import from_origin

            # metadata = {
            #     'driver': 'GTiff',
            #     'dtype': 'float32',
            #     'nodata': None,
            #     'width': 256,
            #     'height': 256,
            #     'count': 22,
            #     'crs': rasterio.crs.CRS.from_epsg(32198),  # Updated EPSG code
            #     'transform': from_origin(-180, 90, 0.1, 0.1)
            #     }

            # img_opt = img_opt.detach().cpu().numpy()

            # with rasterio.open('results/debug/img_opt.tif', 'w', **metadata) as dst:
            #     for band in range(img_opt.shape[0]):
            #         dst.write(img_opt[band], band+1)  # Write each band

        if self.sensors in ['all', 's2s1', 's2lr', 's1lr']:
            return {
                'all': (img_opt, img_lidar, mask, img_rad, sen2_ete_path),
                's2s1': (img_opt, img_rad, mask, sen2_ete_path),
                's2lr': (img_opt, img_lidar, mask, sen2_ete_path),
                's1lr': (img_rad, img_lidar, mask, sen2_ete_path),
            }.get(self.sensors)
        else:
            print('Error: sensors argument not valid. Please choose between "all", "s2s1" or "s2lr".')