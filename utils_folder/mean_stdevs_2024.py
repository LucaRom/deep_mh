from omegaconf import OmegaConf
import rasterio
import numpy as np
import os

# Load the configuration
def load_config(path_to_yaml):
    conf = OmegaConf.load(path_to_yaml)
    return OmegaConf.to_container(conf, resolve=True)

# Function to calculate mean and standard deviation of an image
def calculate_statistics(image_path):
    with rasterio.open(image_path) as src:
        means = []
        stdevs = []
        for band in range(1, src.count + 1):
            data = src.read(band)
            # Fetch no data value from the file metadata, if it exists
            nodata_value = src.nodatavals[band - 1]  # src.nodatavals is a tuple, one value for each band
            if nodata_value is not None:
                valid_data = data[data != nodata_value]
            else:
                valid_data = data  # Use all data if no nodata value is specified
            if valid_data.size > 0:  # Check if there is any valid data left
                means.append(np.mean(valid_data))
                stdevs.append(np.std(valid_data))
            else:
                means.append(None)  # or use np.nan if appropriate
                stdevs.append(None)
    return means, stdevs

# Main function to process the images and handle specific no data values
def process_images(config):
    # Results dictionary
    results = {
        'means': {},
        'stdevs': {}
    }

    # Define the groups
    groups = {
        'sen2': ['sen2_ete', 'sen2_pri'],
        'sen1': ['sen1_ete', 'sen1_pri'],
        'lidar': ['mnt', 'mhc', 'slo', 'tpi', 'tri', 'twi']
    }

    # Calculate statistics for each group
    for group_key, images in groups.items():
        group_means = []
        group_stdevs = []
        for image_key in images:
            image_path = config['data_sources']['raw_clipped'][image_key]
            means, stdevs = calculate_statistics(image_path)
            group_means.append(means)
            group_stdevs.append(stdevs)

        results['means'][f'{group_key}_means'] = group_means
        results['stdevs'][f'{group_key}_stdevs'] = group_stdevs

    return results

# Example usage
config_file = 'conf/datasets/portneuf_nord.yaml'
config = load_config(config_file)
results = process_images(config)

# Printing results with explicit details
for key, value in results['means'].items():
    print(f"{key}: {value}")

for key, value in results['stdevs'].items():
    print(f"{key}: {value}")
