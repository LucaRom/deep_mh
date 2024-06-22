import rasterio
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calculate_pixels_info(file_paths):
    for name, path in file_paths.items():
        with rasterio.open(path) as dataset:
            data = dataset.read(1)  # Read the first band into a 2D array
            
            # Calculate pixel counts by class
            unique, counts = np.unique(data, return_counts=True)
            total_pixels = data.size
            pixel_counts = dict(zip(unique, counts))
            
            # Calculate ratios
            ratios = {k: v / total_pixels for k, v in pixel_counts.items()}
            
            # Output results
            print(f"File: {name}")
            print(f"Total Pixels: {total_pixels}")
            print("Pixels by Class:", pixel_counts)
            print("Ratios:", ratios)
            print("-" * 40)

def calculate_pixels_info_exclude_value(file_paths, exclude_value):
    for name, path in file_paths.items():
        with rasterio.open(path) as dataset:
            data = dataset.read(1)  # Read the first band into a 2D array

            # Mask the data to ignore the specified value
            masked_data = data[data != exclude_value]
            
            # Calculate pixel counts by class for the masked data
            unique, counts = np.unique(masked_data, return_counts=True)
            total_pixels = masked_data.size
            pixel_counts = dict(zip(unique, counts))
            
            # Calculate ratios excluding the specified value
            ratios = {k: v / total_pixels for k, v in pixel_counts.items()}
            
            # Output results
            print(f"File: {name}")
            print(f"Total Pixels Excluding Nodata or -1'{exclude_value}': {total_pixels}")
            print("Pixels by Class (Excluding Nodata or -1):", pixel_counts)
            print("Ratios (Excluding Nodata or -1):", ratios)
            print("-" * 40)

def replace_nodata_value(input_raster_path, output_raster_path, new_nodata_value):
    with rasterio.open(input_raster_path) as src:
        # Read the dataset's metadata
        meta = src.meta
        
        # Update the NoData value in the metadata to the new value
        meta.update(nodata=new_nodata_value)
        
        # Read the first band of raster data
        data = src.read(1)
        
        # Identify the NoData values using isclose for floating-point comparison
        nodata_mask = np.isclose(data, src.nodata, atol=1e-5)
        
        # Replace the NoData values with the new NoData value
        data[nodata_mask] = new_nodata_value
        
        # Write the modified array to a new file using the updated metadata
        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(data, 1)

if __name__ == "__main__":

    # Define the paths and values
    input_raster_path = "/mnt/d/00_Donnees/02_maitrise/01_trainings/portneuf_zone1/clipped_mask_portneuf_nord.tif"
    output_raster_path = "/mnt/d/00_Donnees/02_maitrise/01_trainings/portneuf_zone1/clipped_mask_portneuf_nord_v2.tif"
    old_value = -3.4028235e+38
    new_value = -1  # Replace with the value you want

    # # Run the function
    # replace_value_in_raster(input_raster_path, output_raster_path, old_value, new_value)
    # replace_nodata_value(input_raster_path, output_raster_path, new_value)

    # Load the configuration file
    config_path = '/mnt/f/01_Code_nvme/Master_2024/conf/paths/masks.yaml'  # Update this path if your file is located elsewhere
    config = load_config(config_path)

    # Run the function with the loaded file paths
    #calculate_pixels_info(config['full_mask'])
    calculate_pixels_info_exclude_value(config['full_mask'], -1)