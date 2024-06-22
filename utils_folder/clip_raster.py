"""
Script to Clip Multiple Raster Images Using a Shapefile

This script loads image paths from a specified YAML file, where each path may contain a placeholder for a base directory.
It then clips each image using a specified vector shapefile using rasterio and pyshp and saves the clipped images with a modified filename.
"""

from omegaconf import OmegaConf
import rasterio
from rasterio.mask import mask
import fiona
import os

# Function to load the configuration
def load_config(path_to_yaml):
    conf = OmegaConf.load(path_to_yaml)
    return OmegaConf.to_container(conf, resolve=True)  # Resolve interpolations

# Function to clip a raster with a shapefile
def clip_raster(input_raster, shapefile_path, output_path):
    # Extract the base filename without extension
    base_filename = os.path.basename(input_raster)
    output_filename = f"{os.path.splitext(base_filename)[0]}_clipped.tif"
    output_raster = os.path.join(output_path, output_filename)

    with rasterio.open(input_raster) as src:
        with fiona.open(shapefile_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            with rasterio.open(output_raster, "w", **out_meta) as dest:
                dest.write(out_image)
            print(f"Clipped and saved: {output_raster}")

# Main processing function
def process_images(config_path, shapefile_path, output_path):
    config = load_config(config_path)
    for key, path in config['data_sources']['raw'].items():
        clip_raster(path, shapefile_path, output_path)

# Example usage
config_file = 'conf/datasets/portneuf_nord.yaml'
shapefile = '/mnt/d/00_Donnees/02_maitrise/shapefiles/portneuf_nord.shp'
output_folder = '/mnt/d/00_Donnees/02_maitrise/01_trainings/portneuf_nord/processed_raw_clipped'

process_images(config_file, shapefile, output_folder)
