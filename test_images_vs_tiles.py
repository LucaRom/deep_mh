import rasterio
from rasterio.enums import Resampling
import numpy as np

# def extract_and_save_tile(src_path, dest_path, tile_size=256):
#     # Open the source image
#     with rasterio.open(src_path) as src:
#         # Define the window to extract the tile
#         window = rasterio.windows.Window(0, 0, tile_size, tile_size)
        
#         # Read the data from the window
#         tile_data = src.read(window=window)

#         # Update the metadata for the destination file
#         out_meta = src.meta.copy()
#         out_meta.update({
#             "driver": "GTiff",
#             "height": tile_size,
#             "width": tile_size,
#             "transform": rasterio.windows.transform(window, src.transform)
#         })

#         # Write the tile to a new file
#         with rasterio.open(dest_path, 'w', **out_meta) as dest:
#             dest.write(tile_data)

# full_img_path = '/mnt/f/01_Code_nvme/Master_2024/results/full_stack/stack_estrie_3m_test_v04.tif'
# save_path_root = '/mnt/SN750/01_Code_nvme/Master_2024/results/inference/'

# # Specify the source image path and the destination path for the tile
# src_image_path = full_img_path
# dest_tile_path = save_path_root + "patate.tif" 

# # Call the function to extract and save the tile
# #extract_and_save_tile(src_image_path, dest_tile_path)

# test_idx_lst = np.load('results/estrie_test_idx_v16_v0p.npy')
# print(test_idx_lst)


import tifffile as tiff
import rasterio
import numpy as np

path_file_yo = "/mnt/d/00_Donnees/02_maitrise/01_trainings/estrie/256/sen2_ete/sen2_ete.13.tif"

# Read with tifffile
image_tiff = tiff.imread(path_file_yo)

# Read with rasterio
with rasterio.open(path_file_yo) as src:
    image_rasterio = src.read()  # Reading the first band
    print(src.shape)

# Compare
difference = np.isclose(image_tiff, image_rasterio, atol=1e-6)
print(f"Differences in arrays: {np.sum(~difference)} non-matching elements")