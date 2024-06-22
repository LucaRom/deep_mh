import os
import rasterio
from shapely.geometry import box
from collections import defaultdict

# This script doesnt work properly, needs more tuning

# # Directory containing your geospatial TIFF files
# directory_path = "/mnt/f/00_Donnees_SSD/256_over50p/sen2_ete"

# # Function to get the geographic bounds of a tile
# def get_tile_bounds(filepath):
#     with rasterio.open(filepath) as dataset:
#         bounds = dataset.bounds
#     return box(bounds.left, bounds.bottom, bounds.right, bounds.top)

# # Load the bounds for each tile
# tiles_bounds = {}
# for filename in os.listdir(directory_path):
#     if filename.endswith(".tif"):
#         filepath = os.path.join(directory_path, filename)
#         tiles_bounds[filename] = get_tile_bounds(filepath)

# # Determine overlapping tiles
# overlaps = defaultdict(list)
# for filename1, bounds1 in tiles_bounds.items():
#     for filename2, bounds2 in tiles_bounds.items():
#         if filename1 != filename2 and bounds1.intersects(bounds2):
#             overlaps[filename1].append(filename2)

# # Now, overlaps is a dictionary where each key is a filename of a tile,
# # and the value is a list of filenames of tiles it overlaps with.

# # Example: Print overlapping tiles
# for filename, overlapping_tiles in overlaps.items():
#     print(f"{filename} overlaps with {', '.join(overlapping_tiles)}")
