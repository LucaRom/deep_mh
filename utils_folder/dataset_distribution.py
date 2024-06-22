import numpy as np
import matplotlib.pyplot as plt
import rasterio
from collections import Counter
import os

# Create the specified lists
list1 = list(range(0, 520))  # Range from 1 to 571 inclusive
list2 = list(range(572, 2912))  # Range from 572 to 2911 inclusive

# Folder containing the label images
label_folder = "/mnt/d/00_Donnees/02_maitrise/01_trainings/portneuf_nord/processed_raw/mask_retile_test_2024"
no_data_value = -1  # No-data class value

# Function to calculate class distribution
def calculate_class_distribution(indices, folder, no_data_value):
    class_counts = Counter()
    for idx in indices:
        filename = f"mask_multiclass_3223.{idx}.tif"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            with rasterio.open(filepath) as src:
                image = src.read(1)
                unique, counts = np.unique(image, return_counts=True)
                class_counts.update(dict(zip(unique, counts)))
        else:
            print(f"File not found: {filepath}")
    if no_data_value in class_counts:
        del class_counts[no_data_value]
    return class_counts

# Calculate class distributions
test_distribution = calculate_class_distribution(list1, label_folder, no_data_value)
trainval_distribution = calculate_class_distribution(list2, label_folder, no_data_value)

# # Debug prints for distributions
# print(f"Test Distribution: {test_distribution}")
# print(f"Train/Val Distribution: {trainval_distribution}")

# Calculate total pixels
total_test_pixels = sum(test_distribution.values())
total_trainval_pixels = sum(trainval_distribution.values())

# # Debug prints for total pixels
# print(f"Total Test Pixels: {total_test_pixels}")
# print(f"Total Train/Val Pixels: {total_trainval_pixels}")

# Calculate ratio with error handling
if total_trainval_pixels != 0:
    ratio = total_test_pixels / total_trainval_pixels
    print(f"Ratio of test pixels to training/val pixels (excluding no-data class): {ratio:.2f}")
else:
    print("Error: Total number of training/val pixels is zero. Cannot calculate ratio.")

# Plotting the class distribution
classes = sorted(set(test_distribution.keys()).union(trainval_distribution.keys()))

test_values = [test_distribution.get(cls, 0) for cls in classes]
trainval_values = [trainval_distribution.get(cls, 0) for cls in classes]

bar_width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(np.arange(len(classes)), test_values, bar_width, label='Test')
bar2 = ax.bar(np.arange(len(classes)) + bar_width, trainval_values, bar_width, label='Training/Val')

ax.set_xlabel('Class')
ax.set_ylabel('Number of Pixels')
ax.set_title('Class Distribution Comparison')
ax.set_xticks(np.arange(len(classes)) + bar_width / 2)
ax.set_xticklabels(classes)
ax.legend()

# Add text labels with the number of pixels
for i, (test_val, trainval_val) in enumerate(zip(test_values, trainval_values)):
    ax.text(i, test_val, f'{test_val}', ha='center', va='bottom', fontsize=8)
    ax.text(i + bar_width, trainval_val, f'{trainval_val}', ha='center', va='bottom', fontsize=8)

plt.show()
