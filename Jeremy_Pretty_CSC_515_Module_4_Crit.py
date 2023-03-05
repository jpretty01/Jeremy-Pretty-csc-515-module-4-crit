# Jeremy Pretty
# CSC 515
# Module 4 Crit
# March 5 2023
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

Mod_Img = os.path.join(os.path.dirname(__file__), 'Mod4CT1.jpg')

# Read the image
img = cv2.imread(Mod_Img, 0)

# Define kernel sizes
kernel_sizes = [3, 5, 7]

# Define sigma values for Gaussian filter
sigma_values = [1, 2]

# Define filter types
filters = {'Mean': lambda img, ksize: cv2.blur(img, (ksize,ksize)),
           'Median': lambda img, ksize: cv2.medianBlur(img, ksize),
           'Gaussian 1': lambda img, ksize: cv2.GaussianBlur(img, (ksize,ksize), sigmaX=sigma_values[0]),
           'Gaussian 2': lambda img, ksize: cv2.GaussianBlur(img, (ksize,ksize), sigmaX=sigma_values[1])}

# Apply filters for different kernel sizes and sigma values
results = {}
for ksize in kernel_sizes:
    for filter_name, filter_func in filters.items():
        if 'Gaussian' in filter_name:
            filtered_img = filter_func(img, ksize)
        else:
            filtered_img = filter_func(img, ksize)
        results[(ksize, filter_name)] = filtered_img

# Plot the results
fig, axs = plt.subplots(len(kernel_sizes), len(filters), figsize=(16,12))
for i, ksize in enumerate(kernel_sizes):
    for j, filter_name in enumerate(filters.keys()):
        filtered_img = results[(ksize, filter_name)]
        axs[i,j].imshow(filtered_img, cmap='gray')
        axs[i,j].set_title(f'{filter_name} {ksize}x{ksize}')
        axs[i,j].axis('off')
    axs[i,0].set_ylabel(f'{ksize}x{ksize}', fontsize=12)
axs[0,2].set_title(f'{filters["Gaussian 1"].__name__} {kernel_sizes[0]}x{kernel_sizes[0]} sigma={sigma_values[0]}')
axs[0,3].set_title(f'{filters["Gaussian 2"].__name__} {kernel_sizes[0]}x{kernel_sizes[0]} sigma={sigma_values[1]}')
plt.tight_layout()
plt.show()
