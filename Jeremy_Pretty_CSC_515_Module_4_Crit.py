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
sigmas = [1, 2]

# Define filter types
filters = {'Mean': cv2.blur, 'Median': cv2.medianBlur, 'Gaussian (sigma=1)': lambda img: cv2.GaussianBlur(img, (0,0), sigmaX=1),
           'Gaussian (sigma=2)': lambda img: cv2.GaussianBlur(img, (0,0), sigmaX=2)}

# Apply filters for different kernel sizes and sigma values
results = {}
for ksize in kernel_sizes:
    for filter_name, filter_func in filters.items():
        if 'Gaussian' in filter_name:
            for sigma in sigmas:
                filtered_img = filter_func(img, (ksize, ksize), sigmaX=sigma)
                results[(ksize, filter_name + f' (sigma={sigma})')] = filtered_img
        else:
            filtered_img = filter_func(img, (ksize, ksize))
            results[(ksize, filter_name)] = filtered_img

# Plot the results
fig, axs = plt.subplots(len(kernel_sizes), len(filters), figsize=(12,8))
for i, ksize in enumerate(kernel_sizes):
    for j, filter_name in enumerate(filters.keys()):
        if 'Gaussian' in filter_name:
            for sigma in sigmas:
                filtered_img = results[(ksize, filter_name + f' (sigma={sigma})')]
                axs[i,j].imshow(filtered_img, cmap='gray')
                axs[i,j].set_title(f'{filter_name} {ksize}x{ksize} (sigma={sigma})')
        else:
            filtered_img = results[(ksize, filter_name)]
            axs[i,j].imshow(filtered_img, cmap='gray')
            axs[i,j].set_title(f'{filter_name} {ksize}x{ksize}')
        axs[i,j].axis('off')
    axs[i,0].set_ylabel(f'{ksize}x{ksize}', fontsize=12)
plt.tight_layout()
plt.show()
