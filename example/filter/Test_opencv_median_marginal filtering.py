# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:46:24 2023

@author: hjammoul
median arginal filter using opencv
"""

from Synthesize import generate_img_shape_size_difference_effect
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import normalize_without_ref, construct_conv_kernel, CRA_median_filtering
from RGB_MorphoMath_tools import  extend_convergence_pts
import cv2
######################################################################################"
def median_filter_rgb_image(img_rgb, kernel_size):
    """
    Applies median filtering on an RGB image.

    Parameters
    ----------
    img_rgb : numpy.ndarray
        The input RGB image.
    kernel_size : int
        The size of the median filter kernel. It should be an odd integer.

    Returns
    -------
    filtered_img_rgb : numpy.ndarray
        The median-filtered RGB image.
    """
    # Ensure that the kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Perform median filtering on each channel of the RGB image separately
    channels = cv2.split(img_rgb)
    filtered_channels = [cv2.medianBlur(channel, kernel_size) for channel in channels]

    # Merge the filtered channels back to form the filtered RGB image
    filtered_img_rgb = cv2.merge(filtered_channels)

    return filtered_img_rgb

##################################
img_rgb, se_inf_rgb, se_sup_rgb =  generate_img_shape_size_difference_effect(height_img=150, img_background_color=(0, 0,255),
                                                target_color=(255, 125, 50),
                                                num_shapes_per_line=5, initial_shape_size=6,target_radius = 8,noise_std=0.35)
size_of_SE = se_inf_rgb.shape[0]
#########################################"

# Apply median filtering with a kernel size of 3
filtered_img_rgb = median_filter_rgb_image(img_rgb, kernel_size=5)

# Plot the original and filtered images using matplotlib
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(filtered_img_rgb, cv2.COLOR_BGR2RGB))
plt.title('Filtered Image')
plt.axis('off')

plt.tight_layout()
plt.show()