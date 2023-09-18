# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:09:58 2023

@author: hjammoul

Test median filtering on the synthesized rgb image
"""
from Synthesize import generate_img_shape_size_difference_effect
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import normalize_without_ref, construct_conv_kernel, CRA_median_filtering
from RGB_MorphoMath_tools import  extend_convergence_pts

##################################
O_inf_rgb = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((0.,0.,1.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')

img_rgb, se_inf_rgb, se_sup_rgb =  generate_img_shape_size_difference_effect(height_img=150, img_background_color=(0, 0, 0), target_color=(0, 0, 230),
                                              num_shapes_per_line=5, initial_shape_size=6,target_radius = 8,noise_std=0.35)
size_of_SE = se_inf_rgb.shape[0]
#########################################"

#Convert rgb img & se to lab space
img_lab = normalize_without_ref(img_rgb, size_of_SE) #normalizes by dividing by 255 and convert to lab, also adds padding of size_of_SE

se_inf_rgb_float = se_inf_rgb.astype(np.float32)
se_inf_normalized = se_inf_rgb_float / 255.
se_inf_lab = color.rgb2lab(se_inf_normalized, illuminant='D65')

se_sup_rgb_float = se_sup_rgb.astype(np.float32)
se_sup_normalized = se_sup_rgb_float / 255.
se_sup_lab = color.rgb2lab(se_sup_normalized, illuminant='D65')

O_sup_lab,O_inf_lab,d_conv_pts = extend_convergence_pts(O_sup_lab, O_inf_lab,se_sup_lab)

# Construct the convolution kernel with linear shape and size=3
kernel_linear = construct_conv_kernel(shape='linear', size=3)

# Apply median filtering with the linear kernel
median_filtered_lab_linear = CRA_median_filtering(img_lab, kernel_linear, O_sup_lab, O_inf_lab)

# Define convolution kernels with square and round shapes and size=3, 5, 10, 15
kernel_square = construct_conv_kernel(shape='square', size=3)
kernel_round_3 = construct_conv_kernel(shape='round', size=3)
kernel_round_5 = construct_conv_kernel(shape='round', size=5)
kernel_round_10 = construct_conv_kernel(shape='round', size=10)
kernel_round_15 = construct_conv_kernel(shape='round', size=15)

# Apply median filtering with different kernels
median_filtered_lab_square = CRA_median_filtering(img_lab, kernel_square, O_sup_lab, O_inf_lab)
median_filtered_lab_round_3 = CRA_median_filtering(img_lab, kernel_round_3, O_sup_lab, O_inf_lab)
median_filtered_lab_round_5 = CRA_median_filtering(img_lab, kernel_round_5, O_sup_lab, O_inf_lab)
median_filtered_lab_round_10 = CRA_median_filtering(img_lab, kernel_round_10, O_sup_lab, O_inf_lab)
median_filtered_lab_round_15 = CRA_median_filtering(img_lab, kernel_round_15, O_sup_lab, O_inf_lab)

# Plot each pair on a separate figure
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("RGB Image")

plt.subplot(2, 2, 2)
plt.imshow(img_lab)
plt.title("LAB Image")

plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(median_filtered_lab_linear)
plt.title("Median Filtered LAB (Linear Kernel)")

plt.subplot(2, 2, 2)
plt.imshow(kernel_linear, cmap='gray')
plt.title("Linear Convolution Kernel")


plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(median_filtered_lab_square)
plt.title("Median Filtered LAB (Square Kernel)")

plt.subplot(2, 2, 2)
plt.imshow(kernel_square, cmap='gray')
plt.title("Square Convolution Kernel")


plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(median_filtered_lab_round_3)
plt.title("Median Filtered LAB (Round Kernel, Size=3)")

plt.subplot(2, 2, 2)
plt.imshow(kernel_round_3, cmap='gray')
plt.title("Round Convolution Kernel (Size=3)")


plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(median_filtered_lab_round_5)
plt.title("Median Filtered LAB (Round Kernel, Size=5)")

plt.subplot(2, 2, 2)
plt.imshow(kernel_round_5, cmap='gray')
plt.title("Round Convolution Kernel (Size=5)")


plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(median_filtered_lab_round_10)
plt.title("Median Filtered LAB (Round Kernel, Size=10)")

plt.subplot(2, 2, 2)
plt.imshow(kernel_round_10, cmap='gray')
plt.title("Round Convolution Kernel (Size=10)")


plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.imshow(median_filtered_lab_round_15)
plt.title("Median Filtered LAB (Round Kernel, Size=15)")

plt.subplot(2, 2, 2)
plt.imshow(kernel_round_15, cmap='gray')
plt.title("Round Convolution Kernel (Size=15)")


plt.show()
