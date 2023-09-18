# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:48:46 2023

@author: hjammoul
Test CMOMP threshold choice by variation of shapes and sizes - img Blue backround
"""
from Synthesize import generate_img_shape_size_difference_effect
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import normalize_without_ref, complemantarity
from RGB_MorphoMath_tools import  extend_convergence_pts, Cmomp

##################################
O_inf_rgb = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((0.,0.,1.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
target_color_lab = color.rgb2lab(np.array((0.,0.,float(230./255.)), dtype=np.float32), illuminant='D65')
img_rgb, se_inf_rgb, se_sup_rgb =  generate_img_shape_size_difference_effect(height_img=150, img_background_color=(0, 0, 0), target_color=(0, 0, 230),
                                              num_shapes_per_line=5, initial_shape_size=6,target_radius = 10)
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
CMOMP, anti_dilated,eroded = Cmomp(img_lab, se_inf_lab, se_sup_lab,  O_inf_lab,O_sup_lab)

d_over_20 = int(d_conv_pts) / 39
red_cross_mask = CMOMP < d_over_20

cropped_CMOMP = CMOMP[size_of_SE:-size_of_SE, size_of_SE:-size_of_SE]
# Add red crosses to the CMOMP plot
red_cross_indices = np.where(red_cross_mask[size_of_SE:-size_of_SE, size_of_SE:-size_of_SE])
# Calculate the delta E distance (l2 norm) of the image in LAB to the coordinates of target_color_lab
delta_E = np.linalg.norm(img_lab - target_color_lab, axis=-1)

# Plot the images
plt.figure(figsize=(12, 12))  # Increase the figure size to accommodate the additional plots
# Plot the RGB image
plt.subplot(3, 2, 1)
plt.imshow(img_rgb)
plt.title("RGB Image")
plt.axis('on')
# Plot the LAB image
plt.subplot(3, 2, 2)
plt.imshow(img_lab.astype(np.int16))
plt.title("LAB Image")
plt.axis('on')
# Plot SE_inf
plt.subplot(3, 2, 3)
plt.imshow(se_inf_lab.astype(np.int16))
plt.title("LAB SE Inf")
plt.axis('on')
# Plot SE_sup
plt.subplot(3, 2, 4)
plt.imshow(se_sup_lab.astype(np.int16))
plt.title("LAB SE Sup")
plt.axis('on')
# Plot the delta E distance
plt.subplot(3, 2, 5)
plt.imshow(delta_E, cmap='jet')
plt.title("Delta E Distance to target color coordinate")
plt.colorbar()
plt.axis('on')
# Plot CMOMP
plt.subplot(3, 2, 6)
plt.imshow(cropped_CMOMP, cmap='jet')
plt.title("CMOMP map")
plt.colorbar()
plt.scatter(red_cross_indices[1], red_cross_indices[0], color='red', marker='x', s=50)
plt.axis('on')

plt.tight_layout()
plt.show()
