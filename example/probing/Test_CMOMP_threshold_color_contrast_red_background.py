# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:51:46 2023

@author: hjammoul
CMOMP tolerance red background. Color variation
"""
from Synthesize import generate_img_contrast_difference_effect
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import normalize_without_ref, complemantarity
from RGB_MorphoMath_tools import  extend_convergence_pts, Cmomp

##################################
O_inf_rgb = np.array((1.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((0.,1.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')

img_rgb, se_inf_rgb, se_sup_rgb = generate_img_contrast_difference_effect(height_img=150, img_background_color=(255, 0, 0),
                                                    target_color = (0, 230, 0),num_circles_per_line = 5, circle_radius = 8 )
size_of_SE = se_inf_rgb.shape[0]
##################################

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

# Plot the images
plt.figure(figsize=(12, 12))  # Increase the figure size to accommodate the additional plots
plt.subplot(3, 2, 1)
plt.imshow(img_rgb)
plt.title("RGB Image")
plt.subplot(3, 2, 2)
plt.imshow(se_inf_lab.astype(np.int16))
plt.title("lab SE inf ")
plt.subplot(3, 2, 3)
plt.imshow(se_sup_lab.astype(np.int16))
plt.title("lab SE sup ")
# Calculate the padding size for cropping
size_of_SE = se_inf_rgb.shape[0]  # Assuming SE_inf and SE_sup are square images
# Crop CMOMP map with padding
cropped_CMOMP = CMOMP[size_of_SE:-size_of_SE, size_of_SE:-size_of_SE]
# Plot the cropped CMOMP map with red crosses where CMOMP < d/20
plt.subplot(3, 2, 4)
plt.imshow(cropped_CMOMP)
plt.title("CMOMP map")
# Addred crosses to the CMOMP plot
red_cross_indices = np.where(red_cross_mask[size_of_SE:-size_of_SE, size_of_SE:-size_of_SE])
plt.scatter(red_cross_indices[1], red_cross_indices[0], color='red', marker='x', s=50)
plt.subplot(3, 2, 5)
plt.imshow(img_lab.astype(np.int16))
plt.title("LAB Image")
plt.tight_layout()
plt.show()

##Distance in lab between background of image and black: O_inf_lab and 0,0,0
origin2 = se_inf_lab[int(size_of_SE/2),int(size_of_SE/2)]
origin1 = se_sup_lab[int(size_of_SE/2),int(size_of_SE/2)]
d_backgrounds = np.linalg.norm(origin1-origin2)