# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 15:28:29 2023

@author: hjammoul
CMOMP on black background
"""
from Synthesize import generate_color_img
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
img_rgb, se_inf_rgb, se_sup_rgb = generate_color_img(height_img=200, num_shapes=10, shape_min_size=10, shape_max_size=10, mode='CMOMP'
                                     ,img_background_color= (0,0,0), target_color = (20,20,250) )
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
red_cross_mask = CMOMP < 15

# Plot the images
# Plot the images
plt.figure(figsize=(12, 12))  # Increase the figure size to accommodate the additional plots

plt.subplot(3, 3, 1)
plt.imshow(img_rgb)
plt.title("RGB Image")

plt.subplot(3, 3, 2)
plt.imshow(se_inf_lab.astype(np.int16))
plt.title("lab g' ")

plt.subplot(3, 3, 3)
plt.imshow(se_sup_lab.astype(np.int16))
plt.title("lab g'' ")

# Calculate the padding size for cropping
size_of_SE = se_inf_rgb.shape[0]  # Assuming SE_inf and SE_sup are square images

# Plot the cropped CMOMP map with red crosses where CMOMP < d/20
plt.subplot(3, 3, 4)
plt.imshow(CMOMP)
plt.title("CMOMP map")
plt.colorbar()  # Add a colorbar to the CMOMP plot

plt.subplot(3, 3, 5)
plt.imshow(img_lab.astype(np.int16))
plt.title("LAB Image")

# Add plots of anti_dilated and eroded images
plt.subplot(3, 3, 6)
plt.imshow(anti_dilated.astype(np.int16))
plt.title("Anti-Dilated with g'' ")

plt.subplot(3, 3, 7)
plt.imshow(eroded.astype(np.int16))
plt.title("Eroded with g'")
plt.tight_layout()
plt.show()