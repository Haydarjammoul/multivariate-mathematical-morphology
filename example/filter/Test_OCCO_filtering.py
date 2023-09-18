# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:36:16 2023

@author: hjammoul
Experiment opening and closing on ganglia slice
divide image by slice max
convegence points choice: (black,yellow) or (green,yellow)
SE: disk shape 11px (background black, foreground: yellow)
"""
from Synthesize import generate_img_shape_size_difference_effect,get_slice
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import CRA_median_filtering
from RGB_MorphoMath_tools import  extend_convergence_pts,Construct_round_SE_flat,CRA_erosion_rgb,CRA_dilation_rgb,opening,closing
########################
path = "C:/Users\hjammoul\Desktop\stage Hjammoul\stat_code\MorphoMath"
image_file = r"C:\Users\hjammoul\Desktop\stage Hjammoul\New Aquisition ganglia\ganglia1.tif"
##################################
O_inf_rgb = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((1.,1.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')

#SE parameteres:
size_of_SE = 13
shape_size = 11
Value_rgb = np.array((1.,1.,0.), dtype=np.float32) 
Value_lab = color.rgb2lab(Value_rgb, illuminant='D65') 
#Construct SE
SE_lab = Construct_round_SE_flat(Value_lab, size_of_SE, shape_size)
#extend conv pts
O_sup_lab,O_inf_lab,d_conv_pts = extend_convergence_pts(O_sup_lab, O_inf_lab,SE_lab)
##
z=136
#get slice: channel,y,x
slice_rgb = get_slice(image_file, z)
#transform to channel last
#channel 1 red and two is green add a third channel blue full of zeros
slice_rgb = np.transpose(slice_rgb, (1,2,0))
slice_rgb = np.concatenate((slice_rgb, np.zeros_like(slice_rgb[:, :, :1])), axis=2)
# Divide the slice by its max value (float)
max_value = slice_rgb.max()
slice_rgb_float = slice_rgb.astype(np.float32) / max_value
# Transform to LAB space using color.rgb2lab()
slice_lab = color.rgb2lab(slice_rgb_float, illuminant='D65')

#eroded and filated
eroded_value = CRA_erosion_rgb(slice_lab, SE_lab,  O_inf_lab,O_sup_lab)
dilated_value =  CRA_dilation_rgb(slice_lab, SE_lab,  O_inf_lab,O_sup_lab)

# Perform opening and closing operations
opened_value = opening(slice_lab, SE_lab, O_inf_lab, O_sup_lab)
closed_value = closing(slice_lab, SE_lab, O_inf_lab, O_sup_lab)

# Open_close filter
open_close_value = closing(opened_value, SE_lab, O_inf_lab, O_sup_lab)

# Close_open filter
close_open_value = opening(closed_value, SE_lab, O_inf_lab, O_sup_lab)

# open_close_open filter
open_close_open_value = opening(open_close_value, SE_lab, O_inf_lab, O_sup_lab)

# close_open_close filter
close_open_close_value = closing(close_open_value, SE_lab, O_inf_lab, O_sup_lab)

median_filter_value = CRA_median_filtering(slice_lab,SE_lab, O_sup_lab, O_inf_lab)


# Plot the original RGB slice, its LAB space slice, and the SE_lab using matplotlib
# Define a list of filtered images
filtered_images = [
    opened_value,
    closed_value,
    open_close_value,
    close_open_value,
    open_close_open_value,
    close_open_close_value,
    median_filter_value
]

# Convert each filtered image from LAB to RGB
rgb_filtered_images = [color.lab2rgb(img_lab, illuminant='D65') for img_lab in filtered_images]

filtered_img_name =  [
    "open filter",
    "close filter",
    "open-close filter",
    "close-open filter",
    "open-close-open filter",
    "close-open-close filter",
    "median filter"
]

plt.figure(figsize=(16, 6))
plt.subplot(1, 3, 1)
plt.imshow(slice_rgb_float)
plt.title('Original RGB Slice (z = {})'.format(z))
plt.axis('on')
plt.subplot(1, 3, 2)
plt.imshow(slice_lab.astype(np.int16))
plt.title('LAB Space Slice (z = {})'.format(z))
plt.axis('on')
plt.subplot(1, 3, 3)
plt.imshow(SE_lab.astype(np.int16))
plt.title('SE LAB')
plt.axis('on')
plt.tight_layout()
plt.show()


luminosity_factor = 2  # Adjust this value as needed
brightened_img = slice_rgb_float * luminosity_factor
plt.figure(figsize=(8, 6))
plt.imshow(brightened_img)
plt.axis('on')
plt.show()



# Plot each filtered image on a separate figure
for i, filtered_img in enumerate(rgb_filtered_images):
    luminosity_factor = 2  # Adjust this value as needed
    brightened_img = filtered_img * luminosity_factor
    # Make sure the pixel values are within the valid range [0, 255]
    brightened_img = np.clip(brightened_img, 0, 1)
    # Plot the brightened image
    plt.figure(figsize=(8, 6))
    plt.imshow(brightened_img)
    plt.title(f"Filtered Image {filtered_img_name[i]}")
    plt.axis('on')
    plt.show()
