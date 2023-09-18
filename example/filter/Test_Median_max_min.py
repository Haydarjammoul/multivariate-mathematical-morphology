# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:10:12 2023

@author: hjammoul
Median and mim max filters
min max filters are erosions and dilations with flat SE
"""
from Synthesize import generate_img_shape_size_difference_effect,get_slice
import matplotlib.pyplot as plt
from pretreatment import CRA_median_filtering
import numpy as np
from skimage import io, color
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
size_of_SE = 16
shape_size = 15
Value_lab =  np.array((0.,0.,0.), dtype=np.float32) 
#Construct flat SE: Value lab=(0,0,0)
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

# Plot the original RGB slice, its LAB space slice, and the SE_lab using matplotlib
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



#Median filter
median_filter_value = CRA_median_filtering(slice_lab,SE_lab, O_sup_lab, O_inf_lab)
#Min filter: erosion
#min_value = CRA_erosion_rgb(slice_lab, SE_lab,  O_inf_lab,O_sup_lab)
#Max filter: dilation
#Max_value =  CRA_dilation_rgb(slice_lab, SE_lab,  O_inf_lab,O_sup_lab)

#Plot filtered values
# Plot the original RGB slice, its LAB space slice, and the SE_lab using matplotlib
# Define a list of filtered images
filtered_images = [
    median_filter_value,
   # min_value,
    #Max_value
]
filtered_img_name =  [
    "median_filter",
    #"min_filter",
    #"Max_filter"
]

# Convert each filtered image from LAB to RGB
rgb_filtered_images = [color.lab2rgb(img_lab, illuminant='D65') for img_lab in filtered_images]

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

# Plot each filtered image on a separate figure
for i, filtered_img in enumerate(rgb_filtered_images):
    plt.figure(figsize=(8, 6))
    plt.imshow(filtered_img)
    plt.title(f"Filtered Image {filtered_img_name[i]}")
    plt.axis('off')
    plt.show()

#Plot Original image(slice_lab), plot quadratic difference between Original image and filtered(median_filter_value) 
# Compute the quadratic difference between the original and filtered images
quadratic_difference = np.linalg.norm(slice_lab[8:, 8:] - median_filter_value, axis=-1)

# Plot the images
plt.figure(figsize=(12, 6))

# Plot the original image (slice_lab) in lab space
plt.subplot(1, 2, 1)
plt.imshow(slice_lab.astype(np.int16))
plt.title("Original Image (LAB)") 

# Plot the quadratic difference between original and filtered images
plt.subplot(1, 2, 2)
plt.imshow(quadratic_difference.astype(np.int16), cmap='gray')
plt.title("DeltaE Difference between oriinal slice and filtered slice")
# Add a color bar to the right of the second subplot
cbar = plt.colorbar(orientation='vertical')
cbar.set_label('Quadratic Difference', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()