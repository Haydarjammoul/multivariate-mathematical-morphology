# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:12:35 2023

@author: hjammoul

Experimenting with multivariate morphological filtering and probing parameters: 
    Aim is to detect more neurons in the images
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color


from lib.synthesize import get_slice,construct_disk_se 
from lib.pretreatment import normalize_without_ref, cra_median_filtering
from lib.rgb_math_morphology_tools import  extend_convergence_pts, cmomp, construct_round_se_flat, opening, closing, open_close_open_filter

os.makedirs('./filtered_imgs', exist_ok=True)
os.makedirs('./cmomp_values', exist_ok=True)

THRESHOLD = 15/39

path = "C:/Users\hjammoul\Desktop\stage Hjammoul\stat_code\MorphoMath"
image_file = r"C:\Users\hjammoul\Desktop\stage Hjammoul\New Aquisition ganglia\ganglia1.tif"
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

#background is black, sup value=yelllow, inf value = green
#ITS IMPERATIVE FOR THE COLOR COORDINATES TO BE DEFINED AS FLOATS
value_sup_rgb = (1.,1.,0.)
value_inf_rgb = (0.,1.,0.)
value_sup_lab =  color.rgb2lab(value_sup_rgb, illuminant='D65')
value_inf_lab = color.rgb2lab(value_inf_rgb, illuminant='D65')
background_rgb = (0.,0.,0.)
background_lab = color.rgb2lab(background_rgb, illuminant='D65')
O_inf_lab = np.array((0., 0., 0.), dtype=np.float32)
O_sup_rgb = np.array((1., 1., 0.), dtype=np.float32)
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')

shape_size_list = [11, 13,17,23,25,30]
for shape_size in shape_size_list:
    print(shape_size)
    #inside loop for each shape_size
    se_sup = construct_disk_se(value_sup_lab, int(shape_size//2)+1, shape_size+2 , background_lab)
    se_inf = construct_disk_se(value_inf_lab, int(shape_size//2)-1, shape_size+2 , background_lab)
    O_sup_lab, O_inf_lab, d_conv_pts = extend_convergence_pts(O_sup_lab, O_inf_lab, se_sup)
    #apply filter
    filter_kernel =  construct_round_se_flat(value_sup_lab, shape_size, shape_size)
    filtered_lab = open_close_open_filter(slice_lab, filter_kernel, O_inf_lab, O_sup_lab)
    filtered_rgb = color.lab2rgb(filtered_lab, illuminant='D65')
    #apply probing
    CMOMP_value, anti_dilation, erosion = cmomp(filtered_lab, se_inf, se_sup, O_inf_lab, O_sup_lab)
    #save files(.npy): filtered_lab img and cmomp
    np.save(f'./filtered_imgs/filtered_lab_{shape_size}.npy', filtered_lab)
    np.save(f'./cmomp_values/cmomp_value_{shape_size}.npy', CMOMP_value)
    #plot filtered slice rgb,
    luminosity_factor = 2  # Adjust this value as needed
    brightened_img = filtered_rgb * luminosity_factor
    brightened_img = np.clip(brightened_img, 0, 1) # Make sure the pixel values are within the valid range [0, 255]
    # Plot the brightened image
    plt.figure(figsize=(8, 6))
    plt.imshow(brightened_img)
    plt.title(f"Filtered Image (Shape Size {shape_size})")
    plt.show()
    #Thresholding
    thresholded_cmomp = CMOMP_value[50:,50:] < THRESHOLD
    y_indices, x_indices = np.where(thresholded_cmomp)
    #plot cmomp value with crosses on the figure where value is bellow cmomp_value
    plt.figure()
    plt.imshow(CMOMP_value[50:, 50:].astype(np.float), cmap='viridis')  # Adjust the colormap as needed
    plt.title(f'Cmomp')
    plt.scatter(x_indices, y_indices, color='red', marker='x', s=50)
    plt.colorbar()
    plt.show()


