# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:58:45 2023

@author: hjammoul
CMOMP on a z slice, z-3, z+3
Single template
save them and threshold them later
"""
from Synthesize import generate_img_shape_size_difference_effect,get_slice
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from RGB_MorphoMath_tools import  Cmomp,extend_convergence_pts,Construct_round_SE_flat,CRA_erosion_rgb,CRA_dilation_rgb,opening,closing
########################
# Load the saved filtered LAB values
filtered_lab_values_135 = np.load('filtered_CloseOpenClose_lab_values_135.npy')
filtered_lab_values_130 = np.load('filtered_CloseOpenClose_lab_values_130.npy')
filtered_lab_values_140 = np.load('filtered_CloseOpenClose_lab_values_140.npy')

# Extract SE_sup from the filtered image for slice z=135
SE_sup = filtered_lab_values_135[655:680, 600:625, :]
#extend conv pts
O_inf_lab =np.array((0.,1.,0.), dtype=np.float32) 
O_sup_rgb = np.array((1.,1.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
O_sup_lab,O_inf_lab,d_conv_pts = extend_convergence_pts(O_sup_lab, O_inf_lab,SE_sup)

# Initialize lists to store CMOMP values
CMOMP_values_135 = []
CMOMP_values_130 = []
CMOMP_values_140 = []

# Loop through slices
for filtered_lab_value, CMOMP_values_slice in zip([filtered_lab_values_135, filtered_lab_values_130, filtered_lab_values_140],
                                                   [CMOMP_values_135, CMOMP_values_130, CMOMP_values_140]):
    # Apply CMOMP to each slice
    CMOMP_value, _, _ = Cmomp(filtered_lab_value, SE_sup, SE_sup, O_inf_lab, O_sup_lab)
    # Append CMOMP value to the corresponding list
    CMOMP_values_slice.append(CMOMP_value)

# Convert the lists of CMOMP values to numpy arrays
CMOMP_values_135_array = np.array(CMOMP_values_135)
CMOMP_values_130_array = np.array(CMOMP_values_130)
CMOMP_values_140_array = np.array(CMOMP_values_140)

# Save the CMOMP values arrays for each slice
np.save('CMOMP_values_135.npy', CMOMP_values_135_array)
np.save('CMOMP_values_130.npy', CMOMP_values_130_array)
np.save('CMOMP_values_140.npy', CMOMP_values_140_array)