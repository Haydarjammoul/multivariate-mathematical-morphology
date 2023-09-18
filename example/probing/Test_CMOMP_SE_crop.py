# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:41:08 2023

@author: hjammoul

CMOMP with SEs are crops from the image cz modelizing neurone with a disk was shitty
test on a single slice
"""
from Synthesize import generate_img_shape_size_difference_effect,get_slice
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from RGB_MorphoMath_tools import  Cmomp,extend_convergence_pts,Construct_round_SE_flat,CRA_erosion_rgb,CRA_dilation_rgb,opening,closing
########################
O_inf_lab =np.array((0.,1.,0.), dtype=np.float32) 
O_sup_rgb = np.array((1.,1.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
#Crop SE from filtered img
#get numpy array of filtered img
filtered_img_lab = np.load('filtered_lab_values.npy')
#SE_sup = filtered_img_lab[655:680,  600:625]
#635,600
x, y = 635, 600
size = 25
SE_inf = filtered_img_lab[x - size // 2: x + size // 2 + 1, y - size // 2: y + size // 2 + 1]
#SE_sup = np.maximum(SE_sup, SE_inf)
#SE_inf = np.minimum(SE_sup, SE_inf)

#extend conv pts
O_sup_lab,O_inf_lab,d_conv_pts = extend_convergence_pts(O_sup_lab, O_inf_lab,SE_inf)
CMOMP, anti_dilated,eroded = Cmomp(filtered_img_lab, SE_inf, SE_inf,  O_inf_lab,O_sup_lab)

###########################"



#plot filtered img
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(filtered_img_lab.astype(np.int16))
plt.title("Close_open_close filtered Image (LAB)") 

filtered_img_rgb = color.lab2rgb(filtered_img_lab, illuminant='D65')  
plt.subplot(2, 3, 2)
plt.imshow(filtered_img_rgb)
plt.title("filtered img (RGB)")

plt.subplot(2, 3, 3)
plt.imshow(SE_inf.astype(np.int16))
plt.title("SE inf(lab)")

plt.subplot(2, 3,4)
plt.imshow(SE_sup.astype(np.int16))
plt.title("SE sup(lab)")

plt.subplot(2, 3, 5)
plt.imshow(CMOMP.astype(np.int16))
plt.title("CMOMP")
# Find the indices where CMOMP values are below 39 and below 41
red_cross_indices = np.where(CMOMP < 26)
blue_cross_indices = np.where((CMOMP < 26) & (CMOMP < 29))
# Plot red crosses where CMOMP is below 39
plt.scatter(red_cross_indices[1], red_cross_indices[0], color='red', marker='x', s=50)
# Plot blue crosses where CMOMP is between 39 and 41
plt.scatter(blue_cross_indices[1], blue_cross_indices[0], color='blue', marker='x', s=50)
cbar = plt.colorbar(orientation='vertical')
cbar.set_label("CMOMP values")

plt.tight_layout()
plt.show()

#save cmomp np array
np.save('cmomp_model_cropped_Csomp_two_models_green_conv_pt.npy', CMOMP)