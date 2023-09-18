# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:01:36 2023

@author: hjammoul

Apply filtering and CMOMP on ganglia image.
0-Choose convergence point
a-choose filter
b-construct SE: circle 11px.. maybe crope it to adapt to filtered image
c-plot original filtered and CMOMP 
"""
from Synthesize import get_slice,Construct_disk_SE 
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import normalize_without_ref, CRA_median_filtering
from RGB_MorphoMath_tools import  extend_convergence_pts, Cmomp, Construct_round_SE_flat, opening, closing
########################
path = "C:/Users\hjammoul\Desktop\stage Hjammoul\stat_code\MorphoMath"
image_file = r"C:\Users\hjammoul\Desktop\stage Hjammoul\New Aquisition ganglia\ganglia1.tif"
##################################
#O_inf_rgb = np.array((0.,1.,0.), dtype=np.float32) #black is (0,0,0) in lab
#O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_inf_lab =np.array((20,5,10), dtype=np.float32) 
O_sup_rgb = np.array((1.,1.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')

#SE parameteres:
size_of_SE = 13
shape_size = 11
Value_rgb = np.array((1.,1.,0.), dtype=np.float32) 
Value_lab = color.rgb2lab(Value_rgb, illuminant='D65') 
#Construct SE
filter_kernel = Construct_round_SE_flat(Value_lab, size_of_SE, shape_size)
#extend conv pts
O_sup_lab,O_inf_lab,d_conv_pts = extend_convergence_pts(O_sup_lab, O_inf_lab,filter_kernel)
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

#Median filter
median_filter_value = CRA_median_filtering(slice_lab,filter_kernel, O_sup_lab, O_inf_lab)

# Define the values for SE_inf and SE_sup
value_inf = (30, 30, 30)
value_sup = (50, 65, 60)
# Define the sizes for SE_inf and SE_sup
size_inf = 8
size_sup = 30
# Construct SE_inf with disk shape and specified values
SE_inf = Construct_disk_SE(value_inf, size_inf, 61)
# Construct SE_sup with disk shape and specified values
SE_sup = Construct_disk_SE(value_sup, size_sup, 61)

#CMOMP
CMOMP, anti_dilated,eroded = Cmomp(median_filter_value, SE_inf, SE_sup,  O_inf_lab,O_sup_lab)

#plot eroded and anti-dilated too
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.imshow(eroded.astype(np.int16))
plt.title("Eroded")
plt.axis('off')
plt.subplot(2, 1, 2)
plt.imshow(anti_dilated.astype(np.int16))
plt.title("Anti-dilaed Image")
plt.axis('on')
plt.tight_layout()
plt.show()

# Plot the images
plt.figure(figsize=(12, 12))

plt.subplot(3, 2, 1)
plt.imshow(slice_lab.astype(np.int16))
plt.title("Image in lab")
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(median_filter_value.astype(np.int16))
plt.title("Median Filtered Image")
plt.axis('on')

plt.subplot(3, 2, 3)
plt.imshow(SE_inf.astype(np.int16))
plt.title("SE_inf")
plt.axis('on')

plt.subplot(3, 2, 4)
plt.imshow(SE_sup.astype(np.int16))
plt.title("SE_sup")
plt.axis('on')

cropped_CMOMP = CMOMP[2*size_of_SE:-size_of_SE, 2*size_of_SE:-size_of_SE]
plt.subplot(3, 2, 5)
plt.imshow(cropped_CMOMP)
plt.title("CMOMP")
plt.axis('off')
# Add color bar to the CMOMP plot
cbar = plt.colorbar(orientation='vertical')
cbar.set_label("CMOMP values")
# Find the indices where CMOMP is zero
zero_indices = np.where(cropped_CMOMP == np.amin(cropped_CMOMP))
# Plot red crosses where CMOMP is zero
plt.scatter(zero_indices[1], zero_indices[0], color='red', marker='x', s=50)


plt.tight_layout()
plt.show()
