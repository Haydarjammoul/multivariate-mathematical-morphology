# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 21:55:33 2023

@author: hjammoul
Test CRA erosion: validate behavior using complementary of image
"""
from pretreatment import normalize_without_ref, complemantarity 
from RGB_MorphoMath_tools import CRA_erosion_rgb, CRA_dilation_rgb ,Construct_square_SE_flat
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
#############################################################
path = "C:/Users/hjammoul/Downloads/HMT(avec TP)/HMT(avec TP)/color images/colorshape.png"
rgb_img = cv2.imread(path) # Load the image
#rgb_img = rgb_img[0:20,0:20,:]


size_of_SE = 5 #size of se
shape_size = 3#shape of circle inside SE
Value = np.array((0,0,0), dtype=np.float32) ##value in flat SE in LAB: black
###
O_inf_lab = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
#O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((1.,1.,1.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
#######MAIN
###############################################################################################################"
SE_lab = Construct_square_SE_flat(Value,size_of_SE,shape_size) #consruct the SE flat of color black
print("O_sup_lab",O_sup_lab)
img_lab = normalize_without_ref(rgb_img,size_of_SE) #convert RGB img to lab 
img_lab = complemantarity(O_sup_lab, O_inf_lab, img_lab) #complementary of image(white background)

eroded_value = CRA_erosion_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)
dilated_value =  CRA_dilation_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)


# Plot img_rgb
plt.subplot(2, 4, 1)
plt.imshow(rgb_img)
plt.title("RGB Image")
# Plot img_lab
plt.subplot(2, 4, 2)
plt.imshow(img_lab)
plt.title("CIE LAB Image")
# Plot SE_flat_lab
plt.subplot(2, 4, 3)
plt.imshow(SE_lab)
plt.title("Flat Structuring Element (LAB)")
# Plot eroded_lab1
plt.subplot(2, 4, 4)
plt.imshow(eroded_value)
plt.title("Erosion (O_inf: {}, O_sup: {})".format(O_inf_lab, O_sup_lab))
# Plot dilated_lab1
plt.subplot(2, 4, 5) 
plt.imshow(dilated_value)
plt.title("Dilation (O_inf: {}, O_sup: {})".format(O_inf_lab, O_sup_lab))
# Plot eroded_lab2
plt.subplot(2, 4, 6)
plt.imshow(eroded_value.astype(np.int16))
plt.title("Erosion ")
# Plot dilated_lab2
plt.subplot(2, 4, 7)
plt.imshow(dilated_value.astype(np.int16))
plt.title("Dilation ")
plt.tight_layout()
plt.show()

# Convert dilation and erosion to RGB and plot them
eroded_rgb = color.lab2rgb(eroded_value, illuminant='D65')
dilated_rgb = color.lab2rgb(dilated_value, illuminant='D65')
# Plot eroded_rgb
# Plot RGB Image
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title("RGB Image")
# Plot RGB Erosion
plt.subplot(1, 3, 2)
plt.imshow(eroded_rgb)
plt.title("Erosion (RGB)")
# Plot RGB Dilation
plt.subplot(1, 3, 3)
plt.imshow(dilated_rgb)
plt.title("Dilation (RGB)")
plt.tight_layout()
plt.show()

