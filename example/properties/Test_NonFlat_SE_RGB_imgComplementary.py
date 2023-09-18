# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:38:55 2023

@author: hjammoul
On the crosses image with white background
Test Erosion, dilation and their duality in color
"""
from pretreatment import normalize_without_ref, complemantarity 
from RGB_MorphoMath_tools import erosion_rgb, dilation_rgb ,Construct_round_SE_flat
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
#############################################################
path = "C:/Users/hjammoul/Downloads/HMT(avec TP)/HMT(avec TP)/color images/colorshape.png"
rgb_img = cv2.imread(path) # Load the image
size_of_SE = 3 #size of se
Value = np.array((0.,0.,0.), dtype=np.float32) ##value in flat SE in LAB: black
Value = np.array((1.,0.,0.), dtype=np.float32) ##value in flat SE in LAB: White

O_inf_lab = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_sup_lab = np.array((100,0,0), dtype=np.float32)

O_inf_lab2 = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_sup_rgb2 = np.array((1.,1.,1.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab2 = color.rgb2lab(O_sup_rgb2, illuminant='D65')
#######MAIN
###############################################################################################################"
SE_lab = Construct_round_SE_flat(Value,size_of_SE,size_of_SE) #consruct the SE flat of color black
img_lab = normalize_without_ref(rgb_img,size_of_SE) #convert RGB img to lab 
img_lab = complemantarity(O_sup_lab, O_inf_lab, img_lab)  #Complemntary of the img, white background
rgb_img = color.lab2rgb(img_lab,illuminant="D65") #complementary of rgb
eroded_value = erosion_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)
dilated_value =  dilation_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)

eroded_value2 = erosion_rgb(img_lab, SE_lab, O_inf_lab2, O_sup_lab2)# Erosion for O_inf and O_sup pair 2 
dilated_value2 = dilation_rgb(img_lab, SE_lab, O_inf_lab2, O_sup_lab2) # dilation for O_inf and O_sup pair 2 

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
plt.title("non Flat Structuring Element (LAB)")
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
plt.imshow(eroded_value2)
plt.title("Erosion (O_inf: {}, O_sup: {})".format(O_inf_lab2, O_sup_lab2))
# Plot dilated_lab2
plt.subplot(2, 4, 7)
plt.imshow(dilated_value2)
plt.title("Dilation (O_inf: {}, O_sup: {})".format(O_inf_lab2, O_sup_lab2))
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
#To test the duality
#perform complemantary on img_lab, apply erosion and perform complemantrity on it
#plot these two and the dilation of img_lab next to them to compare
C_img_lab = complemantarity(O_sup_lab, O_inf_lab, img_lab) #complemantary of F
eroded = erosion_rgb(C_img_lab, SE_lab,  O_inf_lab,O_sup_lab) #Erosion on C_img
C_erosion = complemantarity(O_sup_lab, O_inf_lab, eroded) #complemantary of Erosion(C_F)
dilated_value =  dilation_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)
#plot img
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(img_lab)
plt.title("Lab Image")
plt.subplot(2, 3, 2)
plt.imshow(C_img_lab)
plt.title("Complementary of img lab ")
# Plot Erosion
plt.subplot(2, 3, 3)
plt.imshow(eroded)
plt.title("Erosion of C_img lab ")
plt.subplot(2, 3, 4)
plt.imshow(C_erosion)
plt.title("Complementary of erosion lab ")
# Plot Dilation
plt.subplot(2, 3, 5)
plt.imshow(dilated_value)
plt.title("Dilation lab")
plt.tight_layout()
# Plot SE
plt.subplot(2, 3, 6)
plt.imshow(SE_lab)
plt.title("Flat SE lab")
plt.tight_layout()
plt.show()

# #2nd Test of Duality
C_img_lab = complemantarity(O_sup_lab, O_inf_lab, img_lab) #complemantary of F
eroded = erosion_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab) #Erosion on img
dilated_value =  dilation_rgb(C_img_lab, SE_lab,  O_inf_lab,O_sup_lab) #Dilation on C_img
C_dilation = complemantarity(O_sup_lab, O_inf_lab, dilated_value) #complemantary of Dilation(C_F)
#plot img
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(img_lab)
plt.title("Lab Image")
plt.subplot(2, 3, 2)
plt.imshow(C_img_lab)
plt.title("Complementary of img lab ")
# Plot Erosion
plt.subplot(2, 3, 3)
plt.imshow(eroded)
plt.title("Erosion of img lab ")
plt.subplot(2, 3, 4)
plt.imshow(dilated_value)
plt.title("Dilation of C_img C_dilation")
# Plot Dilation
plt.subplot(2, 3, 5)
plt.imshow(C_dilation)
plt.title("C_Dilation lab")
plt.tight_layout()
# Plot SE
plt.subplot(2, 3, 6)
plt.imshow(SE_lab)
plt.title("SE lab")
plt.tight_layout()
plt.show()
