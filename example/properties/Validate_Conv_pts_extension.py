# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:04:45 2023

@author: hjammoul
Validate Convergence points coordinates extension
"""
from pretreatment import normalize_without_ref, complemantarity
from RGB_MorphoMath_tools import CRA_erosion_rgb, CRA_dilation_rgb, Construct_square_SE_flat, opening, closing, extend_convergence_pts
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
###########################################################
path = "C:/Users/hjammoul/Downloads/HMT(avec TP)/HMT(avec TP)/color images/colorshape.png"
rgb_img = cv2.imread(path) # Load the image
size_of_SE = 2 # Size of the SE
shape_size = 2 # Shape of the circle inside the SE
Value = np.array((0, 0, 0), dtype=np.float32) # Value in flat SE in LAB: black
O_inf_rgb = np.array((1.,1.,1.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((0.,0.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
##########################################MMAIN
# Construct the SE
SE_lab = Construct_square_SE_flat(Value, size_of_SE, shape_size)
#Extension of convergence points along their axis
O_sup_lab,O_inf_lab = extend_convergence_pts(O_sup_lab, O_inf_lab,SE_lab)
# Convert RGB image to LAB
img_lab = normalize_without_ref(rgb_img, size_of_SE)
C_img_lab = complemantarity(O_sup_lab, O_inf_lab, img_lab) #complemantary of image
#To test duality of erosion and dilation: 
    #compute erosion on image complementary then compute its complementary 
    #plot the latter next to image dilation to see if duality is verified
eroded_C_img = CRA_erosion_rgb(C_img_lab, SE_lab,  O_inf_lab,O_sup_lab) #Erosion on C_img
C_erosion = complemantarity(O_sup_lab, O_inf_lab, eroded_C_img) #complemantary of Erosion(C_F)
dilated_value =  CRA_dilation_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)
#plot img rgb, img lab,C_img_lab, C_erosion, dilation of img lab 
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 2, 3)
plt.imshow(C_erosion)
plt.title("Complementary of Erosion of complementaryof img")
plt.subplot(2, 2, 2)
plt.imshow(rgb_img)
plt.title("Img rgb")
plt.subplot(2, 2, 4)
plt.imshow(dilated_value)
plt.title("Dilated value")
plt.tight_layout()
plt.show()

#Test duality of opening and closing
opened_C_img = opening(C_img_lab, SE_lab, O_inf_lab, O_sup_lab)
C_opening = complemantarity(O_sup_lab, O_inf_lab, opened_C_img) #complemantary of Opening(C_F)
closed_value = closing(img_lab, SE_lab, O_inf_lab, O_sup_lab)
#plot img rgb, img lab,C_img_lab, C_opening, closing of img lab 
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 2, 3)
plt.imshow(C_opening)
plt.title("Complementary of Opening of complementaryof img")
plt.subplot(2, 2, 2)
plt.imshow(rgb_img)
plt.title("Img rgb")
plt.subplot(2, 2, 4)
plt.imshow(closed_value)
plt.title("Closed value")
plt.tight_layout()
plt.show()
#Test idempotence of closing and opening
# Perform opening on opening
opened_value = opening(img_lab, SE_lab, O_inf_lab, O_sup_lab)
opened_opened_value = opening(opened_value, SE_lab, O_inf_lab, O_sup_lab)
# Perform closing on closing
closed_closed_value = closing(closed_value, SE_lab, O_inf_lab, O_sup_lab)
#plot img rgb, img lab,opening, Cosing, opening on opening, closing on closing 
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 3, 2)
plt.imshow(opened_value)
plt.title("Opening")
plt.subplot(2, 3, 3)
plt.imshow(closed_value)
plt.title("Closing")
plt.subplot(2, 3, 4)
plt.imshow(rgb_img)
plt.title("Img rgb")
plt.subplot(2, 3, 5)
plt.imshow(opened_opened_value)
plt.title("Opening on Opening")
plt.subplot(2, 3, 6)
plt.imshow(closed_closed_value)
plt.title("Closing on Closing")
plt.tight_layout()
plt.show()
##Repeat duality tests and idempotence test for the C_img_lab instead
#...
img_lab = C_img_lab
C_img_lab = complemantarity(O_sup_lab, O_inf_lab, img_lab)

#To test duality of erosion and dilation: 
    #compute erosion on image complementary then compute its complementary 
    #plot the latter next to image dilation to see if duality is verified
eroded_C_img = CRA_erosion_rgb(C_img_lab, SE_lab,  O_inf_lab,O_sup_lab) #Erosion on C_img
C_erosion = complemantarity(O_sup_lab, O_inf_lab, eroded_C_img) #complemantary of Erosion(C_F)
dilated_value =  CRA_dilation_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)
#plot img rgb, img lab,C_img_lab, C_erosion, dilation of img lab 
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 2, 3)
plt.imshow(C_erosion)
plt.title("Complementary of Erosion of complementaryof img")
plt.subplot(2, 2, 2)
plt.imshow(rgb_img)
plt.title("Img rgb")
plt.subplot(2, 2, 4)
plt.imshow(dilated_value)
plt.title("Dilated value")
plt.tight_layout()
plt.show()

#Test duality of opening and closing
opened_C_img = opening(C_img_lab, SE_lab, O_inf_lab, O_sup_lab)
C_opening = complemantarity(O_sup_lab, O_inf_lab, opened_C_img) #complemantary of Opening(C_F)
closed_value = closing(img_lab, SE_lab, O_inf_lab, O_sup_lab)
#plot img rgb, img lab,C_img_lab, C_opening, closing of img lab 
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 2, 3)
plt.imshow(C_opening)
plt.title("Complementary of Opening of complementaryof img")
plt.subplot(2, 2, 2)
plt.imshow(rgb_img)
plt.title("Img rgb")
plt.subplot(2, 2, 4)
plt.imshow(closed_value)
plt.title("Closed value")
plt.tight_layout()
plt.show()
#Test idempotence of closing and opening
# Perform opening on opening
opened_value = opening(img_lab, SE_lab, O_inf_lab, O_sup_lab)
opened_opened_value = opening(opened_value, SE_lab, O_inf_lab, O_sup_lab)
# Perform closing on closing
closed_closed_value = closing(closed_value, SE_lab, O_inf_lab, O_sup_lab)
#plot img rgb, img lab,opening, Cosing, opening on opening, closing on closing 
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 3, 2)
plt.imshow(opened_value)
plt.title("Opening")
plt.subplot(2, 3, 3)
plt.imshow(closed_value)
plt.title("Closing")
plt.subplot(2, 3, 4)
plt.imshow(rgb_img)
plt.title("Img rgb")
plt.subplot(2, 3, 5)
plt.imshow(opened_opened_value)
plt.title("Opening on Opening")
plt.subplot(2, 3, 6)
plt.imshow(closed_closed_value)
plt.title("Closing on Closing")
plt.tight_layout()
plt.show()
