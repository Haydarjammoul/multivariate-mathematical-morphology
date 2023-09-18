# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:37:57 2023

@author: hjammoul
Test idempatence of opening and closing in Color MM
Opening= dilation of the erosion
Closing = Erosion of dilation
"""
from pretreatment import normalize_without_ref, complemantarity 
from RGB_MorphoMath_tools import erosion_rgb, dilation_rgb ,Construct_SE_nonFlat
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
SE_lab = Construct_SE_nonFlat(Value,size_of_SE) #consruct the SE flat of color black
img_lab = normalize_without_ref(rgb_img,size_of_SE) #convert RGB img to lab 
img_lab = complemantarity(O_sup_lab, O_inf_lab, img_lab)  #Complemntary of the img, white background
rgb_img = color.lab2rgb(img_lab,illuminant="D65") #complementary of rgb
eroded_value = erosion_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)
dilated_value =  dilation_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)
# Apply opening operation
opened_value = dilation_rgb(erosion_rgb(img_lab, SE_lab, O_inf_lab, O_sup_lab), SE_lab, O_inf_lab, O_sup_lab)

# Apply closing operation
closed_value = erosion_rgb(dilation_rgb(img_lab, SE_lab, O_inf_lab, O_sup_lab), SE_lab, O_inf_lab, O_sup_lab)

