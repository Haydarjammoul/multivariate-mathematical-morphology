# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:13:34 2023

@author: hjammoul
CRA test idempotence
"""
from pretreatment import normalize_without_ref, complemantarity
from RGB_MorphoMath_tools import CRA_erosion_rgb, CRA_dilation_rgb, Construct_square_SE_flat, opening, closing
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2

#############################################################
path = "C:/Users/hjammoul/Downloads/HMT(avec TP)/HMT(avec TP)/color images/colorshape.png"
rgb_img = cv2.imread(path) # Load the image
size_of_SE = 2 # Size of the SE
shape_size = 2 # Shape of the circle inside the SE
Value = np.array((0, 0, 0), dtype=np.float32) # Value in flat SE in LAB: black
O_inf_lab = np.array((-100.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
#O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((1.,1.,1.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
O_sup_lab[0]=200
###########################Main
# Construct the SE
SE_lab = Construct_square_SE_flat(Value, size_of_SE, shape_size)
# Convert RGB image to LAB
img_lab = normalize_without_ref(rgb_img, size_of_SE)
# Perform opening (dilation of the erosion)
opened_value = opening(img_lab, SE_lab, O_inf_lab, O_sup_lab)
# Perform closing (erosion of the dilation)
closed_value = closing(img_lab, SE_lab, O_inf_lab, O_sup_lab)

# Perform opening on closing
opened_opened_value = opening(opened_value, SE_lab, O_inf_lab, O_sup_lab)
# Perform closing on opening
closed_closed_value = closing(closed_value, SE_lab, O_inf_lab, O_sup_lab)


# Plot the results
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
plt.subplot(2, 3, 5)
plt.imshow(opened_opened_value)
plt.title("Opening on Opening")
plt.subplot(2, 3, 6)
plt.imshow(closed_closed_value)
plt.title("Closing on Closing")
plt.tight_layout()
plt.show()
