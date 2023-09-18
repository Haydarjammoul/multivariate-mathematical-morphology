# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:46:25 2023

@author: hjammoul
distance function test and validation
"""
from pretreatment import normalize_without_ref, complemantarity 
from RGB_MorphoMath_tools import erosion_rgb, dilation_rgb ,Construct_square_SE_flat
from Order import compute_distance_map
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
#############################################################
path = "C:/Users/hjammoul/Downloads/HMT(avec TP)/HMT(avec TP)/color images/colorshape.png"
rgb_img = cv2.imread(path) # Load the image
mask_rgb = rgb_img[6:25,44:64]
mask_rgb = mask_rgb[13:14,13:14]
size_of_SE = 5 #size of se
shape_size = 3#shape of square inside SE
Value = np.array((0.,0.,0.), dtype=np.float32) ##value in flat SE in LAB: black
img_lab = normalize_without_ref(mask_rgb,size_of_SE) #convert RGB img to lab 
###
O_inf_lab = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab

O_inf_lab2 = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_sup_rgb2 = np.array((1.,1.,1.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb2, illuminant='D65')
###############################################################
print("O_sup_lab",O_sup_lab)
print("value color lab",img_lab[2:3,2:3])
distance_map = compute_distance_map(img_lab[2:3,2:3], O_sup_lab)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img_lab.astype(np.int16), origin="lower")
axs[0].set_title("Mask lab")
axs[1].imshow(distance_map, origin="lower")
axs[1].set_title("distance map")
#axs[1].set_title(f"dist_map_to_O_sup - max_dist_idx: {max_dist_idx} -value{dist_map_to_O_sup[max_dist_idx]}")
axs[2].imshow(mask_rgb, origin="lower")
axs[2].set_title("Mask rgb")
plt.show()