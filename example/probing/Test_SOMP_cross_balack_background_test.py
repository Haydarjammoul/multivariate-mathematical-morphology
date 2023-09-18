# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:02:10 2023

@author: hjammoul
Extract SE from image vamidation
Then test CSOMP
"""
from pretreatment import normalize_without_ref, complemantarity
from RGB_MorphoMath_tools import CRA_erosion_rgb, CRA_dilation_rgb, Construct_square_SE_flat, opening, closing, extend_convergence_pts, extract_SE_from_img, Csomp
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
import cv2
###########################################################
path = "C:/Users/hjammoul/Downloads/HMT(avec TP)/HMT(avec TP)/color images/colorshape.png"
rgb_img = cv2.imread(path) # Load the image
size_of_SE = 7 # Size of the SE
posx = 18 # Position of se in
posy = 99
se_background_value = [0,0,0]
O_inf_rgb = np.array((0.,0.,0.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((0.,0.,1.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
##########################################MMAIN
img_lab = normalize_without_ref(rgb_img, size_of_SE)
#img_lab = img_lab1[posy - int(size_of_SE//2) : posy + 33, posx- int(size_of_SE//2): posx +  33]
#
plt.figure(figsize=(12, 8))
plt.imshow(img_lab)
plt.title("img lab")
plt.show()
#
img_lab_ = normalize_without_ref(rgb_img, size_of_SE)
SE_lab = extract_SE_from_img(img_lab_, posx,posy,size_of_SE,se_background_value)#Extract SE
# #SE_lab = color.rgb2lab(SE_rgb, illuminant='D65')
#
plt.figure(figsize=(12, 8))
plt.imshow(SE_lab)
plt.title("SE lab")
plt.show()
#
O_sup_lab,O_inf_lab = extend_convergence_pts(O_sup_lab, O_inf_lab,SE_lab)

# #eroded = CRA_erosion_rgb(img_lab, SE_lab,  O_inf_lab,O_sup_lab)

CSOMP, anti_dilated,eroded = Csomp(img_lab, SE_lab,  O_inf_lab,O_sup_lab)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 2)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 3, 3)
plt.imshow(eroded)
plt.title("Eroded")
plt.subplot(2, 3, 1)
plt.imshow(rgb_img)
plt.title("Img rgb")
plt.subplot(2, 3, 4)
plt.imshow(anti_dilated)
plt.title("Anti_Dilated value")
plt.subplot(2, 3, 5)
plt.imshow(CSOMP)
plt.title("CSOMP")
plt.tight_layout()
plt.show()