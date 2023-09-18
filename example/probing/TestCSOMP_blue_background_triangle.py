# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 10:33:10 2023

@author: hjammoul
Test CSOMP on blue background
threshold on 60; distance_betwee_conv_pts//14 
background has distance(btw erosion & anti-dilation) of 117= distance_betwee_conv_pts//8 
"""

from Synthesize import generate_color_img
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import normalize_without_ref, complemantarity
from RGB_MorphoMath_tools import  extend_convergence_pts, Csomp

##################################
O_inf_rgb = np.array((0.,0.,1.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((1.,0.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
img_rgb, se_rgb = generate_color_img(height_img=200, num_shapes=10, shape_min_size=10, shape_max_size=10, mode='CSOMP'
                                     ,img_background_color= (0,0,255), target_color = (255,0,0) )
size_of_SE = se_rgb.shape[0]
##################################

#Plot RGB img and SE
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(img_rgb)
plt.title("RGB Image")
plt.subplot(2, 1, 2)
plt.imshow(se_rgb)
plt.title("RGB SE ")
plt.tight_layout()
plt.show()

#Convert rgb img & se to lab space
img_lab = normalize_without_ref(img_rgb, size_of_SE) #normalizes by dividing by 255 and convert to lab, also adds padding of size_of_SE

se_rgb_float = se_rgb.astype(np.float32)
se_normalized = se_rgb_float / 255.
se_lab = color.rgb2lab(se_normalized, illuminant='D65')
#Plot LAB img and SE
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.imshow(img_lab)
plt.title("LAB Image")
plt.subplot(2, 1, 2)
plt.imshow(se_lab)
plt.title("LAB SE ")
plt.tight_layout()
plt.show()

#Apply CSOMP 
O_sup_lab,O_inf_lab = extend_convergence_pts(O_sup_lab, O_inf_lab,se_lab)

CSOMP, anti_dilated,eroded = Csomp(img_lab, se_lab,  O_inf_lab,O_sup_lab)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 2)
plt.imshow(img_lab)
plt.title("lab Image")
plt.subplot(2, 3, 3)
plt.imshow(eroded.astype(np.int8))
plt.title("Eroded")
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Img rgb")
plt.subplot(2, 3, 4)
plt.imshow(anti_dilated.astype(np.int8))
plt.title("Anti_Dilated value")
plt.subplot(2, 3, 5)
plt.imshow(CSOMP)
plt.title("CSOMP")
plt.tight_layout()
plt.show()
