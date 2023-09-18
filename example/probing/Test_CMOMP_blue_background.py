# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 17:12:34 2023

@author: hjammoul
CMOMP on image with blue background
At 60 detection = distance_between_conv_pts //21
"""
from Synthesize import generate_color_img
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from pretreatment import normalize_without_ref, complemantarity
from RGB_MorphoMath_tools import  extend_convergence_pts, Cmomp

##################################
O_inf_rgb = np.array((0.,0.,1.), dtype=np.float32) #black is (0,0,0) in lab
O_inf_lab = color.rgb2lab(O_inf_rgb, illuminant='D65')
O_sup_rgb = np.array((1.,0.,0.), dtype=np.float32)  # [1.,1.,1.] is array([ 1.00000000e+02, -2.45493786e-03,  4.65342115e-03]) in lab
O_sup_lab = color.rgb2lab(O_sup_rgb, illuminant='D65')
img_rgb, se_inf_rgb, se_sup_rgb = generate_color_img(height_img=200, num_shapes=10, shape_min_size=10, shape_max_size=10, mode='CMOMP'
                                     ,img_background_color= (0,0,255), target_color = (250,20,20) )
size_of_SE = se_inf_rgb.shape[0]
##################################

#Plot RGB img and SE
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.imshow(img_rgb)
plt.title("RGB Image")
plt.subplot(3, 1, 2)
plt.imshow(se_inf_rgb)
plt.title("RGB SE inf ")
plt.subplot(3, 1, 3)
plt.imshow(se_sup_rgb)
plt.title("RGB SE sup ")
plt.tight_layout()
plt.show()

#Convert rgb img & se to lab space
img_lab = normalize_without_ref(img_rgb, size_of_SE) #normalizes by dividing by 255 and convert to lab, also adds padding of size_of_SE

se_inf_rgb_float = se_inf_rgb.astype(np.float32)
se_inf_normalized = se_inf_rgb_float / 255.
se_inf_lab = color.rgb2lab(se_inf_normalized, illuminant='D65')

se_sup_rgb_float = se_sup_rgb.astype(np.float32)
se_sup_normalized = se_sup_rgb_float / 255.
se_sup_lab = color.rgb2lab(se_sup_normalized, illuminant='D65')

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.imshow(img_lab)
plt.title("LAB Image")
plt.subplot(3, 1, 2)
plt.imshow(se_inf_lab)
plt.title("LAB SE inf ")
plt.subplot(3, 1, 3)
plt.imshow(se_sup_lab)
plt.title("LAB SE sup ")
plt.tight_layout()
plt.show()

O_sup_lab,O_inf_lab,d_conv_pts = extend_convergence_pts(O_sup_lab, O_inf_lab,se_sup_lab)
CMOMP, anti_dilated,eroded = Cmomp(img_lab, se_sup_lab, se_sup_lab,  O_inf_lab,O_sup_lab)

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
plt.imshow(CMOMP)
plt.title("CMOMP")
plt.tight_layout()
plt.show()

