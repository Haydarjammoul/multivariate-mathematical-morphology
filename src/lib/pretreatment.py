# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 08:26:58 2023

@author: hjammoul

Pre-treatment functions such as adjusting the dynamics of the image in RGB before conversion to LAB, adding padding, reflectivity, complementarity, CRA based median filtering...

Functions:
    reflectivity
    normalize_without_ref
    add_padding_to_img
    complemantarity
    CRA_median_filtering
    construct_conv_kernel
    construct_conv_kernel
    
"""
import numpy as np
from skimage import io, color


from lib.order import cra_median
from lib.rgb_math_morphology_tools import construct_round_se_flat, construct_square_se_flat,construct_linear_se_flat
####################""""
def reflectivity(se):
    """
    Apply reflectivity to a structuring element.

    Anti-dilation requires reflectivity on SE.
    For non-flat SE, a transposition around the origin is needed.

    Parameters:
        se (numpy.ndarray): The structuring element.

    Returns:
        numpy.ndarray: The reflected structuring element.

    Example:
        reflected_se = reflectivity(se)
    """
    ##SE shape needs to be impaire
    origin = (se.shape[0] // 2, se.shape[1] // 2)  # Compute the origin of the structuring element
    reflected_se = np.flip(se, axis=0)  # Flip the SE along the vertical axis
    reflected_se = np.flip(reflected_se, axis=1)  # Flip the SE along the horizontal axis
    reflected_se[origin[0], origin[1]] = se[origin[0], origin[1]]  # Restore the original value at the origin
    return reflected_se

def normalize_without_ref(rgb,se_size):
    """
    Before converting to lab the dnamic of values will be to [0,1]
    
    Normalize the RGB image without assuming a white reference and convert to LAB color space using illuminant D65.
    Calls add_padding_to_img to add padding to the image.

    Parameters:
        rgb (numpy.ndarray): The synthesized color image of 8 or 16 bit.
        se_size (int): The size of the corresponding SE.

    Returns:
        numpy.ndarray(float32): The normalized LAB image.

    Example:
        img_lab = normalize_without_ref(rgb_image, 5)
    """
    rgb = add_padding_to_img(rgb,se_size)
    # Normalize the RGB image
    rgb_float = rgb.astype(np.float32)
    rgb_normalized = rgb_float / 255.
    # Convert RGB image to CIE LAB color space using illuminant D65
    img_lab = color.rgb2lab(rgb_normalized, illuminant='D65')
    return img_lab

def add_padding_to_img(img,se_size):
    """
   Add padding to a 2D RGB image.

   Parameters:
       img (numpy.ndarray): The 2D image array.
       se_size (int): The size of the corresponding SE.

   Returns:
       numpy.ndarray: The padded image.

   Example:
       padded_img = add_padding_to_img(img, 3)
   """
    pad_size = int(np.floor(2*se_size))
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    return padded_img

def complemantarity(O_sup_lab, O_inf_lab, img_lab):
    """
   Compute the complementary of each color coordinate in CIE LAB format.

   Parameters:
       img_lab (numpy.ndarray): Image in CIE LAB format.
       O_sup_lab (tuple): Color coordinates in CIE LAB for the upper bound (O_sup).
       O_inf_lab (tuple): Color coordinates in CIE LAB for the lower bound (O_inf).

   Returns:
       numpy.ndarray: Complementary of the image.

   Example:
       complementary_image = complemantarity(O_sup_lab, O_inf_lab, img_lab)
   """

    # Compute the midpoint between O_inf and O_sup
    midpoint = (np.array(O_inf_lab) + np.array(O_sup_lab)) / 2.0

    # Compute the complementary of each color coordinate
    C_img = midpoint - (img_lab - midpoint)

    return C_img


def cra_median_filtering(img_lab,conv_kernel, O_sup_lab, O_inf_lab):
    """
   Perform median filtering on color images in LAB space.

   Parameters:
       img_lab (numpy.ndarray): The image in LAB format.
       conv_kernel (numpy.ndarray): The kernel that defines the surroundings of the pixel to be considered in the filtering iteration.
       O_sup_lab (tuple): Color coordinates in CIE LAB for the upper bound (O_sup).
       O_inf_lab (tuple): Color coordinates in CIE LAB for the lower bound (O_inf).

   Returns:
       numpy.ndarray: Filtered LAB image.

   Example:
       filtered_image = CRA_median_filtering(img_lab, conv_kernel, O_sup_lab, O_inf_lab)
   """
    pad_size = int(np.floor(conv_kernel.shape[0] // 2))
    filtered_img_lab_size = (img_lab.shape[0]-pad_size, img_lab.shape[1]-pad_size, 3)
    filtered_img_lab = np.zeros(filtered_img_lab_size, dtype=np.float32)
    # Create a mask of NaN values, true where value is not NaN
    nan_mask = np.invert(np.isnan(conv_kernel))
    for i in range(pad_size, img_lab.shape[0] - pad_size):
        for j in range(pad_size, img_lab.shape[1] - pad_size):
            # Consider the surrounding mask around the pixel
            mask = img_lab[i - pad_size: i + pad_size+1, j - pad_size: j + pad_size+1,:]
            median_idx = cra_median(mask, O_sup_lab, O_inf_lab, nan_mask)
            #print("Mask:",mask)
            #print("median idx",median_idx)
            #print("filtered value:",img_lab[median_idx[0], median_idx[1], :])
            #put the corresponding value inside filtered_img_lab
            filtered_img_lab[i, j, :] = mask[median_idx[0], median_idx[1], :]
    return filtered_img_lab.astype(np.float32)


def construct_conv_kernel(shape, size):
    """
    Construct a convolution kernel for filtering.

    Parameters:
        shape (str): The shape of the convolution kernel. Either 'square', 'round', or 'linear'.
        size (int): The size of the convolution kernel.

    Returns:
        numpy.ndarray: The constructed convolution kernel.

    Example:
        kernel = construct_conv_kernel('square', 5)
    """
    # Define the value to fill the SE with (here, np.nan)
    value = (1,1,1)
    # Define the size of the surrounding area of the pixel to be considered
    #radius or square size
    shape_size = size
    if shape == 'square':
        # Use the provided function to construct the square SE
        kernel = construct_square_SE_flat(value, size, shape_size)
    elif shape == 'round':
        # Use the provided function to construct the circular SE
        kernel = construct_round_SE_flat(value, size, shape_size)
    elif shape == 'linear':
            # Use the provided function to construct the circular SE
            kernel =  construct_linear_SE_flat(value, shape_size)
    else:
        raise ValueError("Invalid shape. Supported shapes are 'square' and 'round' or 'linear'.")
    
    return kernel