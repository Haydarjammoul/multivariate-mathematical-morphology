# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:16:26 2023

@author: hjammoul
Ordering & distance computation is here

Ordering_fct_inf & Ordering_fct_sup have the same implementation but different philosophies (WaW)
"""

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
#####################################################################
def compute_distance_lab_img(image1, image2):
    """
    Compute the Euclidean distance map between pixels of two images in CIELAB color space.

    Parameters:
        image1 (ndarray): The first image.
        image2 (ndarray): The second image.

    Returns:
        ndarray: The distance map, representing the Euclidean distances between the pixels of the two images.
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    # Compute the Euclidean distances between the pixels of the two images
    distance_map = np.linalg.norm(image1 - image2, axis=2)
    return distance_map
def compute_distance_map(displaced_mask, ref_A):
    """
    Compute the distance map between the pixels of a displaced mask and a reference point in CIELAB color space.
    
    Parameters:
        displaced_mask (ndarray): The mask after displacement by a structuring element (shape: (se_size, se_size, 3)).
        ref_A (ndarray): The LAB coordinates of the reference point O+ or O- (shape: (3,)).

    Returns:
        ndarray: The distance map, representing the Euclidean distances between each pixel and the reference point.
    """
    
    # Reshape the displaced mask to a 2D array of shape (num_pixels, 3)
    ref_A_2d = ref_A.reshape(-1, 3)
    pixels = displaced_mask.reshape(-1, 3) ## transform the cube into a 2D array (nblg xnbcol, 3)
    # Compute the distance between each pixel and ref_A
    distances = cdist(pixels, ref_A_2d, metric='euclidean')
    # Reshape the distance array to match the original mask shape
    distance_map = distances.reshape(displaced_mask.shape[:2])
    return distance_map


###NEW ordering function these will only compare to O- and O+
def ordering_fct_inf(distamp_inf):
    """
    Compute the pixel index having the closest point O- based on a given distance map.
    
    Parameters:
        distamp_inf (numpy.ndarray): Distance map of the displaced mask's inferior pixels to the reference point.

    Returns:
        tuple: Index (row, column) of the chosen pixel in the `g_inf` array.
    """
    sorted_indices = np.argwhere(distamp_inf == np.nanmin(distamp_inf))
    chosen_index = tuple(sorted_indices[0])
    return chosen_index
def ordering_fct_sup(distamp_sup):
    """
    Compute the pixel index having the closest point O+ based on a given distance map.
    
    Parameters:
        distamp_sup (numpy.ndarray): Distance map of the displaced mask's superior pixels to the reference point.

    Returns:
        tuple: Index (row, column) of the chosen pixel in the `g_sup` array.
    """
    #discard NaN values positions
    sorted_indices = np.argwhere(distamp_sup == np.nanmin(distamp_sup))
    chosen_index = tuple(sorted_indices[0])
    return chosen_index

##########CRA ordering

def cra_sup(displaced_mask, O_sup_lab, O_inf_lab, nan_mask):
    """
    Calculate the pixel index with the maximum g(Ci) value based on CRA ordering for superior pixels.
    
    Rational:
        Compute distance of each pixel to both O_sup and O_inf => 2 distance maps
        use them to compute g(Ci) array 
        g(Ci) = d(O-,Ci)/d(O+,Ci)
        g value higher => Ci is HIGHER +> USED FOR SUP SEARCH
        g value lower => Ci is lOWER +> USED FOR INF SEARCH
        particular cases:
        Maximum: inf/nan g(Ci) => Ci is equal to O+
        Minimum: g(Ci)=0 => Ci is equal to O-
        Find maximum:
        In g(Ci) array, if at a certain position there is inf/nan, return its indices.
        Otherwise, return the position of the minimum g value.
        
    Parameters:
        displaced_mask (ndarray): Mask after displacement by a structuring element (shape: (se_size, se_size, 3)).
        O_sup_lab (ndarray): LAB coordinates of the upper convergence point (O+) (shape: (3,)).
        O_inf_lab (ndarray): LAB coordinates of the lower convergence point (O-) (shape: (3,)).
        nan_mask (ndarray): Mask indicating NaN values (shape: (se_size, se_size, 1)).

    Returns:
        tuple: Index (row, column) of the chosen pixel for superior convergence.
    """
    
    # Reshape the displaced mask to a 2D array of shape (num_pixels, 3)
    O_sup_lab = O_sup_lab.reshape(-1, 3)
    O_inf_lab = O_inf_lab.reshape(-1, 3)
    pixels = displaced_mask.reshape(-1, 3)
    
    # Compute the distance between each pixel and O_sup_lab and O_inf_lab
    distances_to_O_sup = cdist(pixels, O_sup_lab, metric='euclidean')
    distances_to_O_inf = cdist(pixels, O_inf_lab, metric='euclidean')
    
    # Reshape the distance arrays to match the original mask shape
    distances_to_O_sup = distances_to_O_sup.reshape(displaced_mask.shape[:2])
    distances_to_O_inf = distances_to_O_inf.reshape(displaced_mask.shape[:2])
    
    # Compute g(Ci) array
    g_values = distances_to_O_inf / distances_to_O_sup
    
    # Check if a certain element of the g_values is nan/inf
    nan_inf_indices = np.where((np.isnan(g_values) | np.isinf(g_values)) & nan_mask[:,:,0])
    if len(nan_inf_indices[0]) > 0:
        # Return the indices of the first nan/inf value found this is O+
        return nan_inf_indices[0][0], nan_inf_indices[1][0]
    else:
        # Discard positions in g_values where nan_mask is False
        g_values[~nan_mask[:,:,0]] = np.nan
        # Find the minimum index in g_values
        max_index = np.unravel_index(np.nanargmax(g_values), g_values.shape)
        return max_index
 
def cra_inf(displaced_mask, O_sup_lab, O_inf_lab, nan_mask):
    """
    Calculate the pixel index with the minimum g(Ci) value based on CRA ordering for inferior pixels.
    
    Rational:
        Compute distance of each pixel to both O_sup and O_inf => 2 distance maps
        use them to compute g(Ci) array 
        g(Ci) = d(O-,Ci)/d(O+,Ci)
        g value higher => Ci is HIGHER +> USED FOR SUP SEARCH
        g value lower => Ci is lOWER +> USED FOR INF SEARCH
        particular cases:
        Maximum: inf/nan g(Ci) => Ci is equal to O+
        Minimum: g(Ci)=0 => Ci is equal to O-
        Find maximum:
        In g(Ci) array, if at a certain position there is inf/nan, return its indices.
        Otherwise, return the position of the minimum g value.
    
    Parameters:
        displaced_mask (ndarray): Mask after displacement by a structuring element (shape: (se_size, se_size, 3)).
        O_sup_lab (ndarray): LAB coordinates of the upper convergence point (O+) (shape: (3,)).
        O_inf_lab (ndarray): LAB coordinates of the lower convergence point (O-) (shape: (3,)).
        nan_mask (ndarray): Mask indicating NaN values (shape: (se_size, se_size, 1)).

    Returns:
        tuple: Index (row, column) of the chosen pixel for inferior convergence.
    """
    # Reshape the displaced mask to a 2D array of shape (num_pixels, 3)
    O_sup_lab = O_sup_lab.reshape(-1, 3)
    O_inf_lab = O_inf_lab.reshape(-1, 3)
    pixels = displaced_mask.reshape(-1, 3)
    
    # Compute the distance between each pixel and O_sup_lab and O_inf_lab
    distances_to_O_sup = cdist(pixels, O_sup_lab, metric='euclidean')
    distances_to_O_inf = cdist(pixels, O_inf_lab, metric='euclidean')
    
    # Reshape the distance arrays to match the original mask shape
    distances_to_O_sup = distances_to_O_sup.reshape(displaced_mask.shape[:2])
    distances_to_O_inf = distances_to_O_inf.reshape(displaced_mask.shape[:2])
    
    # Compute g(Ci) array
    g_values =  distances_to_O_sup / distances_to_O_inf
    # plt.figure(figsize=(12, 8))
    # plt.imshow(g_values)
    # plt.title("g_values")
    # plt.show()
    # Check if a certain element of the g_values is nan/inf
    nan_inf_indices = np.where((np.isnan(g_values) | np.isinf(g_values)) & nan_mask[:,:,0])
    if len(nan_inf_indices[0]) > 0:
        # Return the indices of the first nan/inf value found this is O+
        return nan_inf_indices[0][0], nan_inf_indices[1][0]
    else:
        # Discard positions in g_values where nan_mask is False
        g_values[~nan_mask[:,:,0]] = np.nan
        # Find the minimum index in g_values
        max_index = np.unravel_index(np.nanargmax(g_values), g_values.shape)
        return max_index
  
    """
    Rational:
        Compute distance of each pixel to both O_sup and O_inf => 2 distance maps
        use them to compute g(Ci) array 
        g(Ci) = d(O-,Ci)/d(O+,Ci)
        g value higher => Ci is HIGHER +> USED FOR SUP SEARCH
        g value lower => Ci is lOWER +> USED FOR INF SEARCH
        particular cases:
        Maximum: inf/nan g(Ci) => Ci is equal to O+
        Minimum: g(Ci)=0 => Ci is equal to O-
        Find maximum:
        In g(Ci) array, if at a certain position there is inf/nan, return its indices.
        Otherwise, return the position of the minimum g value.
    """
    
    # Reshape the displaced mask to a 2D array of shape (num_pixels, 3)
    O_sup_lab = O_sup_lab.reshape(-1, 3)
    O_inf_lab = O_inf_lab.reshape(-1, 3)
    pixels = displaced_mask.reshape(-1, 3)
    
    # Compute the distance between each pixel and O_sup_lab and O_inf_lab
    distances_to_O_sup = cdist(pixels, O_sup_lab, metric='euclidean')
    distances_to_O_inf = cdist(pixels, O_inf_lab, metric='euclidean')
    
    # Reshape the distance arrays to match the original mask shape
    distances_to_O_sup = distances_to_O_sup.reshape(displaced_mask.shape[:2])
    distances_to_O_inf = distances_to_O_inf.reshape(displaced_mask.shape[:2])
    
    # Compute g(Ci) array
    g_values = distances_to_O_inf / distances_to_O_sup
    # Check if a certain element of the g_values is null
    # Return the indices of the first null value found
    null_indices = np.where((distances_to_O_inf == 0) & nan_mask[:, :, 0])
    if null_indices[0].size > 0:
        return null_indices[0][0], null_indices[1][0]
    else:
        # Discard positions in g_values where nan_mask is False
        g_values[~nan_mask[:,:,0]] = np.nan
        # Find the minimum index in g_values
        min_index = np.unravel_index(np.nanargmin(g_values), g_values.shape)
        return min_index
    
def compute_distance_lab_img(image1, image2):
    """
    Compute the Euclidean distance map between pixels of two images in CIELAB color space.
    Parameters:
        image1 (ndarray): The first image.
        image2 (ndarray): The second image.
    Returns:
        ndarray: The distance map, representing the Euclidean distances between the pixels of the two images.
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    # Compute the Euclidean distances between the pixels of the two images
    distance_map = np.linalg.norm(image1 - image2, axis=2)
    return distance_map

def cra_median(displaced_mask, O_sup_lab, O_inf_lab, nan_mask):
    """
     Calculate the pixel index with the median g(Ci) value based on CRA ordering.
    
        Rational:
            Compute distance of each pixel to both O_sup and O_inf => 2 distance maps
            use them to compute g(Ci) array 
            g(Ci) = d(O-,Ci)/d(O+,Ci)
            g value higher => Ci is HIGHER +> USED FOR SUP SEARCH
            g value lower => Ci is lOWER +> USED FOR INF SEARCH
            particular cases:
            Maximum: inf/nan g(Ci) => Ci is equal to O+
            Minimum: g(Ci)=0 => Ci is equal to O-
            Find median:
            In g(Ci) array, calculate the median value and find the index of the pixel with the closest g value to the median.
            Return the index of the pixel with the closest g value to the median.
    
     Parameters:
         displaced_mask (ndarray): Mask after displacement by a structuring element (shape: (se_size, se_size, 3)).
         O_sup_lab (ndarray): LAB coordinates of the upper convergence point (O+) (shape: (3,)).
         O_inf_lab (ndarray): LAB coordinates of the lower convergence point (O-) (shape: (3,)).
         nan_mask (ndarray): Mask indicating NaN values (shape: (se_size, se_size, 1)).
    
     Returns:
         tuple: Index (row, column) of the chosen pixel for median convergence.
     """
    #print("pixels:",displaced_mask)
    # Reshape the displaced mask to a 2D array of shape (num_pixels, 3)
    O_sup_lab = O_sup_lab.reshape(-1, 3)
    O_inf_lab = O_inf_lab.reshape(-1, 3)
    pixels = displaced_mask.reshape(-1, 3)
    # Compute the distance between each pixel and O_sup_lab and O_inf_lab
    distances_to_O_sup = cdist(pixels, O_sup_lab, metric='euclidean')
    distances_to_O_inf = cdist(pixels, O_inf_lab, metric='euclidean')
    
    # Reshape the distance arrays to match the original mask shape
    distances_to_O_sup = distances_to_O_sup.reshape(displaced_mask.shape[:2])
    distances_to_O_inf = distances_to_O_inf.reshape(displaced_mask.shape[:2])
    
    # Compute g(Ci) array
    g_values = distances_to_O_inf / distances_to_O_sup
    #nan_mask: true where value is not nan
    # Check if a certain element of the g_values is nan/inf
    nan_inf_indices = np.where((np.isnan(g_values) | np.isinf(g_values)) & nan_mask[:,:,0])
    if len(nan_inf_indices[0]) > 0:
        # Return the indices of the first nan/inf value found this is O+
        return nan_inf_indices[0][0], nan_inf_indices[1][0]
    else:
        # Discard positions in g_values where nan_mask is False
        g_values[~nan_mask[:,:,0]] = np.nan
        #print("g_values inside median ordering:",g_values)
        # Calculate the median value of g_values
        median_value = np.nanmedian(g_values)
        #print("median g-value:",median_value)
        # Find the index of the pixel with the closest g value to the median
        median_index = np.unravel_index(np.nanargmin(np.abs(g_values - median_value)), g_values.shape)
        return median_index