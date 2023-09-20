# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 13:01:28 2023

@author: hjammoul
RGB_MorphoMath_tools
Dilation & erosion_rgb perform ordering usuch as lower is closest to O- and higher closest to O+
CRA_dilation and erosion (allow idempotence & duality) perform ordering based on both convergence coordinates
opening & closing
Extension_of_convergence_points: choose ur colors and it makes them more distant to prevent saturation 
and ensure the transformations are increasing
Ensure_convegence: to prevent divergence and thus ensure the tansformations are increasing
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from lib.order import compute_distance_map, ordering_fct_inf,ordering_fct_sup,cra_sup, cra_inf, compute_distance_lab_img
###########################################################

def erosion_rgb(img_lab, se_lab,  O_inf_lab,O_sup_lab):
    """
    Performs erosion on an RGB image using a structuring element (SE).

    Erosion is a morphological operation that shrinks the shapes in an image by considering the local neighborhood
    around each pixel and comparing it with the corresponding elements of the SE. The resulting image represents the
    minimum values of the pixel colors within the SE.

    Parameters:
        img_lab (numpy.ndarray): The input image in LAB space.
        se_lab (numpy.ndarray): The structuring element (SE) in LAB space.
        O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
        O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

    Returns:
        numpy.ndarray: The eroded image after applying the erosion operation in LAB space.

    Example:
    >>> img_lab = np.array(...)  # Replace with your LAB image
    >>> se_lab = np.array(...)   # Replace with your LAB SE
    >>> O_inf_lab = np.array([32.29567432, 79.18557525, -107.85729885])
    >>> O_sup_lab = np.array([100.0, -0.00244379, 0.00466108])
    >>> eroded_image = erosion_rgb(img_lab, se_lab, O_inf_lab, O_sup_lab)
     # Now 'eroded_image' contains the result of the erosion operation.
    """
    #distance between convergence points
    distance_ref = np.linalg.norm(O_sup_lab - O_inf_lab)
    # Define the padding size, se shape is always impaire
    pad_size = int(np.floor(se_lab.shape[0] // 2))
    # Compute the eroded image size
    eroded_size = (img_lab.shape[0]-pad_size, img_lab.shape[1]-pad_size, 3)
    eroded_image = np.zeros(eroded_size, dtype=np.float32) 
    # Apply erosion on each pixel in the image
    for i in range(pad_size, img_lab.shape[0] - pad_size):
        for j in range(pad_size, img_lab.shape[1] - pad_size):
            # Consider the surrounding mask around the pixel
            mask = img_lab[i - pad_size: i + pad_size+1, j - pad_size: j + pad_size+1,:]
            mask_size = mask.shape
            displaced_mask = np.zeros(mask_size, dtype=np.float32)
            # Create a mask of NaN values, true where value is not NaN
            nan_mask = np.invert(np.isnan(se_lab))
            ##if mask contains O_inf value=>just take its index in min_dist_idx
            # Check if O_inf_lab exists in mask[nan_mask, :]
            O_inf_exists = np.any(np.all(np.isclose(mask[nan_mask[:, :, 0]], O_inf_lab, atol=1), axis=-1))
            if O_inf_exists:
                ##the condtion may be always true!!!!!!
                min_dist_idx = np.argwhere( np.ceil(mask) ==  np.ceil(O_inf_lab))[0]
                displaced_mask[min_dist_idx[0], min_dist_idx[1], :] = O_inf_lab
            else:
                # Compute the displacement vector from x to O_inf: vector CO-
                displacement_vector = O_inf_lab.astype(np.float32) - mask.astype(np.float32)
                # Normalize the displacement vector: ensembe of vectors(vecteur directeur de deplacement) for each pixel:
                normalized_displacement = displacement_vector.astype(np.float32) / np.linalg.norm(displacement_vector, axis=-1, keepdims=True) 
                #print("shape normalized_displacement",normalized_displacement.shape)
                #print(" nan values in normalized_displacement",np.isnan(normalized_displacement[:,:,0]))
                # Compute the displacement value SE * normalized_displacement = E * vector(C, O_inf)
                ##magnitudes of se_lab along axis 2
                se_magnitude = np.linalg.norm(np.nan_to_num(se_lab), axis=-1)
                se_magnitude_reshaped = se_magnitude[:, :, np.newaxis]
                displacement_vector = np.multiply(normalized_displacement.astype(np.float32), se_magnitude_reshaped)
                displaced_mask = displacement_vector + mask
                #Ensure non-divergence from convergence  point
                # Clip the values in displaced_mask to ensure they are within the range of O_inf_lab and O_sup_lab
                displaced_mask[:, :, 0] = np.clip(displaced_mask[:, :, 0], -0., 100.)
                ##Compute the distance map of the displaced mask's pixels to O_inf
                dist_map_to_O_inf = compute_distance_map(displaced_mask, O_inf_lab)
                ##in the dist map put NaNs in the positions where nan_mask is false
                min_dist_idx = ordering_fct_inf(dist_map_to_O_inf)
            # Update the eroded image with the chosen pixel value
            eroded_image[i, j, :] = displaced_mask[min_dist_idx[0], min_dist_idx[1], :]
    #eroded_image_rgb = color.lab2rgb(eroded_image, illuminant='D65')
    return eroded_image.astype(np.float32)

def dilation_rgb(img_lab, se_lab,  O_inf_lab,O_sup_lab):
    """
    Performs dilation on an RGB image using a structuring element (SE).

    Dilation is a morphological operation that expands the shapes in an image by considering the local neighborhood
    around each pixel and comparing it with the corresponding elements of the SE. The resulting image represents the
    maximum values of the pixel colors within the SE.

    Parameters:
        img_lab (numpy.ndarray): The input image in LAB space.
        se_lab (numpy.ndarray): The structuring element (SE) in LAB space.
        O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
        O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

    Returns:
        numpy.ndarray: The dilated image after applying the dilation operation in LAB space.

    Example:
    >>> img_lab = np.array(...)  # Replace with your LAB image
    >>> se_lab = np.array(...)   # Replace with your LAB SE
    >>> O_inf_lab = np.array([32.29567432, 79.18557525, -107.85729885])
    >>> O_sup_lab = np.array([100.0, -0.00244379, 0.00466108])
    >>> dilated_image = dilation_rgb(img_lab, se_lab, O_inf_lab, O_sup_lab)
     # Now 'dilated_image' contains the result of the dilation operation.
    """
    # Convert RGB image to Lab color space using illuminant D65
    distance_ref = np.linalg.norm(O_sup_lab - O_inf_lab)
    # Define the padding size
    pad_size = int(np.floor(se_lab.shape[0] // 2))
    # Compute the eroded image size
    dilated_size = (img_lab.shape[0]-pad_size, img_lab.shape[1]-pad_size, 3)
    # Create an empty eroded image
    dilated_image = np.zeros(dilated_size, dtype=np.float32)
    # Apply erosion on each pixel in the image
    for i in range(pad_size, img_lab.shape[0] - pad_size ):
        for j in range(pad_size, img_lab.shape[1] - pad_size):
            # Consider the surrounding mask around the pixel
            mask = img_lab[i - pad_size: i + pad_size+1, j - pad_size: j + pad_size+1,:]
            mask_size = mask.shape
            displaced_mask = np.zeros(mask_size, dtype=np.float32)
            # Create a mask of NaN values
            nan_mask = np.invert(np.isnan(se_lab))
            ##if mask contains O_inf value=>just take its index in min_dist_idx
            # Check if O_inf_lab exists in mask[nan_mask, :]
            O_sup_exists = np.any(np.all(np.isclose(mask[nan_mask[:, :, 0]], O_sup_lab, atol=1), axis=-1))
            #print(np.ceil(mask[nan_mask[:,:,0],:]))
            if O_sup_exists:
                #for this to work O_sup_lab needs to be part of CIE lab
                max_dist_idx = np.argwhere( np.ceil(mask) ==  np.ceil(O_sup_lab))[0]
                displaced_mask[max_dist_idx[0], max_dist_idx[1], :] = O_sup_lab
            else:
                # Compute the displacement vector from x to O_inf
                displacement_vector = O_sup_lab.astype(np.float32) - mask.astype(np.float32)
                #print(mask)
                # Normalize the displacement vector: ensembe of vectors for each pixel: shape:(31,31,3)
                normalized_displacement = displacement_vector.astype(np.float32) / np.linalg.norm(displacement_vector, axis=-1, keepdims=True)
                # Compute the displacement value SE * normalized_displacement = E * vector(C, O_inf)
                # ##magnitudes of se_lab along axis 2
                # Compute the magnitude of se_lab, handling NaN values
                se_magnitude = np.linalg.norm(np.nan_to_num(se_lab), axis=-1)
                se_magnitude_reshaped = se_magnitude[:, :, np.newaxis]
                displacement_vector = np.multiply(normalized_displacement.astype(np.float32), se_magnitude_reshaped)
                displaced_mask = displacement_vector + mask
                # Clip the values in displaced_mask to ensure they are within the range of O_inf_lab and O_sup_lab
                #displaced_mask[:, :, 0] = np.clip(displaced_mask[:, :, 0], -0., 100.) #To ensure convergence(prevent divergence)
                #Compute distance maps and do the ordering to 
                dist_map_to_O_sup = compute_distance_map(displaced_mask, O_sup_lab)
                max_dist_idx =  ordering_fct_sup(dist_map_to_O_sup)
                # Update the eroded image with the chosen pixel value
            dilated_image[i, j, :] = displaced_mask[max_dist_idx[0], max_dist_idx[1], :]
    return dilated_image.astype(np.float32)


def cra_erosion_rgb(img_lab, se_lab,  O_inf_lab,O_sup_lab):
    """
    Performs erosion on an RGB image using a structuring element (SE). Ordering based on CRA

    Erosion is a morphological operation that shrinks the shapes in an image by considering the local neighborhood
    around each pixel and comparing it with the corresponding elements of the SE. The resulting image represents the
    minimum values of the pixel colors within the SE.

    Parameters:
        img_lab (numpy.ndarray): The input image in LAB space.
        se_lab (numpy.ndarray): The structuring element (SE) in LAB space.
        O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
        O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

    Returns:
        numpy.ndarray: The eroded image after applying the erosion operation in LAB space.

    Example:
    >>> img_lab = np.array(...)  # Replace with your LAB image
    >>> se_lab = np.array(...)   # Replace with your LAB SE
    >>> O_inf_lab = np.array([32.29567432, 79.18557525, -107.85729885])
    >>> O_sup_lab = np.array([100.0, -0.00244379, 0.00466108])
    >>> eroded_image = erosion_rgb(img_lab, se_lab, O_inf_lab, O_sup_lab)
     # Now 'eroded_image' contains the result of the erosion operation.
    """
    #distance between convergence points
    distance_ref = np.linalg.norm(O_sup_lab - O_inf_lab)
    # Define the padding size, se shape is always impaire
    pad_size = int(np.floor(se_lab.shape[0] // 2))
    # Compute the eroded image size
    eroded_size = (img_lab.shape[0]-pad_size, img_lab.shape[1]-pad_size, 3)
    eroded_image = np.zeros(eroded_size, dtype=np.float32) 
    # Apply erosion on each pixel in the image
    for i in range(pad_size, img_lab.shape[0] - pad_size):
        for j in range(pad_size, img_lab.shape[1] - pad_size):
            # Consider the surrounding mask around the pixel
            mask = img_lab[i - pad_size: i + pad_size+1, j - pad_size: j + pad_size+1,:]
            mask_size = mask.shape
            displaced_mask = np.zeros(mask_size, dtype=np.float32)
            # Create a mask of NaN values, true where value is not NaN
            nan_mask = np.invert(np.isnan(se_lab))
            ##if mask contains O_inf value=>just take its index in min_dist_idx
            # Check if O_inf_lab exists in mask[nan_mask, :]
            O_inf_exists = np.any(np.all(np.isclose(mask[nan_mask[:, :, 0]], O_inf_lab, atol=1), axis=-1))
            if O_inf_exists:
                ##the condtion may be always true!!!!!!
                min_dist_idx = np.argwhere( np.ceil(mask) ==  np.ceil(O_inf_lab))[0]
                displaced_mask[min_dist_idx[0], min_dist_idx[1], :] = O_inf_lab
            else:
                # Compute the displacement vector from x to O_inf: vector CO-
                displacement_vector = O_inf_lab.astype(np.float32) - mask.astype(np.float32)
                # Normalize the displacement vector: ensembe of vectors(vecteur directeur de deplacement) for each pixel:
                normalized_displacement = displacement_vector.astype(np.float32) / np.linalg.norm(displacement_vector, axis=-1, keepdims=True) 
                #print("shape normalized_displacement",normalized_displacement.shape)
                #print(" nan values in normalized_displacement",np.isnan(normalized_displacement[:,:,0]))
                # Compute the displacement value SE * normalized_displacement = E * vector(C, O_inf)
                ##magnitudes of se_lab along axis 2
                se_magnitude = np.linalg.norm(np.nan_to_num(se_lab), axis=-1)
                se_magnitude_reshaped = se_magnitude[:, :, np.newaxis]
                displacement_vector = np.multiply(normalized_displacement.astype(np.float32), se_magnitude_reshaped)
                displaced_mask = displacement_vector + mask
                ##in the dist map put NaNs in the positions where nan_mask is false
                min_dist_idx = cra_inf(displaced_mask, O_sup_lab, O_inf_lab, nan_mask)
            # Update the eroded image with the chosen pixel value
            eroded_image[i, j, :] = displaced_mask[min_dist_idx[0], min_dist_idx[1], :]
    return eroded_image.astype(np.float32)

def cra_dilation_rgb(img_lab, se_lab,  O_inf_lab,O_sup_lab):
    """
    Performs dilation on an RGB image using a structuring element (SE). Performs erosion on an RGB image using a structuring element (SE). Ordering based on CRA

    Dilation is a morphological operation that expands the shapes in an image by considering the local neighborhood
    around each pixel and comparing it with the corresponding elements of the SE. The resulting image represents the
    maximum values of the pixel colors within the SE.

    Parameters:
        img_lab (numpy.ndarray): The input image in LAB space.
        se_lab (numpy.ndarray): The structuring element (SE) in LAB space.
        O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
        O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

    Returns:
        numpy.ndarray: The dilated image after applying the dilation operation in LAB space.

    Example:
    >>> img_lab = np.array(...)  # Replace with your LAB image
    >>> se_lab = np.array(...)   # Replace with your LAB SE
    >>> O_inf_lab = np.array([32.29567432, 79.18557525, -107.85729885])
    >>> O_sup_lab = np.array([100.0, -0.00244379, 0.00466108])
    >>> dilated_image = dilation_rgb(img_lab, se_lab, O_inf_lab, O_sup_lab)
     # Now 'dilated_image' contains the result of the dilation operation.
    """
    # Convert RGB image to Lab color space using illuminant D65
    distance_ref = np.linalg.norm(O_sup_lab - O_inf_lab)
    # Define the padding size
    pad_size = int(np.floor(se_lab.shape[0] // 2))
    # Compute the eroded image size
    dilated_size = (img_lab.shape[0]-pad_size, img_lab.shape[1]-pad_size, 3)
    # Create an empty eroded image
    dilated_image = np.zeros(dilated_size, dtype=np.float32)
    # Apply erosion on each pixel in the image
    for i in range(pad_size, img_lab.shape[0] - pad_size ):
        for j in range(pad_size, img_lab.shape[1] - pad_size):
            # Consider the surrounding mask around the pixel
            mask = img_lab[i - pad_size: i + pad_size+1, j - pad_size: j + pad_size+1,:]
            mask_size = mask.shape
            displaced_mask = np.zeros(mask_size, dtype=np.float32)
            # Create a mask of NaN values
            nan_mask = np.invert(np.isnan(se_lab))
            ##if mask contains O_inf value=>just take its index in min_dist_idx
            # Check if O_inf_lab exists in mask[nan_mask, :]
            O_sup_exists = np.any(np.all(np.isclose(mask[nan_mask[:, :, 0]], O_sup_lab, atol=1), axis=-1))
            #print(np.ceil(mask[nan_mask[:,:,0],:]))
            if O_sup_exists:
                #for this to work O_sup_lab needs to be part of CIE lab
                max_dist_idx = np.argwhere( np.ceil(mask) ==  np.ceil(O_sup_lab))[0]
                displaced_mask[max_dist_idx[0], max_dist_idx[1], :] = O_sup_lab
            else:
                # Compute the displacement vector from x to O_inf
                displacement_vector = O_sup_lab.astype(np.float32) - mask.astype(np.float32)
                # Normalize the displacement vector: ensembe of vectors for each pixel: shape:(31,31,3)
                normalized_displacement = displacement_vector.astype(np.float32) / np.linalg.norm(displacement_vector, axis=-1, keepdims=True)
                # Compute the displacement value SE * normalized_displacement = E * vector(C, O_inf)
                # ##magnitudes of se_lab along axis 2
                # Compute the magnitude of se_lab, handling NaN values
                se_magnitude = np.linalg.norm(np.nan_to_num(se_lab), axis=-1)
                se_magnitude_reshaped = se_magnitude[:, :, np.newaxis]
                displacement_vector = np.multiply(normalized_displacement.astype(np.float32), se_magnitude_reshaped)
                displaced_mask = displacement_vector + mask
                #Compute distance maps and do the ordering to 
                max_dist_idx = cra_sup(displaced_mask, O_sup_lab, O_inf_lab, nan_mask)
                # Update the eroded image with the chosen pixel value
            dilated_image[i, j, :] = displaced_mask[max_dist_idx[0], max_dist_idx[1], :]
    return dilated_image.astype(np.float32)

def cra_anti_dilation_rgb(img_lab, se_lab, O_inf_lab, O_sup_lab):
    """
    Performs Anti-dilation on an RGB image using a structuring element (SE).

    Anti-dilation is a morphological operation that dilates towards the O- color convergence point the shapes in an image by considering the local neighborhood
    around each pixel and comparing it with the corresponding elements of the SE. The resulting image represents the
    maximum values of the pixel colors within the SE, displacing the colors towards O- (black).

    Parameters:
        img_lab (numpy.ndarray): The input image in LAB space (dtype: float32).
        se_lab (numpy.ndarray): The structuring element (SE) in LAB space (dtype: float32).
        O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
        O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

    Returns:
        numpy.ndarray: The anti-dilated image after applying the anti-dilation operation in LAB space (dtype: float32).
    """

    #distance between convergence points
    distance_ref = np.linalg.norm(O_sup_lab - O_inf_lab)
    # Define the padding size, se shape is always impaire
    pad_size = int(np.floor(se_lab.shape[0] // 2))
    # Compute the eroded image size
    anti_dilated_size = (img_lab.shape[0]-pad_size, img_lab.shape[1]-pad_size, 3)
    anti_dilated = np.zeros(anti_dilated_size, dtype=np.float32) 
    # Apply erosion on each pixel in the image
    for i in range(pad_size, img_lab.shape[0] - pad_size):
        for j in range(pad_size, img_lab.shape[1] - pad_size):
            # Consider the surrounding mask around the pixel
            mask = img_lab[i - pad_size: i + pad_size+1, j - pad_size: j + pad_size+1,:]
            mask_size = mask.shape
            displaced_mask = np.zeros(mask_size, dtype=np.float32)
            # Create a mask of NaN values, true where value is not NaN
            nan_mask = np.invert(np.isnan(se_lab))
            ##if mask contains O_inf value=>just take its index in min_dist_idx
            # Check if O_inf_lab exists in mask[nan_mask, :] : discard it, I want the max
            nan_mask[np.all(mask == O_inf_lab, axis=-1)] = False
            # Compute the displacement vector from x to O_inf: vector CO-
            displacement_vector = O_inf_lab.astype(np.float32) - mask.astype(np.float32)
            # Normalize the displacement vector: ensembe of vectors(vecteur directeur de deplacement) for each pixel:
            normalized_displacement = displacement_vector.astype(np.float32) / np.linalg.norm(displacement_vector, axis=-1, keepdims=True) 
            # Compute the displacement value SE * normalized_displacement = E * vector(C, O_inf)
            ##magnitudes of se_lab along axis 2
            se_magnitude = np.linalg.norm(np.nan_to_num(se_lab), axis=-1)
            se_magnitude_reshaped = se_magnitude[:, :, np.newaxis]
            displacement_vector = np.multiply(normalized_displacement.astype(np.float32), se_magnitude_reshaped)
            displaced_mask = displacement_vector + mask
            #Ensure non-divergence from convergence  point
            ##Compute the distance map of the displaced mask's pixels to O_inf
            ##in the dist map put NaNs in the positions where nan_mask is false
            min_dist_idx = cra_sup(displaced_mask, O_sup_lab, O_inf_lab, nan_mask)
            # Update the anti-dilated image with the chosen pixel value
            anti_dilated[i, j, :] = displaced_mask[min_dist_idx[0], min_dist_idx[1], :]
    return anti_dilated.astype(np.float32)

def opening(value, SE_lab, O_inf_lab, O_sup_lab):
    """
   Performs morphological opening on an RGB image using a structuring element (SE).

   Opening is a morphological operation that first erodes the input image using the specified structuring element,
   followed by a dilation operation. It helps to remove small bright spots while preserving the larger structures.

   Parameters:
       value (numpy.ndarray): The input image in LAB space.
       SE_lab (numpy.ndarray): The structuring element (SE) in LAB space.
       O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
       O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

   Returns:
       numpy.ndarray: The result of the opening operation in LAB space.
   """
    opened_value = cra_erosion_rgb(value, SE_lab, O_inf_lab, O_sup_lab)
    #SE_lab = reflectivity(SE_lab)
    opened_opened_value = cra_dilation_rgb(opened_value, SE_lab, O_inf_lab, O_sup_lab)
    return opened_opened_value

def closing(value, SE_lab, O_inf_lab, O_sup_lab):
    """
   Performs closing on an RGB image using a structuring element (SE).

   Closing is a morphological operation that first dilates the input image using the specified structuring element,
   followed by an erosion operation. It helps to close small dark gaps while preserving the larger structures.

   Parameters:
       value (numpy.ndarray): The input image in LAB space.
       SE_lab (numpy.ndarray): The structuring element (SE) in LAB space.
       O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
       O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

   Returns:
       numpy.ndarray: The result of the closing operation in LAB space.
   """
    closed_value = cra_dilation_rgb(value, SE_lab, O_inf_lab, O_sup_lab)
    #SE_lab = reflectivity(SE_lab)
    closed_closed_value = cra_erosion_rgb(closed_value, SE_lab, O_inf_lab, O_sup_lab)
    return closed_closed_value

def open_close_open_filter(value, SE_lab, O_inf_lab, O_sup_lab):
    opened = opening(value, SE_lab, O_inf_lab, O_sup_lab)
    closed_opened = closing(opened, SE_lab, O_inf_lab, O_sup_lab)
    opene_close_opened_lab = opening(closed_opened, SE_lab, O_inf_lab, O_sup_lab)
    return opene_close_opened_lab

def csomp(img_lab, se_lab,  O_inf_lab,O_sup_lab):
    """
    Performs the Color Space Opening by Morphological Processing (CSOMP) on an RGB image using a structuring element (SE).

    CSOMP is a morphological operation that computes the difference between an anti-dilated image and an eroded image
    using the specified structuring element. It is useful for detecting and enhancing fine structures in color images.

    Parameters:
        img_lab (numpy.ndarray): The input image in LAB space.
        se_lab (numpy.ndarray): The structuring element (SE) in LAB space.
        O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
        O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing the CSOMP result, the anti-dilated image,
        and the eroded image in LAB space.
    """
    from lib.pretreatment import reflectivity
    #Eroded image with se structuring element
    eroded = cra_erosion_rgb(img_lab, se_lab,  O_inf_lab,O_sup_lab)
    #Anti-dilate with Refelectivity of se
    #se = reflectivity(se_lab)
    anti_dilated = cra_anti_dilation_rgb(img_lab, se_lab,  O_inf_lab,O_sup_lab)
    #CSOMP=Anti_dilation - Erosion
    CSOMP = compute_distance_lab_img(anti_dilated,eroded)
    return CSOMP, anti_dilated,eroded

def cmomp(img_lab, se_inf_lab, su_sup_lab,  O_inf_lab,O_sup_lab):
    """
    Performs the Color Space Morphological Opening and Morphological Processing (CMOMP) on an RGB image
    using structuring elements (SEs).

    CMOMP is a morphological operation that computes the difference between an anti-dilated image and an eroded image
    using two different structuring elements. It is useful for detecting and enhancing fine structures in color images.

    Parameters:
        img_lab (numpy.ndarray): The input image in LAB space.
        se_inf_lab (numpy.ndarray): The structuring element for erosion in LAB space.
        su_sup_lab (numpy.ndarray): The structuring element for anti-dilation in LAB space.
        O_inf_lab (numpy.ndarray): LAB coordinates of the color convergence point O-.
        O_sup_lab (numpy.ndarray): LAB coordinates of the color convergence point O+.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing the CMOMP result, the anti-dilated image,
        and the eroded image in LAB space.
    """
    
    from lib.pretreatment import reflectivity
    #Eroded image with se structuring element
    eroded = cra_erosion_rgb(img_lab, se_inf_lab,  O_inf_lab,O_sup_lab)
    #Anti-dilate with Refelectivity of se
    #se = reflectivity(se_lab)
    su_sup_lab = su_sup_lab[:,::-1,:]
    anti_dilated = cra_anti_dilation_rgb(img_lab, su_sup_lab,  O_inf_lab,O_sup_lab)
    #CSOMP=Anti_dilation - Erosion
    CMOMP = compute_distance_lab_img(anti_dilated,eroded)
    return CMOMP, anti_dilated,eroded
###########################################################################################
def construct_square_se_flat(Value,size_of_SE,shape_size):
    """
    Creates a flat structuring element (SE) as an array of size size_of_SE * size_of_SE.
    The SE is a square with pixel values equal to Value in LAB space, and pixels not part of the square are assigned as NaN.

    Parameters:
        Value (numpy.ndarray): CIE LAB coordinates as a numpy array (e.g., np.array([L, a, b])).
        size_of_SE (int): Size of the square SE.
        shape_size (int): Size of the square within the SE.

    Returns:
        numpy.ndarray: Flat SE with specified size and shape in LAB space.
    """
    # Check if size_of_SE is odd
    if size_of_SE % 2 == 0:
        size_of_SE += 1
    # Create an array of NaN values
    SE = np.full((size_of_SE, size_of_SE,3), np.nan, dtype=np.float32)

    # Calculate the starting and ending indices for the square
    start_index = (size_of_SE - shape_size) // 2
    end_index = start_index + shape_size
    # Fill the square region with the specified Value
    SE[start_index:end_index, start_index:end_index] = Value
    return SE

def construct_round_se_flat(Value, size_of_SE, shape_size):
    """
   Creates a flat linear structuring element (SE) as an array of size (3, 3, 3).
   The SE represents a simple horizontal line with pixel values equal to Value in LAB space.

   Parameters:
       Value (numpy.ndarray): CIE LAB coordinates as a numpy array (e.g., np.array([L, a, b])).
       shape_size (int): Size of the linear SE (should be odd).

   Returns:
       numpy.ndarray: Flat linear SE of specified shape in LAB space.
   """
    # Check if size_of_SE is odd
    if size_of_SE % 2 == 0:
        size_of_SE += 1
    # Create an array of NaN values
    SE = np.full((size_of_SE, size_of_SE, 3), np.nan, dtype=np.float32)
    # Compute the center of the circle
    center = (size_of_SE - 1) // 2
    # Compute the radius of the circle
    radius = shape_size // 2
    # Compute the squared radius for efficient calculation
    squared_radius = radius ** 2
    # Iterate over each pixel in the SE and check if it is inside the circle
    for i in range(size_of_SE):
        for j in range(size_of_SE):
            # Compute the squared distance from the center
            squared_distance = (i - center) ** 2 + (j - center) ** 2
            # Check if the pixel is inside the circle
            if squared_distance <= squared_radius:
                # Assign the value to the pixel
                SE[i, j] = Value
    return SE

def construct_linear_se_flat(Value, shape_size):
    """
    Creates an array of size (3, 3, 3).
    Represents a simple horizontal line with the specified Value.
    Parameters
    ----------
    Value : numpy.ndarray
        CIE lab coordinates as a numpy array (e.g., np.array([L, a, b]))
    shape_size : int
        Size of the linear SE (should be odd).

    Returns
    -------
    SE : numpy.ndarray
        Flat linear SE of shape (3, 3, 3).
    """
    # Check if shape_size is odd
    if shape_size % 2 == 0:
        shape_size += 1

    # Create an array of NaN values
    SE = np.full((shape_size, shape_size, 3), np.nan, dtype=np.float32)

    # Calculate the starting and ending indices for the line
    start_index = (shape_size - shape_size) // 2
    end_index = start_index + shape_size

    # Fill the line region with the specified Value
    SE[int(shape_size//2), start_index:end_index, :] = Value

    return SE

def extract_se_from_img(img, posx, posy,size_se,background_value):
    """
    Extracts a rectangular structuring element (SE) from an LAB image located at position (posx, posy).
    The SE is a rectangle with the specified size, and pixels outside the rectangle are assigned the background_value.

    Parameters:
        img (numpy.ndarray): LAB image from which to extract the SE.
        posx (int): X-coordinate of the SE's center.
        posy (int): Y-coordinate of the SE's center.
        size_se (int): Size of the rectangular SE.
        background_value (numpy.ndarray): CIE LAB coordinates for pixels outside the SE.

    Returns:
        numpy.ndarray: Extracted SE in LAB space.
    """
    if size_se % 2 == 0:
        size_se += 1
    SE = img[posy - int(size_se//2) :posy + int(size_se//2)+1, posx- int(size_se//2): posx +  int(size_se//2)+1]
    #put nan where SE array has (0,0,0) values
    #SE[np.all(SE == [0, 0, 0], axis=-1)] = np.nan
    #32.29567432   79.18557525 -107.85729885 / [1.00000000e+02, -2.44379044e-03,  4.66108322e-03]
    SE[np.all(np.floor(SE) != [32.,79., -107.], axis=-1)] = background_value
    return SE
###############################################################################
def extend_convergence_pts(O_sup_lab, O_inf_lab,SE):
    """
    Extends the convergence points O_sup_lab and O_inf_lab based on a structuring element (SE).
    The extension is performed to prevent saturation, and the new points are calculated along the direction
    of the translation indicated by the SE.

    Parameters:
        O_sup_lab (numpy.ndarray): LAB coordinates of the upper convergence point.
        O_inf_lab (numpy.ndarray): LAB coordinates of the lower convergence point.
        SE (numpy.ndarray): Structuring element used to determine the extension direction.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, float]: A tuple containing the new upper convergence point,
        the new lower convergence point, and the distance between them.
    """
    #Direction vector of translation of O_sup
    direction_O_sup_translation = O_sup_lab - O_inf_lab
    #diveide direction vector with its norm
    normalized_direction_O_sup = direction_O_sup_translation / np.linalg.norm(direction_O_sup_translation, axis=-1)
    #Direction vector of translation of O_inf_lab
    direction_O_inf_translation = O_inf_lab - O_sup_lab 
    #diveide direction vector with its norm
    normalized_direction_O_inf = direction_O_inf_translation / np.linalg.norm(direction_O_inf_translation, axis=-1)
    #norm of displacement = max_norm(SE)*3
    #compute norme of each se vector & get the max 
    se_magnitude = np.linalg.norm(np.nan_to_num(SE), axis=-1)
    se_max = 10*np.max(se_magnitude)
    #new_O_sup = O_sup_lab + np.multiply(direction_O_sup_translation,norm_of_displacement)
    displacement_vector = np.multiply(normalized_direction_O_sup.astype(np.float32), se_max)
    new_O_sup = displacement_vector + O_sup_lab
    #new_O_inf = O_inf_lab + np.multiply(direction_O_sup_translation,norm_of_displacement)
    displacement_vector = np.multiply(normalized_direction_O_inf.astype(np.float32), se_max)
    new_O_inf = displacement_vector + O_inf_lab
    print("|se| max",se_max)
    print("O_sup lab", O_sup_lab)
    print("new_O_sup",new_O_sup)
    print("O_inf lab", O_inf_lab)
    print("new_O_inf",new_O_inf)
    d_conv_pts = np.linalg.norm(new_O_sup-new_O_inf)
    print("convergence points distance",d_conv_pts)
    return new_O_sup, new_O_inf,d_conv_pts

