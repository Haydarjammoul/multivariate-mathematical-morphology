# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 13:46:08 2023

@author: hjammoul
The aim of this file is to Synthesize images for testing.
functions are: 
    draw_shape
    generate_color_img
    generate_img_contrast_difference_effect
    generate_img_shape_size_difference_effect
    get_slice
    Construct_disk_SE
"""
import numpy as np
import cv2
import random
import tifffile
#######################################################

def draw_shape(img, x, y, shape_size, shape_type, shape_color):
    """
    Draw a shape on an image.

    Parameters:
        img (numpy.ndarray): The image on which the shape will be drawn.
        x (int): The x-coordinate of the shape's position.
        y (int): The y-coordinate of the shape's position.
        shape_size (int): The size of the shape.
        shape_type (str): The type of shape ('triangle', 'square', or 'circle').
        shape_color (tuple): The RGB color of the shape.

    Returns:
        img (numpy.ndarray) 

    Example:
        # Draw a blue square at position (50, 50) with a size of 30 pixels.
        draw_shape(img, 50, 50, 30, 'square', (0, 0, 255))
    """
    if shape_size % 2 != 0:
        shape_size = shape_size + 1

    if shape_type == 'triangle':
        points = np.array([(x, y), (x + shape_size, y), (x + shape_size // 2, y + shape_size)])
        cv2.fillPoly(img, [points], shape_color)
    elif shape_type == 'square':
        tpoints = np.array([(x, y), (x, y + shape_size), (x + shape_size, y + shape_size), (x + shape_size, y)])
        cv2.fillPoly(img, [tpoints], shape_color)
    elif shape_type == 'circle':
        center = (x , y)
        cv2.circle(img, center, shape_size // 2, shape_color, -1)
        return img

def generate_color_img(height_img=200, num_shapes=10, shape_min_size=40, shape_max_size=40, mode='CSOMP',
                       img_background_color=(0, 0, 0),target_color = (255,255,255)):
    """
    Generate an RGB image with different shapes (triangles, crosses, circles) of different colors.
    The image is used for testing CMOMP and CSOMP.
    The SEs will be drawn in the image.
    Parameters:
        height_img (int): Height and width of the image.
        num_shapes (int): Number of shapes to generate.
        shape_min_size (int): Minimum size of shapes in pixels.
        shape_max_size (int): Maximum size of shapes in pixels.
        mode (str): 'CSOMP' or 'CMOMP' to generate one or two SEs.
        img_background_color (tuple): RGB color for the background.
        target_color (tuple): RGB color for the target shape in CSOMP mode.

    Returns:
        img (numpy.ndarray) in RGB: The generated RGB image.
        se_img (numpy.ndarray) in RGB: The structuring element (SE) image in CSOMP mode.
        se_inf_img (numpy.ndarray) in RGB: The inferior structuring element (SE) image in CMOMP mode.
        se_sup_img (numpy.ndarray) in RGB : The superior structuring element (SE) image in CMOMP mode.
    Example:
        img, se_inf_img, se_sup_img = generate_color_img(height_img=200, num_shapes=10, shape_min_size=40, shape_max_size=40, mode='CSOMP',
                               img_background_color=(0, 0, 0),target_color = (255,255,255))
    """
    # Create a blank RGB image
    img = np.ones((height_img, height_img, 3), dtype=np.uint8) * np.array(img_background_color, dtype=np.uint8)
    # Generate the shapes
    for _ in range(num_shapes):
        # Generate a random shape position and size
        x = np.random.randint(0, height_img - shape_max_size)
        y = np.random.randint(0, height_img - shape_max_size)
        shape_size = np.random.randint(shape_min_size-1, shape_max_size)
        # Generate a random shape type
        shape_type = np.random.choice(['triangle', 'cross', 'circle'])
        # Generate a random RGB color
        shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0,255))
        if shape_size % 2 != 0:
           shape_size = shape_size+1
        # Draw the shape on the image
        if shape_type == 'triangle':
            points = np.array([(x, y), (x + shape_size, y), (x + shape_size // 2, y + shape_size)])
            cv2.fillPoly(img, [points], shape_color)
        elif shape_type == 'cross':
            tpoints = np.array([(x, y), (x , y+ shape_size), (x+ shape_size , y + shape_size // 2)])
            cv2.fillPoly(img, [tpoints], shape_color)
        elif shape_type == 'circle':
            center = (x + shape_size // 2, y + shape_size // 2)
            cv2.circle(img, center, shape_size // 2, shape_color, -1)
    #Put SE target in image
    x = 150
    y = 150
    shape_size = np.random.randint(shape_min_size-1, shape_max_size)
    points = np.array([(x, y), (x + shape_size, y), (x + shape_size // 2, y + shape_size)])
    cv2.fillPoly(img, [points], target_color)
    #In the case of CSOMP SE is this triangle with black background;
    if mode == 'CSOMP':  # Assuming SE is always a triangle
            # Generate a black background for the SE
            se_background_color = (0, 0, 0)
            #se_background_color = tuple(c // 100 for c in target_color)
            se_img = np.ones((shape_size+2, shape_size+2, 3), dtype=np.uint8) * np.array(se_background_color, dtype=np.uint8)
            #se_img = np.full((shape_size+2, shape_size+2, 3), np.nan, dtype=np.float32)
            # Draw the SE triangle on the black background
            se_points = np.array([(1, 1), (1+shape_size, 1), (1+shape_size // 2, 1+shape_size)])
            cv2.fillPoly(se_img, [se_points], target_color)
            return img, se_img
    elif mode == 'CMOMP':
        # Create two triangles, two SEs: se_inf and se_sup
        # se_inf has color: target_color-5
        se_background_color = (0, 0, 0)
        se_inf_color = tuple(c - 20 for c in target_color)
        se_inf_img = np.ones((1+shape_size*2, 1+shape_size*2, 3), dtype=np.uint8) * np.array(se_background_color, dtype=np.uint8)
        se_inf_shape_size = shape_size//2
        x_offset = (se_inf_img.shape[1] - se_inf_shape_size) // 2
        y_offset = (se_inf_img.shape[0] - se_inf_shape_size) // 2
        se_inf_points = np.array([(x_offset, y_offset), (x_offset + se_inf_shape_size, y_offset), (x_offset + se_inf_shape_size // 2, y_offset + se_inf_shape_size)])
        cv2.fillPoly(se_inf_img, [se_inf_points], se_inf_color)
        # se_sup has color: target_color+5
        se_sup_color = tuple(c + 10 for c in target_color)
        se_sup_img = np.ones((1+shape_size*2,1+shape_size*2, 3), dtype=np.uint8) * np.array(se_background_color, dtype=np.uint8)
        se_sup_shape_size = shape_size*2
        x_offset = (se_sup_img.shape[1] - se_sup_shape_size) // 2
        y_offset = (se_sup_img.shape[0] - se_sup_shape_size) // 2
        se_sup_points = np.array([(x_offset, y_offset), (x_offset + se_sup_shape_size, y_offset), (x_offset + se_sup_shape_size // 2, y_offset + se_sup_shape_size)])
        cv2.fillPoly(se_sup_img, [se_sup_points], se_sup_color)
        return img, se_inf_img, se_sup_img


def generate_img_contrast_difference_effect(height_img=50, img_background_color=(0, 0, 0), target_color=(0, 0, 230),  
                                            num_circles_per_line=5, circle_radius=8,noise_std=0):
    """
  Generate an image with circles of varying colors and contrast differences.
  The variationss are manually set inside the definiton in case u might need to change them.(arrays: blue variations, red variations...)
  They're variations for each RGB channel.
  Parameters:
      height_img (int, optional): Height of the generated image, by default 50.
      img_background_color (tuple, optional): RGB tuple representing the background color of the image, by default (0, 0, 0).
      target_color (tuple, optional): RGB tuple representing the color of the first circle, by default (0, 0, 230).
      num_circles_per_line (int, optional): Number of circles per line, by default 5.
      circle_radius (int, optional): Radius of the circles, by default 8.
      noise_std (int, optional): Standard deviation of noise to add to the image, by default 0.

  Returns:
      rgb_img (numpy.ndarray): The generated RGB image with circles of different colors.
      se_inf_img (numpy.ndarray): The generated RGB image of the SE_inf circle with color variations.
      se_sup_img (numpy.ndarray): The generated RGB image of the SE_sup circle with color variations.
  
  Example:
      img_rgb, se_inf_rgb, se_sup_rgb = generate_img_contrast_difference_effect(height_img=150, img_background_color=(0, 0, 0),
                                                                                target_color=(0, 0, 230), num_circles_per_line=5,
                                                                                circle_radius=8, noise_std=0)
  """
    # Create a blank RGB image
    rgb_img = np.ones((height_img, height_img, 3), dtype=np.uint8) * np.array(img_background_color, dtype=np.uint8)
    # Generate the circles

    # Calculate the step size for placing circles uniformly on the same line
    step_x = height_img // (num_circles_per_line + 1)
    step_y = height_img // 4

    # Define fixed variations in color coordinates for each circle
    blue_variations = [
        (0, 0, -10),
        (0, 0, -20),
        (0, 0, -40),
        (0, 0, -80),
        (0, 0, -100)
    ]

    red_variations = [
        (10, 0, 0),
        (20, 0, 0),
        (100, 0, 0),
        (150, 0, 0),
        (200, 0, 0)
    ]

    green_variations = [
        (0, 10, 0),
        (0, 20, 0),
        (0, 100, 0),
        (0, 150, 0),
        (0, 200, 0)
    ]

    # Draw circles with changes in the blue coordinate (first line)
    for i in range(num_circles_per_line):
        x = (i + 1) * step_x
        y = step_y
        circle_color = tuple(c + blue_variations[i][j] for j, c in enumerate(target_color))
        center = (x, y)
        cv2.circle(rgb_img, center, circle_radius, circle_color, -1)

    # Draw circles with changes in the red coordinate (second line)
    for i in range(num_circles_per_line):
        x = (i + 1) * step_x
        y = 2 * step_y
        circle_color = tuple(c + red_variations[i][j] for j, c in enumerate(target_color))
        center = (x, y)
        cv2.circle(rgb_img, center, circle_radius, circle_color, -1)

    # Draw circles with changes in the green coordinate (third line)
    for i in range(num_circles_per_line):
        x = (i + 1) * step_x
        y = 3 * step_y
        circle_color = tuple(c + green_variations[i][j] for j, c in enumerate(target_color))
        center = (x, y)
        cv2.circle(rgb_img, center, circle_radius, circle_color, -1)

    # Add noise to the rgb_img
    noise = np.random.normal(loc=0, scale=noise_std, size=rgb_img.shape).astype(np.uint8)
    rgb_img = np.clip(rgb_img + noise, 0, 255)

    # Define SE_inf: circle with same shape and color target_color + inf_variations
    inf_variations = (-15, -15, -15)
    se_inf_color = tuple(c + inf_variations[j] for j, c in enumerate(target_color))
    se_inf_img = np.ones((circle_radius*2+3, circle_radius*2+3, 3), dtype=np.uint8) * np.array((0,0,0), dtype=np.uint8)
    center = (circle_radius+1, circle_radius+1)
    cv2.circle(se_inf_img, center, circle_radius, se_inf_color, -1)

    # Define SE_sup: circle with same shape and color target_color + sup_variations
    sup_variations = (15, 15, 15)
    se_sup_color = tuple(c + sup_variations[j] for j, c in enumerate(target_color))
    se_sup_img = np.ones((circle_radius*2+3, circle_radius*2+3, 3), dtype=np.uint8) * np.array((0,0,0), dtype=np.uint8)
    center = (circle_radius+1, circle_radius+1)
    cv2.circle(se_sup_img, center, circle_radius, se_sup_color, -1)
    return rgb_img, se_inf_img, se_sup_img

def generate_img_shape_size_difference_effect(height_img=50, img_background_color=(0, 0, 0), target_color=(0, 0, 230),
                                             num_shapes_per_line=5, initial_shape_size=6,target_radius = 12,noise_std=0):
    """
    Generate an image with different shapes and their size variations.

    Parameters:
        height_img (int, optional): Height of the generated image, by default 50.
        img_background_color (tuple, optional): RGB tuple representing the background color of the image, by default (0, 0, 0).
        target_color (tuple, optional): RGB tuple representing the color of the shapes, by default (0, 0, 230).
        num_shapes_per_line (int, optional): Number of shapes per line, by default 5.
        initial_shape_size (int, optional): Initial size of the shapes, by default 6.
        target_radius (int, optional): Radius of the target shape, by default 12.
        noise_std (int, optional): Standard deviation of noise to add to the image, by default 0.

    Returns:
        rgb_img (numpy.ndarray): The generated RGB image with different shapes and their size variations.
        se_inf_img (numpy.ndarray): The generated RGB image of the SE_inf shape with size variations.
        se_sup_img (numpy.ndarray): The generated RGB image of the SE_sup shape with size variations.
    
    Example:
        img_rgb, se_inf_rgb, se_sup_rgb = generate_img_shape_size_difference_effect(height_img=50, img_background_color=(0, 0, 0),
                                                                                    target_color=(0, 0, 230), num_shapes_per_line=5,
                                                                                    initial_shape_size=6, target_radius=12, noise_std=0)
    """
    # Create a blank RGB image
    rgb_img = np.ones((height_img, height_img, 3), dtype=np.uint8) * np.array(img_background_color, dtype=np.uint8)
    
    # Calculate the step size for placing shapes uniformly on the same line
    step_x = height_img // (num_shapes_per_line + 1)
    step_y = height_img // 4
    
    # Define fixed variations in shape size for each line
    triangle_size_variations = [2, 4, 6, 8, 10]
    square_size_variations = [2, 4, 6, 8, 10]
    circle_size_variations = [2, 4, 6, 8, 10]
    
    # Draw triangles with size variations (first line)
    for i in range(num_shapes_per_line):
        x = (i + 1) * step_x
        y = step_y
        shape_size = initial_shape_size + triangle_size_variations[i]
        draw_shape(rgb_img, x, y, shape_size, 'triangle', target_color)
    
    # Draw squares with size variations (second line)
    for i in range(num_shapes_per_line):
        x = (i + 1) * step_x
        y = 2 * step_y
        shape_size = initial_shape_size + square_size_variations[i]
        draw_shape(rgb_img, x, y, shape_size, 'square', target_color)
    
    # Draw circles with size variations (third line)
    for i in range(num_shapes_per_line):
        x = (i + 1) * step_x
        y = 3 * step_y
        shape_size = initial_shape_size + circle_size_variations[i]
        draw_shape(rgb_img, x, y, shape_size, 'circle', target_color)
    
    # Define SE_inf: shape with same color and size target_color + inf_variations
    inf_variations = -1
    se_inf_img = np.ones((target_radius * 2 + 3, target_radius * 2 + 3, 3), dtype=np.uint8) * np.array((0, 0, 0), dtype=np.uint8)
    draw_shape(se_inf_img, target_radius + 1, target_radius + 1, target_radius + inf_variations, 'circle', target_color)
    
    # Define SE_sup: shape with same color and size target_color + sup_variations
    sup_variations = 1
    se_sup_img = np.ones((target_radius * 2 + 3, target_radius * 2 + 3, 3), dtype=np.uint8) * np.array((0, 0, 0), dtype=np.uint8)
    draw_shape(se_sup_img, target_radius + 1, target_radius + 1, target_radius + sup_variations, 'circle', target_color)
    
    # Add noise to the rgb_img
    noise = np.random.normal(loc=0, scale=noise_std, size=rgb_img.shape).astype(np.uint8)
    rgb_img = np.clip(rgb_img + noise, 0, 255)
    
    print("Se sup radius, Se_inf radius:",target_radius + sup_variations,target_radius + inf_variations )
    print("smallest radius", initial_shape_size+2)
    print("biggest radius", initial_shape_size+10)
    return rgb_img, se_inf_img, se_sup_img


def get_slice(image_file, z):
    """
    Get a slice from a Mice heart image in TIFF format.

    Parameters:
        image_file (str): Path to the TIFF image file.
        z (int): Slice number to retrieve.

    Returns:
        slice (numpy.ndarray): The specified image slice as a 2D numpy array.
        
    Example:
        slice = get_slice(path_to_img , 150)
    """
    # Read the TIFF image
    image = tifffile.imread(image_file)
    # Get the dimensions of the image and the slice
    ##Z, ch, y,x
    #image_height, _, _, _ = image.shape
    # Take the required slice
    slice_image_rgb = image[z,:, :, :] ###ype is unint16
    return slice_image_rgb.astype(np.int32)

def construct_disk_se(Value, size_of_SE, shape_size, background=(0,0,0)):
    """
    Create a flat structuring element (SE) in the shape of a disk.

    Parameters:
        Value (tuple): CIE Lab coordinates for the disk color.
        size_of_SE (int): Size of the square SE (odd value).
        shape_size (int): Size of the disk.
        background (tuple, optional): RGB color for the background, by default (0, 0, 0).

    Returns:
        SE (numpy.ndarray): The generated flat SE.

    Example:
        SE = Construct_disk_SE((L, a, b), 9, 9, (255, 255, 255))
    """

    # Create an array of NaN values
    SE = np.full((shape_size, shape_size, 3), background, dtype=np.float32)
    # Calculate the center of the square
    center = (shape_size // 2, shape_size // 2)
    # Fill the disk region with the specified Value
    cv2.circle(SE, center, size_of_SE, Value, -1)
    return SE