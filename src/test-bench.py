# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:12:35 2023

@author: hjammoul

Experimenting with multivariate morphological filtering and probing parameters: 
    Aim is to detect more neurons in the images
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color


from lib.synthesize import get_slice,construct_disk_SE 
from lib.pretreatment import normalize_without_ref, cra_median_filtering
from lib.rgb_math_morphology_tools import  extend_convergence_pts, cmomp, construct_round_se_flat, opening, closing



