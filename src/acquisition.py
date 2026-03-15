import cv2
import numpy as np

def normalize_gsd(image, original_res, target_gsd=0.5):
    """
    Adjusts the image scale so that each pixel represents 
    a real metric distance (Ground Sample Distance).
    """
    # Calculate the scale factor based on the original capture resolution
    scale_factor = original_res / target_gsd
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    
    # Resize using area interpolation to preserve details
    resized_img = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_img

def load_satellite_pair(nadir_path, oblique_path):
    """Loads a stereo pair for topographic analysis."""
    nadir_img = cv2.imread(nadir_path)
    oblique_img = cv2.imread(oblique_path)
    return nadir_img, oblique_img
