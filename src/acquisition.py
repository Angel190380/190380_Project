import cv2
import numpy as np

def normalize_gsd(image, original_res, target_gsd=0.5):
    """
    Ajusta la escala de la imagen para que cada píxel represente 
    una distancia métrica real (Ground Sample Distance).
    """
    # Calculamos el factor de escala basado en la resolución original de la captura
    scale_factor = original_res / target_gsd
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    
    # Redimensionamos usando interpolación de área para preservar detalles
    img_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return img_resized

def load_satellite_pair(nadir_path, oblique_path):
    """Carga par estéreo para análisis topográfico."""
    img_n = cv2.imread(nadir_path)
    img_o = cv2.imread(oblique_path)
    return img_n, img_o
