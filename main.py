# =================================================================
# PROJECT: Flood Prediction Model in Veracruz 
# STUDENT ID: 190380
# =================================================================

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Importing custom modules from the src directory
try:
    from src.acquisition import normalize_gsd
    from src.depth_map import build_unet
    from src.simulation import apply_flood_mask
    print("[INFO] Modules imported successfully.")
except ImportError as e:
    print(f"[ERROR] Could not import modules: {e}")

def run_prediction_pipeline(image_path, water_level=1.8):
    """
    Orchestrates the computer vision pipeline for flood detection.
    """
    print(f"--- Processing: {image_path} ---")

    # 1. Image Acquisition & Preprocessing
    # Loads the satellite image and applies Ground Sample Distance (GSD) normalization
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print("[ERROR] Image not found.")
        return

    # Normalizing to 0.5m per pixel scale
    img_norm = normalize_gsd(raw_img, original_res=0.3, target_gsd=0.5)

    # 2. Topography & Soil Segmentation (AI Model)
    # Simulated depth and segmentation for the preliminary results
    # In a production environment, this calls the trained U-Net model
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    _, soil_mask = cv2.threshold(gray, 120, 1, cv2.THRESH_BINARY) 
    
    # 3. Dynamic Flood Simulation
    # Generates a bitmask and heatmap based on relative elevation
    flood_heatmap = apply_flood_mask(img_norm, soil_mask, water_level)

    # 4. Result Visualization
    print(f"[SUCCESS] Pipeline completed for water level: {water_level}m")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title("Normalized Satellite Image")
    plt.imshow(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title(f"Flood Risk Heatmap ({water_level}m)")
    plt.imshow(cv2.cvtColor(flood_heatmap, cv2.COLOR_BGR2RGB))
    
    plt.show()

if __name__ == "__main__":
    # Path to the sample image in your data folder
    SAMPLE_IMAGE = "data/veracruz_sample.png"
    
    # Execute the pipeline
    run_prediction_pipeline(SAMPLE_IMAGE, water_level=1.8)
