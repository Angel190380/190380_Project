# =================================================================
# PROJECT: Flood Prediction Model in Veracruz 
# STUDENT ID: 190380
# =================================================================

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# --- 1. U-NET ARCHITECTURE SPECIFICATION (Validated against State-Dict) ---
class build_unet(nn.Module):
    def __init__(self):
        super(build_unet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.pool = nn.MaxPool2d(2)
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256) 
        self.bottleneck = conv_block(256, 512) 

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1) 

    def forward(self, x):
        # Encoder Path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Latent Space Representation
        b = self.bottleneck(self.pool(e3))

        # Decoder Path with Skip Connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

# --- 2. DYNAMIC ASSET LOCALIZATION & MODEL RECONSTRUCTION ---
print("🔍 Localizing model weights...")
base_path = '/content/190380_Project'
output_pth = None

# Recursive search for best_model.pth
for root, dirs, files in os.walk(base_path):
    if 'best_model.pth' in files:
        output_pth = os.path.join(root, 'best_model.pth')
        break

# Attempt reconstruction from multi-part archives if .pth is missing
if not output_pth:
    print("🔄 .pth file not found. Initializing reconstruction from multi-part RAR archives...")
    rar_path = None
    for root, dirs, files in os.walk(base_path):
        if 'best_model.part1.rar' in files:
            rar_path = os.path.join(root, 'best_model.part1.rar')
            dest_folder = root
            break
    
    if rar_path:
        !sudo apt-get install unrar -y > /dev/null
        !unrar x "{rar_path}" "{dest_folder}/" -y > /dev/null
        output_pth = os.path.join(dest_folder, 'best_model.pth')
        print(f"✅ Model successfully reconstructed at: {output_pth}")
    else:
        print("❌ CRITICAL ERROR: best_model.pth and RAR archives are missing.")
        sys.exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Model Initialization and Weight Loading
    model = build_unet().to(device)
    model.load_state_dict(torch.load(output_pth, map_location=device))
    model.eval()
    print("✅ Model Architecture and Inference Pipeline synchronized successfully.")

    # --- 3. PRE-PROCESSING & STATISTICAL INFERENCE ---
    # Input normalization and resizing for U-Net compatibility
    input_final = cv2.resize(processed_data, (512, 512), interpolation=cv2.INTER_AREA)
    tensor_input = torch.from_numpy(input_final).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(tensor_input)
        # Activation via Sigmoid for probabilistic risk assessment
        prob_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        # Binary Classification Threshold (0.4)
        flood_mask = (prob_mask > 0.4).astype(np.uint8) 

    # --- 4. DATA VISUALIZATION & GEOSPATIAL ANALYSIS ---
    plt.figure(figsize=(16, 8))
    
    # Subplot 1: Grayscale Input (Contextual representation)
    plt.subplot(1, 2, 1)
    plt.imshow(input_final, cmap='gray')
    plt.title("Original Input (Veracruz Port Area)")
    plt.axis('off')

    # Subplot 2: Output Synthesis (Topography + U-Net Mask)
    plt.subplot(1, 2, 2)
    plt.imshow(input_final, cmap='terrain')
    
    # Masking zero-risk pixels for overlay clarity
    risk_zone = np.ma.masked_where(flood_mask == 0, flood_mask)
    
    # Visualizing High-Risk Zones via Red Alpha Channel Overlay
    plt.imshow(risk_zone, cmap='Reds', alpha=0.7) 
    plt.title("Flood Risk Prediction Map - U-Net Segmentation (E5)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Technical Metadata Summary
    print(f"✅ Total Risk-Positive Pixels Detected: {np.sum(flood_mask)}")
    print("✅ Inference complete. Visualization aligned with doctoral reporting standards.")

except Exception as e:
    print(f"❌ EXECUTION ERROR: {e}")

