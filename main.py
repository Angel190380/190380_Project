# =================================================================
# PROJECT: Flood Prediction Model in Veracruz 
# STUDENT ID: 190380
# =================================================================

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
from io import BytesIO

# --- 1. ARQUITECTURA U-NET (E5) ---
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final_conv(d1)

# --- 2. FUNCIONES DE PROCESAMIENTO ---
def preprocess_image(file_path):
    """Lectura versátil para formatos TIFF y capturas PNG/JPG[cite: 67, 69]."""
    if file_path.endswith(('.tif', '.tiff')):
        with rasterio.open(file_path) as src:
            data = src.read(1)
    else:
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = img

    # Normalización crítica para la red neuronal 
    data_norm = data.astype(np.float32)
    data_norm = (data_norm - np.min(data_norm)) / (np.max(data_norm) - np.min(data_norm) + 1e-8)
    return data, data_norm

def run_inference(model, device, normalized_data):
    """Ejecuta la inferencia y genera la máscara de inundación[cite: 176]."""
    input_tensor = cv2.resize(normalized_data, (512, 512), interpolation=cv2.INTER_AREA)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Umbral ajustable (0.4 según demo) [cite: 176]
    flood_mask = (prob_mask > 0.4).astype(np.uint8)
    flood_mask_res = cv2.resize(flood_mask, (normalized_data.shape[1], normalized_data.shape[0]))
    return flood_mask_res

# --- 3. FLUJO PRINCIPAL ---
def main(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Localizar pesos del modelo 
    model_path = "190380_Project/Models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    # Cargar Modelo
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Arquitectura e Inferencia sincronizadas con éxito.")

    # Procesar Imagen
    raw_data, norm_data = preprocess_image(image_path)
    flood_mask = run_inference(model, device, norm_data)

    # Visualización [cite: 176, 180]
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(raw_data, cmap='gray')
    plt.title("Entrada Original (Veracruz)")
    
    plt.subplot(1, 2, 2)
    # Superposición de riesgo en rojo [cite: 176]
    plt.imshow(raw_data, cmap='gray')
    mask_rgb = np.zeros((*flood_mask.shape, 4))
    mask_rgb[flood_mask == 1] = [1, 0, 0, 0.5] # Rojo con 50% transparencia
    plt.imshow(mask_rgb)
    plt.title("Mapa de Riesgo Detectado por U-Net (E5)")
    
    plt.tight_layout()
    plt.show()
    
    print(f"Pixeles con riesgo detectados: {np.sum(flood_mask)}")
    print("Proceso completado.")

if __name__ == "__main__":
    # Cambiar por la ruta de tu imagen de prueba
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Uso: python main.py <ruta_de_imagen>")

