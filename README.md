# 190380_Project Flood prediction model in Veracruz through the detection of urban topography
This repository contains the development of a topographic detection model based on satellite images for the simulation of flooding in urban areas of Veracruz, Mexico.

## 1. Problem Description
Veracruz is affected by flooding on its main roads due to urban problems, but primarily due to its coastal plain topography, which hinders the rapid drainage of rainwater. This project proposes the analysis of key areas to obtain their characteristics using computer vision, extracting elevation and soil classification information from publicly available optical satellite imagery. The objective is to classify these areas by risk based on their topography and urbanization conditions.

## 2. Solution Statement
The solution integrates a processing pipeline that normalizes Google Earth images, segments the urban fabric using a U-Net network, and estimates relative depth to generate a synthetic Digital Elevation Model (DEM). Based on this, a Bitmask logic is applied to simulate water accumulation according to critical flood levels.

## 3 Image Analysis Results
| Processing Phase | Technical Description | Observation Results |
| :--- | :--- | :--- |
| **GSD Normalization** | Resolution adjustment to 0.5m/px. | Allows distinguishing street widths in port neighborhoods. |
| **Semantic Segmentation** | Classification of impermeable soils. | Precise identification of concrete and asphalt slabs. |
| **Simulation (1.8m)** | Application of flood threshold. | Detection of stagnant areas in low-lying areas. |
![Preliminary Analysis Results](data/results_veracruz.png)


---

## 4. Attribution and AI Usage Declaration

### AI Usage Declaration
In compliance with the activity instructions, it is declared that the **Gemini (Google)** language model was used as a research and coding assistant for:
* **Logical structuring** of the data pipeline and file architecture.
* **Optimization and debugging** of Computer Vision scripts (specifically for the GSD normalization logic and bitmask operations).
* **Technical writing** of the documentation and this README file.

### Code Citations and Attributions
* **U-Net Architecture:** The semantic segmentation network structure is based on the original model by *Ronneberger et al.*, adapted using the standard implementation available in the official **TensorFlow/Keras** documentation.
* **Image Processing:** Spatial transformation, normalization, and heatmap functions utilize the open-source **OpenCV (Open Source Computer Vision Library)**.
* **Satellite Data:** Imagery used for preliminary testing was obtained via **Google Earth**, respecting the terms of use for educational and research purposes.
