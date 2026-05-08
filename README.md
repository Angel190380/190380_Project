# 190380_Project Flood prediction model in Veracruz through the detection of urban topography
This repository contains the development of a topographic detection model based on satellite images for the simulation of flooding in urban areas of Veracruz, Mexico. The project implements an advanced flood risk prediction model designed specifically for the urban landscape of the Port of Veracruz. By leveraging state-of-the-art Computer Vision techniques and multi-layered geospatial data processing, the system maps urban topography with high precision. Unlike traditional models that rely solely on historical rainfall data, this solution analyzes the physical characteristics of the terrain to simulate how water moves through the city's unique infrastructure. Our approach integrates digital elevation models (DEM) and satellite imagery to identify structural vulnerabilities, providing a robust framework for disaster prevention and urban planning in high-density coastal environments.

## 1. Problem Description
The Port of Veracruz faces a critical challenge: it is historically vulnerable to recurrent flooding that threatens both public safety and economic stability. While most prediction systems focus on real-time meteorological variables, they often overlook the fundamental cause of water pooling—the structural analysis of the terrain.

This project addresses that gap by proposing a solution that prioritizes urban topography as the primary predictor of risk. By identifying critical zones of water accumulation, such as natural depressions and areas with poor drainage connectivity, we can pinpoint exactly where floods will occur before the first drop of rain falls. This shift from reactive monitoring to proactive structural modeling allows city officials to implement targeted engineering interventions, optimize emergency response routes, and ultimately build a more resilient urban future against the increasing frequency of extreme weather events.

## 2. Solution Statement
The core of this solution is a robust Convolutional Neural Network (U-Net) specifically architected for the semantic segmentation of flood-risk areas. By processing complex multichannel datasets, the model can effectively distinguish between safe zones and critical accumulation points with high spatial accuracy.

Data Sources: The model leverages the Continuo de Elevaciones Mexicano (CEM 3.0) provided by INEGI. This high-resolution topographic data is essential for structural validation, allowing the network to understand the physical contours and slopes of the terrain that dictate water flow.

Architecture Implementation: We implemented a customized U-Net architecture using PyTorch. This framework was chosen for its flexibility in handling deep learning layers, enabling efficient feature extraction and precise localization through its symmetrical encoder-decoder structure.

Optimization Metric: To ensure high performance, we utilized Dice Loss as our primary loss function. This is crucial for handling the inherent class imbalance in flood segmentation, where at-risk pixels are often significantly outnumbered by non-risk areas.

Multichannel Processing: The system integrates a multichannel dataset that combines raw elevation data with derived topographical indices. This multi-layered approach allows the neural network to identify subtle patterns in urban drainage that a single-channel analysis would miss.

Structural Predictive Modeling: Beyond simple visual recognition, the solution focuses on structural terrain analysis. By training on precise geomorphological data, the model can predict potential flood zones based on the physical environment's capacity to retain water, rather than relying solely on temporal weather data.

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
