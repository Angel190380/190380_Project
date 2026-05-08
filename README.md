# 190380_Project Flood prediction model in Veracruz through the detection of urban topography
This repository contains the development of a topographic detection model based on satellite images for the simulation of flooding in urban areas of Veracruz, Mexico. The project implements an advanced flood risk prediction model designed specifically for the urban landscape of the Port of Veracruz. By leveraging state-of-the-art Computer Vision techniques and multi-layered geospatial data processing, the system maps urban topography with high precision. Unlike traditional models that rely solely on historical rainfall data, this solution analyzes the physical characteristics of the terrain to simulate how water moves through the city's unique infrastructure. Our approach integrates digital elevation models (DEM) and satellite imagery to identify structural vulnerabilities, providing a robust framework for disaster prevention and urban planning in high-density coastal environments.

## 1. Problem Description
The Port of Veracruz faces a critical challenge: it is historically vulnerable to recurrent flooding that threatens both public safety and economic stability. While most prediction systems focus on real-time meteorological variables, they often overlook the fundamental cause of water pooling—the structural analysis of the terrain.

This project addresses that gap by proposing a solution that prioritizes urban topography as the primary predictor of risk. By identifying critical zones of water accumulation, such as natural depressions and areas with poor drainage connectivity, we can pinpoint exactly where floods will occur before the first drop of rain falls. This shift from reactive monitoring to proactive structural modeling allows city officials to implement targeted engineering interventions, optimize emergency response routes, and ultimately build a more resilient urban future against the increasing frequency of extreme weather events.

## 2. Solution Statement
The core of this solution is a robust Convolutional Neural Network (U-Net) specifically architected for the semantic segmentation of flood-risk areas. By processing complex multichannel datasets, the model can effectively distinguish between safe zones and critical accumulation points with high spatial accuracy.

- Data Sources: The model leverages the Continuo de Elevaciones Mexicano (CEM 3.0) provided by INEGI. This high-resolution topographic data is essential for structural validation, allowing the network to understand the physical contours and slopes of the terrain that dictate water flow.

- Architecture Implementation: We implemented a customized U-Net architecture using PyTorch. This framework was chosen for its flexibility in handling deep learning layers, enabling efficient feature extraction and precise localization through its symmetrical encoder-decoder structure.

- Optimization Metric: To ensure high performance, we utilized Dice Loss as our primary loss function. This is crucial for handling the inherent class imbalance in flood segmentation, where at-risk pixels are often significantly outnumbered by non-risk areas.

- Multichannel Processing: The system integrates a multichannel dataset that combines raw elevation data with derived topographical indices. This multi-layered approach allows the neural network to identify subtle patterns in urban drainage that a single-channel analysis would miss.

- Structural Predictive Modeling: Beyond simple visual recognition, the solution focuses on structural terrain analysis. By training on precise geomorphological data, the model can predict potential flood zones based on the physical environment's capacity to retain water, rather than relying solely on temporal weather data.

## 3 Repository Structure
The project is organized according to development stages E1 through E5, ensuring a modular and traceable workflow:

/src: Contains core Python processing scripts, including modules for data cleaning, geospatial preprocessing, and automated patch generation for model input.

/models: Includes the formal U-Net architecture definitions and the pre-trained weights (checkpoints) required for inference and further fine-tuning.

/notebooks: A collection of Jupyter Notebooks providing a step-by-step walkthrough of each stage (E1-E5). These are pre-configured for seamless execution within Google Colab.

/data: Contains processed data samples derived from the CEM (Continuo de Elevaciones Mexicano) or detailed instructions on how to download the full dataset from official sources.

## 4. Execution Instructions (Self-contained)
To run this project locally or via Google Colab, follow these steps:

Clone the repository:
Use Git to replicate the project environment on your local machine:
git clone https://github.com/Angel190380/190380_Project.git

Install dependencies:
Ensure your environment meets the requirements by installing the necessary packages (including specialized libraries like Rasterio, GeoPandas, and PyTorch):
pip install -r requirements.txt

Run the project:
Navigate to the /notebooks directory and open the main notebook to begin the end-to-end prediction process.



## 5. Attribution and AI Usage Declaration

AI Usage Declaration
In compliance with the activity instructions, it is declared that the Gemini (Google) language model was used as a research and coding assistant for:

- Logical structuring of the multichannel data pipeline and file architecture.

- Optimization and debugging of Computer Vision scripts, specifically for geospatial preprocessing and PyTorch-based U-Net implementation.

- Technical writing of the documentation.

Code Citations and Attributions
- U-Net Architecture: The semantic segmentation network structure is based on the original model by Ronneberger et al., implemented and adapted using the PyTorch framework.

- Geospatial Processing: Data manipulation, spatial transformations, and raster operations utilize the open-source libraries Rasterio, GeoPandas, and OpenCV.

- Topographic Data: Elevation data used for structural validation was obtained via the Continuo de Elevaciones Mexicano (CEM 3.0) from INEGI, ensuring the model remains grounded in official geophysical terrain data.

- Development Environment: The workflow was developed and tested using Google Colab, utilizing its cloud-based GPU resources for model training and validation.
