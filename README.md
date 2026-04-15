# MSci Thesis: Machine Learning for Gravitational Wave Detection

## Overview

This repository contains the code and results for my MSci thesis on developing machine learning models for gravitational wave (GW) detection. The project focuses on training convolutional neural networks (ConvNeXt-based) to classify GW signals from noise and glitches in data from the LIGO detectors (H1 and L1), incorporating time-of-day features to account for diurnal variations.

The main contributions include:
- Simulated GW data generation pipeline using PyCBC
- Novel model architectures with dual-detector inputs and time embeddings
- Ablation studies evaluating the impact of different components
- Comprehensive analysis on both simulated and real GW data

## Key Features

- **Data Generation**: Custom pipeline for creating realistic GW datasets with signals, noise, and glitches
- **Model Architectures**: ConvNeXt encoders with time-aware fusion for improved detection
- **Ablation Studies**: Systematic evaluation of model components (single/dual detectors, time features, pretraining)
- **Performance Analysis**: ROC curves, precision-recall metrics, and attention analysis

## Project Structure

- `Training_Data_Generation/`: Scripts for GW data simulation and preprocessing
  - `Simulation.py`: GW signal and noise generation
  - `Processing.py`: Data preprocessing and dataset creation
  - `Sampling.py`: Sampling strategies for balanced datasets
- `Final_Model/`: Main training script and results for the best-performing model
- `Ablation_*/`: Training scripts for ablation experiments
  - Ablation 1: Single detector with time
  - Ablation 2: Dual detector without time
  - Ablation 3: Dual detector with time and auxiliary head
  - Ablation 4: Single detector without time
  - Ablation 5: No pretrained backbone
- `Early_*/`: Initial model experiments
- `Literature_Inspired_Model/`: Baseline model from literature
- `Result_Analysis_Code/`: Analysis and visualisation scripts
  - `Sim_Data_Analysis.py`: Performance on simulated data
  - `Real_Data_Analysis.py`: Analysis on real LIGO data
  - `compare_.py`: Model comparison utilities
- `*/training_figures*/`: Output directories containing training histories, metrics, and plots

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install pycbc gwpy
pip install numpy pandas matplotlib seaborn scikit-learn
pip install h5py tqdm
```

For PyCBC installation on HPC systems, you may need:
```bash
conda install -c conda-forge pycbc
```

### Setup

1. Clone this repository
2. Ensure data paths are correctly set in the scripts (adjust `PROJECT_ROOT` if needed)
3. For data generation, ensure sufficient storage space (datasets can be large)

### Important Notes

- **Data Availability**: The training data, preprocessed datasets, and some necessary CSV files (e.g., metadata, predictions) are not included in this repository due to size constraints. You will need to regenerate all data from scratch using the provided scripts.
- **Model Files**: Some model file names may have been changed for clarity, but this should not break functionality.
- **Running from Scratch**: To reproduce results, follow the usage steps in order: data generation → training → analysis.

## Usage

### 1. Data Generation

Generate simulated GW data:

```bash
cd Training_Data_Generation
python Simulation.py  # Generate raw waveforms
python Processing.py  # Preprocess and create datasets
```

### 2. Model Training

Train the final model:

```bash
cd Final_Model
python Training.py
```

Adjust hyperparameters in the script as needed. Training outputs will be saved to `training_figures*/` directories.

### 3. Ablation Studies

Run specific ablation experiments by navigating to the corresponding `Ablation_*/` directory and executing `Training.py`.

### 4. Analysis

Evaluate model performance:

```bash
cd Result_Analysis_Code
python Sim_Data_Analysis.py  # Simulated data analysis
python Real_Data_Analysis.py  # Real data analysis
```

## Results

The final model achieves:
- Total binary detection accuracy: ~83%
- ROC AUC: ~88%
- Precision: ~83%, Recall: ~61%

Detailed metrics and plots are available in the `training_figures*/` directories for each experiment.

## Data

- Simulated data generated using PyCBC with realistic noise PSDs
- Includes GW signals from various merger families and glitch types
- Time features based on GPS time converted to cyclic representations

## Model Architecture

- **Encoder**: ConvNeXt-Tiny adapted for 2-channel input (H1/L1 strain data)
- **Time Network**: MLP for processing cyclic time features
- **Fusion**: Concatenation with layer normalization
- **Heads**: Binary classification for GW detection, optional 3-class for signal types


## Author

Saskia Knight  
MSci Physics, Imperial College London
Date: 2026

## Acknowledgments

- LIGO Scientific Collaboration for open data access
- PyCBC and GWpy libraries for GW data handling
- PyTorch ecosystem for deep learning framework
- ConvNeXt model: Liu, Zhuang, et al. "A ConvNet for the 2020s." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
