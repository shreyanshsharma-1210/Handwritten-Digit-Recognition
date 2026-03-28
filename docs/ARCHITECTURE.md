# Project Architecture: Handwritten Digit Recognition

This document provides a high-level overview of the technical architecture, data processing pipeline, and model designs implemented in this project.

## 1. System Overview
The system is built as a modular machine learning pipeline using TensorFlow/Keras for modeling and Streamlit for the user interface.

### Project Structure
- `src/`: Core logic for data loading, augmentation, training, and evaluation.
- `models/`: Model architecture definitions.
- `web_app/`: Streamlit application and utilities.
- `results/`: Saved training logs, evaluation reports, and figures.
- `notebooks/`: Exploratory analysis and model comparisons.

## 2. Data Processing Pipeline
### Data Source
We use the standard **MNIST** database (70,000 samples of 28x28 grayscale handwritten digits).

### Preprocessing
1. **Normalization**: Pixel values are scaled from `[0, 255]` to `[0, 1]` to aid gradient descent.
2. **Reshaping**: Images are reshaped to `(28, 28, 1)` to provide the channel dimension required by Convolutional layers.
3. **Splitting**: We perform a stratified split to ensure equal class representation in training (80%) and validation (20%) sets.

### Data Augmentation
To improve model robustness and generalize to real-world scans/sketches, we implement:
- **Rotations**: ± 15°
- **Translations (Shifts)**: ± 2 pixels
- **Zooming**: 0.9x to 1.1x
- **Elastic Deformations**: Simulating non-rigid distortions typical in handwriting.

## 3. Model Architectures
We implement three distinct architectures to explore the accuracy vs. complexity trade-off:

### A. Simple Dense NN
- **Architecture**: `Flatten` -> `Dense(512, ReLU)` -> `Dropout(0.2)` -> `Dense(256, ReLU)` -> `Dropout(0.2)` -> `Dense(10, Softmax)`
- **Purpose**: Baseline performance for non-spatial feature extraction.

### B. Basic CNN (LeNet-5 Style)
- **Architecture**: Two `(Conv2D -> MaxPool2D)` blocks followed by a `Dense` classifier.
- **Purpose**: A standard spatial-aware baseline.

### C. Advanced CNN
- **Architecture**: Deep convolutional stacks with `BatchNormalization` after every layer and `GlobalAveragePooling2D` before the final `Dense` layer.
- **Purpose**: Production-grade architecture with high generalization capability and small parameter footprint (~130k params).

## 4. Explainability (XAI)
To make the model "glass-box," we implement:
- **Grad-CAM**: Gradient-weighted Class Activation Mapping to show which regions of a digit (e.g., the loop of a '9' or the cross of a '7') influenced the prediction.
- **Saliency Maps**: Individual pixel-level attribution to highlight the skeleton of the digit.

## 5. Deployment
The final models are deployed via **Streamlit**, allowing for:
- Live digit drawing on a canvas.
- Real-time prediction with confidence scoring.
- Side-by-side visualization of Grad-CAM heatmaps.
