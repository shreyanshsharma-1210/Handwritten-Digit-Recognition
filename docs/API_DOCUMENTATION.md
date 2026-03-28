# API Documentation: Core Modules

This document provides function-level reference for all modules in the `src/` and `models/` directories.

## 1. Data Loader (`src/data_loader.py`)

### `load_mnist_data()`
- **Purpose**: Load the MNIST dataset.
- **Returns**: Tuple `(X_train, y_train), (X_test, y_test)`.

### `normalize_images(images)`
- **Purpose**: Scales pixel values from [0, 255] to [0, 1].
- **Args**: `numpy.ndarray`.
- **Returns**: `numpy.ndarray` with type `float32`.

### `reshape_for_cnn(images)`
- **Purpose**: Adds channel dimension (N, 28, 28) -> (N, 28, 28, 1).
- **Args**: `numpy.ndarray`.
- **Returns**: `numpy.ndarray`.

---

## 2. Data Augmentation (`src/data_augmentation.py`)

### `augment_dataset(images, labels, factor=1)`
- **Purpose**: Main entry point for augmenting a training set.
- **Args**: Images, labels, and the number of augmented samples to generate per original image.

### `rotate_image(image, angle)`
- **Purpose**: Rotates image +/- 15 degrees.

### `elastic_deformation(image)`
- **Purpose**: Non-rigid displacement field deformation.

---

## 3. Training Logic (`src/train.py`)

### `compile_model(model, lr=0.001)`
- **Purpose**: Compiles with Adam and Sparse Categorical Crossentropy.

### `train_model(model, X_train, y_train, X_val, y_val, ...)`
- **Purpose**: Executes training with standard callbacks (Checkpointing, EarlyStopping).

---

## 4. Evaluation (`src/evaluate.py`)

### `evaluate_model(model, X_test, y_test, model_name)`
- **Purpose**: Calculates accuracy and saves Confusion Matrix/Classification Report.

### `analyze_misclassifications(X_test, y_test, y_pred, model_name)`
- **Purpose**: Visualizes images where the model was incorrect.

---

## 5. Explainable AI (`src/explainability.py`)

### `grad_cam(model, image, layer_name=None)`
- **Purpose**: Generates a 2D focus heatmap for the last convolution layer.

### `saliency_map(model, image)`
- **Purpose**: Generates a gradient-based attribution map at the pixel level.

---

## 6. Model Comparison (`src/compare_models.py`)

### `get_model_metrics(model_path, X_test, y_test)`
- **Purpose**: Benchmarks accuracy vs. model size vs. inference latency.
