import sys
import os

from src.data_loader import load_mnist_data, normalize_images, reshape_for_cnn, create_train_val_split
from src.data_augmentation import augment_dataset
from src.train import compile_model, train_model
from src.evaluate import plot_training_curves, evaluate_model, analyze_misclassifications
from models.simple_nn import create_simple_nn

import tensorflow as tf

def main():
    print("=== Phase 2: Simple NN Implementation ===")
    
    # 1. Load Data
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_mnist_data()
    
    # 2. Preprocess Data
    X_train_norm = normalize_images(X_train_raw)
    X_test_norm = normalize_images(X_test_raw)
    
    X_train_cnn = reshape_for_cnn(X_train_norm)
    X_test_cnn = reshape_for_cnn(X_test_norm)
    
    X_t, X_v, y_t, y_v = create_train_val_split(X_train_cnn, y_train_raw)
    
    # --- Experiment 1: Original Data ---
    print("\n--- Training on Original Data ---")
    model_orig = create_simple_nn()
    model_orig = compile_model(model_orig)
    
    # Train
    train_model(
        model=model_orig,
        model_name="simple_nn_original",
        X_train=X_t, y_train=y_t,
        X_val=X_v, y_val=y_v,
        epochs=15, # Use 15 for baseline NN to avoid extreme wait
        batch_size=128
    )
    
    # Evaluate
    print("\nEvaluating Original Model...")
    y_pred_orig = evaluate_model(model_orig, X_test_cnn, y_test_raw, "simple_nn_original")
    plot_training_curves("results/logs/simple_nn_original_training.log", "simple_nn_original")
    analyze_misclassifications(X_test_cnn, y_test_raw, y_pred_orig, "simple_nn_original")
    
    # --- Experiment 2: Augmented Data ---
    print("\n--- Training on Augmented Data ---")
    # For speed of the script, we only augment a subset or use factor=1
    print("Generating augmented dataset... This might take a minute.")
    # For NN, let's just augment the original training set with factor 1
    # We apply augment_dataset to X_t, y_t
    X_t_aug, y_t_aug = augment_dataset(X_t, y_t, augmentation_factor=1)
    
    print(f"Original training size: {X_t.shape[0]}")
    print(f"Augmented training size: {X_t_aug.shape[0]}")
    
    model_aug = create_simple_nn()
    model_aug = compile_model(model_aug)
    
    # Train
    train_model(
        model=model_aug,
        model_name="simple_nn_augmented",
        X_train=X_t_aug, y_train=y_t_aug,
        X_val=X_v, y_val=y_v,
        epochs=15,
        batch_size=128
    )
    
    # Evaluate
    print("\nEvaluating Augmented Model...")
    y_pred_aug = evaluate_model(model_aug, X_test_cnn, y_test_raw, "simple_nn_augmented")
    plot_training_curves("results/logs/simple_nn_augmented_training.log", "simple_nn_augmented")
    analyze_misclassifications(X_test_cnn, y_test_raw, y_pred_aug, "simple_nn_augmented")
    
    print("\n=== Phase 2 Complete ===")

if __name__ == '__main__':
    main()
