import sys
import os

from src.data_loader import load_mnist_data, normalize_images, reshape_for_cnn, create_train_val_split
from src.data_augmentation import augment_dataset
from src.train import compile_model, train_model
from src.evaluate import plot_training_curves, evaluate_model, analyze_misclassifications

from models.cnn_basic import create_basic_cnn
from models.cnn_advanced import create_advanced_cnn

import tensorflow as tf

def main():
    print("=== Phase 3: CNN Implementation ===")
    
    # 1. Load Data
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_mnist_data()
    
    # 2. Preprocess Data
    X_train_norm = normalize_images(X_train_raw)
    X_test_norm = normalize_images(X_test_raw)
    
    X_train_cnn = reshape_for_cnn(X_train_norm)
    X_test_cnn = reshape_for_cnn(X_test_norm)
    
    X_t, X_v, y_t, y_v = create_train_val_split(X_train_cnn, y_train_raw)
    
    # Generate augmented dataset (Factor 1 since CNN training takes longer)
    print("\nGenerating augmented dataset... This might take a minute.")
    X_t_aug, y_t_aug = augment_dataset(X_t, y_t, augmentation_factor=1)
    
    print(f"Original training size: {X_t.shape[0]}")
    print(f"Augmented training size: {X_t_aug.shape[0]}")
    
    # --- Experiment 1: Basic CNN (LeNet-5 Style) ---
    print("\n--- Training Basic CNN (LeNet-5 Style) on Original Data ---")
    model_basic = create_basic_cnn()
    model_basic = compile_model(model_basic, learning_rate=0.001)
    
    # Train
    train_model(
        model=model_basic,
        model_name="cnn_basic",
        X_train=X_t, y_train=y_t,
        X_val=X_v, y_val=y_v,
        epochs=15,
        batch_size=128
    )
    
    # Evaluate
    print("\nEvaluating Basic CNN Model...")
    y_pred_basic = evaluate_model(model_basic, X_test_cnn, y_test_raw, "cnn_basic")
    plot_training_curves("results/logs/cnn_basic_training.log", "cnn_basic")
    analyze_misclassifications(X_test_cnn, y_test_raw, y_pred_basic, "cnn_basic")
    
    # --- Experiment 2: Advanced CNN ---
    print("\n--- Training Advanced CNN on Augmented Data ---")
    
    model_adv = create_advanced_cnn()
    model_adv = compile_model(model_adv, learning_rate=0.001)
    
    # Train
    train_model(
        model=model_adv,
        model_name="cnn_advanced_augmented",
        X_train=X_t_aug, y_train=y_t_aug,
        X_val=X_v, y_val=y_v,
        epochs=15, # Use 15 for time constraint, EarlyStopping might kick in
        batch_size=128
    )
    
    # Evaluate
    print("\nEvaluating Advanced CNN Model...")
    y_pred_adv = evaluate_model(model_adv, X_test_cnn, y_test_raw, "cnn_advanced_augmented")
    plot_training_curves("results/logs/cnn_advanced_augmented_training.log", "cnn_advanced_augmented")
    analyze_misclassifications(X_test_cnn, y_test_raw, y_pred_adv, "cnn_advanced_augmented")
    
    print("\n=== Phase 3 Complete ===")

if __name__ == '__main__':
    main()
