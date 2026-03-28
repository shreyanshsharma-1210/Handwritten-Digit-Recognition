import sys
import os

from src.data_loader import load_mnist_data, normalize_images, reshape_for_cnn, create_train_val_split
from src.data_augmentation import augment_dataset
from src.evaluate import plot_training_curves, evaluate_model, analyze_misclassifications

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def main():
    print("=== Resuming Phase 3: Advanced CNN Implementation ===")
    
    # 1. Load Data
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = load_mnist_data()
    
    # 2. Preprocess Data
    X_train_norm = normalize_images(X_train_raw)
    X_test_norm = normalize_images(X_test_raw)
    
    X_train_cnn = reshape_for_cnn(X_train_norm)
    X_test_cnn = reshape_for_cnn(X_test_norm)
    
    X_t, X_v, y_t, y_v = create_train_val_split(X_train_cnn, y_train_raw)
    
    print("\nGenerating augmented dataset... This might take a minute.")
    X_t_aug, y_t_aug = augment_dataset(X_t, y_t, augmentation_factor=1)
    
    # 3. Load existing model
    model_path = "models/saved_models/cnn_advanced_augmented_best.keras"
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Train
    print("\n--- Resuming Training Advanced CNN on Augmented Data ---")
    
    model_name = "cnn_advanced_augmented"
    
    # Prepare custom callbacks to append to the log instead of overwriting
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(f'models/saved_models/{model_name}_best.keras', save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        CSVLogger(f'results/logs/{model_name}_training.log', append=True)
    ]
    
    model.fit(
        X_t_aug, y_t_aug,
        validation_data=(X_v, y_v),
        epochs=10, # Continue for another 10 epochs max
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # 5. Evaluate
    print("\nEvaluating Advanced CNN Model...")
    y_pred_adv = evaluate_model(model, X_test_cnn, y_test_raw, model_name)
    plot_training_curves(f"results/logs/{model_name}_training.log", model_name)
    analyze_misclassifications(X_test_cnn, y_test_raw, y_pred_adv, model_name)
    
    print("\n=== Resumed Training Complete ===")

if __name__ == '__main__':
    main()
