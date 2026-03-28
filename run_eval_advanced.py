import sys
import os
import tensorflow as tf

from src.data_loader import load_mnist_data, normalize_images, reshape_for_cnn
from src.evaluate import evaluate_model, plot_training_curves, analyze_misclassifications

def main():
    print("Loading test data...")
    (_, _), (X_test_raw, y_test_raw) = load_mnist_data()
    X_test_norm = normalize_images(X_test_raw)
    X_test_cnn = reshape_for_cnn(X_test_norm)
    
    model_path = "models/saved_models/cnn_advanced_augmented_best.keras"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}.")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    model_name = "cnn_advanced_augmented"
    
    print("Evaluating Advanced CNN Model...")
    y_pred = evaluate_model(model, X_test_cnn, y_test_raw, model_name)
    
    print("Plotting training curves...")
    plot_training_curves(f"results/logs/{model_name}_training.log", model_name)
    
    print("Analyzing misclassifications...")
    analyze_misclassifications(X_test_cnn, y_test_raw, y_pred, model_name)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
