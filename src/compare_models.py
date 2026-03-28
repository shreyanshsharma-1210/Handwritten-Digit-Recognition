import os
import sys
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Adjust path to find src and models when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_mnist_data, normalize_images, reshape_for_cnn

def get_model_metrics(model_path, X_test, y_test):
    """
    Evaluates a saved model and calculates technical and performance metrics.

    Metrics include accuracy, precision, recall, F1-score, number of parameters, 
    model file size, and average inference time per sample.

    Args:
        model_path (str): Path to the saved .keras model file.
        X_test (numpy.ndarray): Test images for evaluation.
        y_test (numpy.ndarray): True labels for evaluation.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    print(f"Evaluating {os.path.basename(model_path)}...")
    
    # Model size in MB
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Total trainable/non-trainable parameters
    params_count = model.count_params()
    
    # Benchmark inference speed
    # Note: Use a small batch to simulate real-world latency or large for throughput
    # Here we measure total time over the whole test set
    
    # Warm up prediction to initialize GPU/CPU caches
    _ = model.predict(X_test[:100], verbose=0)
    
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    inference_time_per_sample = (total_time / len(X_test)) * 1000 # convert to ms
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Statistical performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    return {
        "Model": os.path.basename(model_path).replace('_best.keras', ''),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Params": params_count,
        "Size (MB)": model_size,
        "Inference Time (ms/sample)": inference_time_per_sample
    }

def main():
    """
    Main execution logic for comparing all trained models in the project.
    
    Loads pre-trained models from the default directory and generates
    a comparative CSV report and summary visualizations.
    """
    print("=== Model Comparison Framework ===")
    
    # 1. Load and preprocess test data
    (_, _), (X_test, y_test) = load_mnist_data()
    X_test = reshape_for_cnn(normalize_images(X_test))
    
    model_dir = 'models/saved_models'
    models_to_compare = [
        'simple_nn_augmented_best.keras',
        'cnn_basic_best.keras',
        'cnn_advanced_augmented_best.keras'
    ]
    
    # 2. Collect metrics for each model
    results = []
    for model_name in models_to_compare:
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            metrics = get_model_metrics(model_path, X_test, y_test)
            results.append(metrics)
        else:
            print(f"Warning: Model {model_name} not found at {model_path}")
            
    if not results:
        print("Error: No models found to compare. Please run training first.")
        return

    # 3. Process and save results
    df = pd.DataFrame(results)
    
    os.makedirs('results/reports', exist_ok=True)
    df.to_csv('results/reports/model_comparison.csv', index=False)
    print("\nComparison results saved to results/reports/model_comparison.csv")
    
    # 4. Display summary table
    print("\nModel Comparison Table:")
    print(df.to_string(index=False))
    
    # 5. Generate Visualizations
    os.makedirs('results/figures', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=df, palette='viridis')
    plt.title('Handwritten Digit Recognition: Model Accuracy', fontsize=15)
    plt.ylim(0.95, 1.0)
    for i, acc in enumerate(df['Accuracy']):
        plt.text(i, acc + 0.001, f'{acc:.4f}', ha='center', fontweight='bold')
    plt.savefig('results/figures/comparison_accuracy.png')
    
    # --- Plot Latency vs Accuracy ---
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Inference Time (ms/sample)'], df['Accuracy'], s=300, c=range(len(df)), cmap='coolwarm', alpha=0.7)
    for i, name in enumerate(df['Model']):
        plt.annotate(name, (df['Inference Time (ms/sample)'][i], df['Accuracy'][i]), 
                     xytext=(10, 10), textcoords='offset points', fontsize=12)
    plt.title('Accuracy vs. Inference Latency', fontsize=15)
    plt.xlabel('Inference Time (ms/sample)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    plt.savefig('results/figures/comparison_latency_vs_accuracy.png')
    
    print("\nBenchmark visualizations generated in results/figures/")

if __name__ == "__main__":
    main()
