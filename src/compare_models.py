import os
import sys
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Adjust path to find src and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_mnist_data, normalize_images, reshape_for_cnn

def get_model_metrics(model_path, X_test, y_test):
    """
    Load a model and calculate various performance metrics.
    """
    print(f"Evaluating {os.path.basename(model_path)}...")
    
    # Model size in MB
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    
    model = tf.keras.models.load_model(model_path)
    
    # Paraemters count
    params_count = model.count_params()
    
    # Inference time (mean of 100 predictions)
    start_time = time.time()
    # Warm up
    _ = model.predict(X_test[:100], verbose=0)
    
    # Actual timing
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    inference_time_per_sample = (total_time / len(X_test)) * 1000 # in ms
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
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
    print("=== Model Comparison Framework ===")
    
    # Load and preprocess test data
    (_, _), (X_test, y_test) = load_mnist_data()
    X_test = reshape_for_cnn(normalize_images(X_test))
    
    model_dir = 'models/saved_models'
    models_to_compare = [
        'simple_nn_augmented_best.keras',
        'cnn_basic_best.keras',
        'cnn_advanced_augmented_best.keras'
    ]
    
    results = []
    for model_name in models_to_compare:
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            metrics = get_model_metrics(model_path, X_test, y_test)
            results.append(metrics)
        else:
            print(f"Warning: Model {model_name} not found at {model_path}")
            
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs('results/reports', exist_ok=True)
    df.to_csv('results/reports/model_comparison.csv', index=False)
    print("\nComparison results saved to results/reports/model_comparison.csv")
    
    # Display table
    print("\nModel Comparison Table:")
    print(df.to_string(index=False))
    
    # Visualizations
    os.makedirs('results/figures', exist_ok=True)
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=df, palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0.95, 1.0)
    for i, acc in enumerate(df['Accuracy']):
        plt.text(i, acc + 0.001, f'{acc:.4f}', ha='center')
    plt.savefig('results/figures/comparison_accuracy.png')
    
    # 2. Parameters vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Params'], df['Accuracy'], s=100, color='red')
    for i, name in enumerate(df['Model']):
        plt.annotate(name, (df['Params'][i], df['Accuracy'][i]), xytext=(5, 5), textcoords='offset points')
    plt.title('Accuracy vs Number of Parameters')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('results/figures/comparison_params_vs_accuracy.png')
    
    # 3. Inference Time vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Inference Time (ms/sample)'], df['Accuracy'], s=100, color='blue')
    for i, name in enumerate(df['Model']):
        plt.annotate(name, (df['Inference Time (ms/sample)'][i], df['Accuracy'][i]), xytext=(5, 5), textcoords='offset points')
    plt.title('Accuracy vs Inference Time (ms/sample)')
    plt.xlabel('Inference Time (ms/sample)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('results/figures/comparison_latency_vs_accuracy.png')
    
    print("\nComparison visualizations saved to results/figures/")

if __name__ == "__main__":
    main()
