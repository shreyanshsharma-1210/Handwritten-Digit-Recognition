import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_training_curves(log_file, model_name):
    """Plot accuracy and loss from CSV log"""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
        
    df = pd.read_csv(log_file)
    epochs = df['epoch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss
    ax1.plot(epochs, df['loss'], label='Train Loss')
    ax1.plot(epochs, df['val_loss'], label='Val Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy
    ax2.plot(epochs, df['accuracy'], label='Train Accuracy')
    ax2.plot(epochs, df['val_accuracy'], label='Val Accuracy')
    ax2.set_title(f'{model_name} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    os.makedirs('results/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'results/figures/{model_name}_training_curves.png')
    plt.close()
    
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test data and print metrics"""
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    # Predict
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'results/figures/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Classification Report
    cr = classification_report(y_test, y_pred)
    print("\nClassification Report:\n", cr)
    
    # Save Report
    os.makedirs('results/reports', exist_ok=True)
    with open(f'results/reports/{model_name}_report.txt', 'w') as f:
        f.write(f"Test Accuracy: {results[1]:.4f}\n\n")
        f.write(cr)
        
    return y_pred

def analyze_misclassifications(X_test, y_test, y_pred, model_name, num_samples=5):
    """Plot sample misclassifications"""
    misclassified_idx = np.where(y_test != y_pred)[0]
    print(f"Total misclassified samples: {len(misclassified_idx)}")
    
    if len(misclassified_idx) == 0:
        return
        
    # Pick random misclassified samples
    sample_idx = np.random.choice(misclassified_idx, min(num_samples, len(misclassified_idx)), replace=False)
    
    fig, axes = plt.subplots(1, len(sample_idx), figsize=(15, 3))
    if len(sample_idx) == 1:
        axes = [axes]
        
    for ax, idx in zip(axes, sample_idx):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {y_test[idx]} | Pred: {y_pred[idx]}")
        ax.axis('off')
        
    plt.suptitle(f'{model_name} - Misclassification Examples')
    plt.tight_layout()
    plt.savefig(f'results/figures/{model_name}_misclassifications.png')
    plt.close()
