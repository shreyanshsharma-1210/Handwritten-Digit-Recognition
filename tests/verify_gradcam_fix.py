import os
import sys
import numpy as np
import tensorflow as tf

# Adjust path to find src/models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.explainability import grad_cam

def verify_fix():
    print("Verifying Grad-CAM fix for Advanced CNN model...")
    model_path = 'models/saved_models/cnn_advanced_augmented_best.keras'
    
    if not os.path.exists(model_path):
        # Full path alternative
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'saved_models', 'cnn_advanced_augmented_best.keras')
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # Create dummy input image
    test_img = np.random.rand(28, 28, 1).astype('float32')
    
    try:
        heatmap = grad_cam(model, test_img)
        print("Success! Grad-CAM heatmap generated successfully.")
        print(f"Heatmap shape: {heatmap.shape}")
        assert heatmap.shape == (28, 28)
    except Exception as e:
        print(f"FAILED to generate Grad-CAM: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_fix()
