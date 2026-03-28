import numpy as np
import cv2

def preprocess_canvas_image(canvas_image):
    """
    Process the RGBA image from streamlit-drawable-canvas to 28x28 grayscale.
    """
    if canvas_image is None:
        return None
        
    # canvas_image is (H, W, 4)
    if canvas_image.shape[-1] == 4:
        gray_image = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2GRAY)
    else:
        gray_image = canvas_image
        
    # Resize to 28x28
    resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0, 1]
    normalized_image = resized_image.astype('float32') / 255.0
    
    return normalized_image

def reshape_for_prediction(normalized_image):
    """
    Reshape to (batch_size, 28, 28, 1)
    """
    final_image = np.expand_dims(normalized_image, axis=-1)
    final_image = np.expand_dims(final_image, axis=0)
    return final_image
