import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer_name(model):
    """
    Identifies the name of the last Conv2D layer in a Keras model.

    This is a helper function for Grad-CAM.

    Args:
        model (tf.keras.Model): The model to search.

    Returns:
        str: The name of the last convolutional layer.

    Raises:
        ValueError: If no Conv2D layer is found in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def grad_cam(model, image_array, layer_name=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.
    """
    img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    if len(img_tensor.shape) == 2:
        img_tensor = tf.expand_dims(tf.expand_dims(img_tensor, axis=-1), axis=0)
    elif len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)

    if layer_name is None:
        try:
            layer_name = get_last_conv_layer_name(model)
        except ValueError:
            return np.zeros((28, 28))

    # --- Extremely Robust Functional Model Reconstruction ---
    grad_model = None
    try:
        # Step 1: Try to reconstruct manually (Most reliable for Sequential in Keras 3)
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        x = inputs
        layer_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == layer_name:
                layer_output = x
        model_output = x
        grad_model = tf.keras.models.Model(inputs, [layer_output, model_output])
    except Exception:
        try:
            # Step 2: Fallback to using symbolic tensors directly if possible
            grad_model = tf.keras.models.Model(
                [model.input],
                [model.get_layer(layer_name).output, model.layers[-1].output]
            )
        except Exception:
            # Step 3: Absolute fallback - return empty if model cannot be tapped
            return np.zeros((28, 28))

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient calculations
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        # Path is disconnected - return blank
        return np.zeros((28, 28))
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    
    # Resize to original shape
    heatmap = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
    return heatmap

def get_overlayed_image(image_array, heatmap, alpha=0.4):
    """
    Superimposes a Grad-CAM heatmap onto an original image.
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if image_array.max() <= 1.0:
        original = np.uint8(255 * image_array)
    else:
        original = np.uint8(image_array)
        
    if len(original.shape) == 3 and original.shape[-1] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        
    superimposed_img = cv2.addWeighted(jet, alpha, original, 1 - alpha, 0)
    return superimposed_img

def saliency_map(model, image_array):
    """
    Generates a gradients-based saliency map.
    """
    img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, img_tensor)
    if grads is None:
        return np.zeros((28, 28))
        
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = saliency / (tf.math.reduce_max(saliency) + 1e-10)
    return saliency.numpy()
