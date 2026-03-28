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

    Grad-CAM (Gradient-weighted Class Activation Mapping) uses the gradients 
    of any target concept, flowing into the final convolutional layer to 
    produce a coarse localization map highlighting important regions.

    Args:
        model (tf.keras.Model): The model to explain.
        image_array (numpy.ndarray): Input image (28, 28) or (28, 28, 1).
        layer_name (str, optional): The name of the convolutional layer to use. 
            If None, the last Conv2D layer is used.

    Returns:
        numpy.ndarray: Normalized 2D heatmap.
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
            # If there's no Conv2D (e.g., Simple NN), return a blank heatmap
            return np.zeros((28, 28))

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the class output with respect to the feature map
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted sum of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # ReLU and normalization
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()
    
    # Resize to original shape (28, 28)
    heatmap = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
    return heatmap

def get_overlayed_image(image_array, heatmap, alpha=0.4):
    """
    Superimposes a Grad-CAM heatmap onto an original image.

    Args:
        image_array (numpy.ndarray): The original grayscale image.
        heatmap (numpy.ndarray): The 2D normalized heatmap.
        alpha (float): Transparency of the heatmap overlay (0 to 1). Defaults to 0.4.

    Returns:
        numpy.ndarray: RGB image with heatmap overlay.
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Ensure original is 0-255 scale
    if image_array.max() <= 1.0:
        original = np.uint8(255 * image_array)
    else:
        original = np.uint8(image_array)
        
    # Convert grayscale original to RGB
    if len(original.shape) == 3 and original.shape[-1] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        
    # Blend images
    superimposed_img = cv2.addWeighted(jet, alpha, original, 1 - alpha, 0)
    return superimposed_img

def saliency_map(model, image_array):
    """
    Generates a gradients-based saliency map.

    Saliency maps highlight pixels that, when slightly changed, would 
    most significantly change the model's prediction.

    Args:
        model (tf.keras.Model): The model to explain.
        image_array (numpy.ndarray): Input image (28, 28, 1).

    Returns:
        numpy.ndarray: Normalized 2D saliency map.
    """
    img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    if len(img_tensor.shape) == 3:
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients of the class output with respect to input pixels
    grads = tape.gradient(class_channel, img_tensor)
    
    # Take the maximum absolute gradient across color channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    
    # Normalize
    saliency = saliency / (tf.math.reduce_max(saliency) + 1e-10)
    return saliency.numpy()
