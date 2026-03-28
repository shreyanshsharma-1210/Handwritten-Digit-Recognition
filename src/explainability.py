import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer_name(model):
    """Finds the last Conv2D layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def grad_cam(model, image_array, layer_name=None):
    """
    Generate Grad-CAM heatmap for an image array.
    """
    img_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    if len(img_tensor.shape) == 3:
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

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize to original shape (28x28)
    heatmap = cv2.resize(heatmap, (image_array.shape[1], image_array.shape[0]))
    return heatmap

def get_overlayed_image(image_array, heatmap, alpha=0.4):
    """
    Combines the heatmap with the original image.
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Normalize original image to 0-255 if it's 0-1
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
    Generate a pure gradients-based saliency map.
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
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = saliency / tf.math.reduce_max(saliency)
    return saliency.numpy()
