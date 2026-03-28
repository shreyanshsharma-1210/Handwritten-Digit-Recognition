import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer_name(model):
    """Returns the name of the last Conv2D layer in a model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def grad_cam(model, image_array, layer_name=None):
    """
    Generates a Grad-CAM heatmap using a manual layer-by-layer forward pass.
    This approach is fully compatible with Keras 3 Sequential models.

    Args:
        model: A trained Keras model.
        image_array: Input image as numpy array (28, 28, 1) or (28, 28).
        layer_name: Name of the conv layer to target. Uses last Conv2D if None.

    Returns:
        numpy.ndarray: A normalized 2D heatmap (28, 28).
    """
    # Normalize input shape
    img = np.array(image_array, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    img_tensor = tf.cast(img[np.newaxis, ...], tf.float32)  # (1, 28, 28, 1)

    if layer_name is None:
        try:
            layer_name = get_last_conv_layer_name(model)
        except ValueError:
            return np.zeros((28, 28))

    # --- Manual forward pass with explicit tape watching ---
    # This is the correct approach for Keras 3 Sequential models.
    # We run each layer manually, watch the conv output, and let the tape
    # track the gradient path from the prediction back to that feature map.
    conv_outputs = None

    try:
        with tf.GradientTape() as tape:
            x = img_tensor
            for layer in model.layers:
                x = layer(x)
                if layer.name == layer_name:
                    tape.watch(x)   # Explicitly mark for gradient tracking
                    conv_outputs = x

            predictions = x
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        if conv_outputs is None:
            return np.zeros((28, 28))

        grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            return np.zeros((28, 28))

        # Pool gradients across feature map spatial dims
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by their importance
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        # Resize to input dimensions
        h, w = img.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        return heatmap

    except Exception:
        return np.zeros((28, 28))

def get_overlayed_image(image_array, heatmap, alpha=0.4):
    """
    Superimposes a Grad-CAM heatmap onto an original image.

    Args:
        image_array: Original grayscale image (H, W, 1) or (H, W).
        heatmap: Normalized 2D heatmap array.
        alpha: Heatmap opacity blend factor (0-1).

    Returns:
        numpy.ndarray: RGB blended image.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    original = np.uint8(255 * image_array) if image_array.max() <= 1.0 else np.uint8(image_array)

    if original.ndim == 3 and original.shape[-1] == 1:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

    return cv2.addWeighted(jet, alpha, original, 1 - alpha, 0)

def saliency_map(model, image_array):
    """
    Generates a pixel-level saliency map using input gradients.

    Args:
        model: A trained Keras model.
        image_array: Input image (28, 28, 1).

    Returns:
        numpy.ndarray: Normalized 2D saliency map.
    """
    img = np.array(image_array, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    img_tensor = tf.Variable(img[np.newaxis, ...])

    with tf.GradientTape() as tape:
        predictions = model(img_tensor, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, img_tensor)
    if grads is None:
        return np.zeros(img.shape[:2])

    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = saliency / (tf.math.reduce_max(saliency) + 1e-10)
    return saliency.numpy()
