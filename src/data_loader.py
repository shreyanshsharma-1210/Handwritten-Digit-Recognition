import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

def load_mnist_data():
    """
    Loads the MNIST dataset from Keras.

    Returns:
        tuple: (X_train, y_train), (X_test, y_test)
            - X_train, X_test (numpy.ndarray): Image data with shape (N, 28, 28)
            - y_train, y_test (numpy.ndarray): Labels 0-9 with shape (N,)
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

def normalize_images(images):
    """
    Normalizes image pixel values from [0, 255] to [0, 1].

    Args:
        images (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Normalized image array with float32 values.
    """
    return images.astype('float32') / 255.0

def reshape_for_cnn(images):
    """
    Reshapes grayscale images for CNN input by adding a channel dimension.

    Args:
        images (numpy.ndarray): Input image array (N, 28, 28).

    Returns:
        numpy.ndarray: Reshaped image array (N, 28, 28, 1).
    """
    if len(images.shape) == 3:
        return images.reshape(-1, 28, 28, 1)
    return images

def create_train_val_split(X, y, val_split=0.2, random_state=42):
    """
    Splits the training data into stratified training and validation sets.

    Args:
        X (numpy.ndarray): Input features (images).
        y (numpy.ndarray): Input targets (labels).
        val_split (float): Fraction of data to use for validation. Defaults to 0.2.
        random_state (int): Seed for reproducibility. Defaults to 42.

    Returns:
        tuple: X_train, X_val, y_train, y_val
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val

if __name__ == '__main__':
    # Brief health check for the data loader module
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    print(f"Loaded {X_train.shape[0]} training samples and {X_test.shape[0]} test samples.")
    
    X_train_norm = normalize_images(X_train)
    X_train_cnn = reshape_for_cnn(X_train_norm)
    
    X_t, X_v, y_t, y_v = create_train_val_split(X_train_cnn, y_train)
    print(f"Splits created: Train={X_t.shape[0]}, Val={X_v.shape[0]}")
