import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

def load_mnist_data():
    """Load and return MNIST dataset"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

def normalize_images(images):
    """Normalize pixel values to [0, 1]"""
    return images.astype('float32') / 255.0

def reshape_for_cnn(images):
    """Reshape for CNN input (add channel dimension)"""
    return images.reshape(-1, 28, 28, 1)

def create_train_val_split(X, y, val_split=0.2, random_state=42):
    """Split training data into train and validation"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val

# Optional testing if module executed directly
if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    print(f"X_train original shape: {X_train.shape}")
    
    X_train_norm = normalize_images(X_train)
    print(f"X_train normalized shape: {X_train_norm.shape}")
    print(f"X_train normalized max value: {np.max(X_train_norm)}")
    
    X_train_cnn = reshape_for_cnn(X_train_norm)
    print(f"X_train CNN shape: {X_train_cnn.shape}")
    
    X_t, X_v, y_t, y_v = create_train_val_split(X_train_cnn, y_train)
    print(f"Final training set size: {X_t.shape[0]}")
    print(f"Validation set size: {X_v.shape[0]}")
