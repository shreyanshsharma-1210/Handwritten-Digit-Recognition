from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

def create_basic_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs a modern variant of the classic LeNet-5 CNN for digit classification.

    The network follows a (Conv -> ReLU -> MaxPool) x 2 structure, followed 
    by a flattening layer and a fully connected classifier with Dropout 
    regularization. This is a reliable baseline for the MNIST dataset.

    Args:
        input_shape (tuple): Shape of individual input images (Height, Width, Channel). 
            Defaults to (28, 28, 1).
        num_classes (int): Number of target classification classes. 
            Defaults to 10.

    Returns:
        tf.keras.Model: The constructed and uncompiled Sequential model.
    """
    model = Sequential([
        # First convolutional block: 32 filters, 3x3 kernel
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name="conv2d_1"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_1"),
        
        # Second convolutional block: 64 filters, 3x3 kernel
        Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2d_2"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_2"),
        
        # Flatten feature maps into linear vector
        Flatten(name="flatten"),
        
        # Dense classifier head
        Dense(128, activation='relu', name="dense_1"),
        Dropout(0.5, name="dropout_1"),
        
        # Final classification output
        Dense(num_classes, activation='softmax', name="output")
    ], name="cnn_basic")
    
    return model

if __name__ == '__main__':
    # Build and print model summary for verification
    model = create_basic_cnn()
    model.summary()
