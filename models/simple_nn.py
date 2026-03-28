from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

def create_simple_nn(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates a baseline feed-forward Dense Neural Network for digit classification.

    The architecture consists of a flattening layer, two fully connected (Dense) 
    layers with ReLU activation, dropout for regularization, and a final 
    Softmax output layer.

    Args:
        input_shape (tuple): Shape of individual input images. Defaults to (28, 28, 1).
        num_classes (int): Number of target classes. Defaults to 10.

    Returns:
        tf.keras.Model: The constructed and uncompiled Sequential model.
    """
    model = Sequential([
        # Flatten (28, 28, 1) into (784,)
        Flatten(input_shape=input_shape, name="flatten_input"),
        
        # Hidden Layer 1
        Dense(512, activation='relu', kernel_initializer='he_normal', name="dense_1"),
        Dropout(0.2, name="dropout_1"),
        
        # Hidden Layer 2
        Dense(256, activation='relu', kernel_initializer='he_normal', name="dense_2"),
        Dropout(0.2, name="dropout_2"),
        
        # Output Layer (10 classes with Softmax)
        Dense(num_classes, activation='softmax', name="output")
    ], name="simple_nn")
    
    return model

if __name__ == '__main__':
    # Build and print a summary for sanity check
    model = create_simple_nn()
    model.summary()
