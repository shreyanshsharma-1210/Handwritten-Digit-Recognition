from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

def create_basic_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates a Basic CNN (LeNet-5 style) for digit classification.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, name="conv2d_1"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_1"),
        
        # Second Convolutional Block
        Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2d_2"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_2"),
        
        # Classifier
        Flatten(name="flatten"),
        Dense(128, activation='relu', name="dense_1"),
        Dropout(0.5, name="dropout_1"),
        Dense(num_classes, activation='softmax', name="output")
    ], name="cnn_basic")
    
    return model

if __name__ == '__main__':
    model = create_basic_cnn()
    model.summary()
