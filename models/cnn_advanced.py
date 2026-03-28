from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

def create_advanced_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates an Advanced CNN with BatchNorm and Dropout for digit classification.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape, name="conv2d_1"),
        BatchNormalization(name="batchnorm_1"),
        Activation('relu', name="relu_1"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_1"),
        
        # Second Convolutional Block
        Conv2D(64, kernel_size=(3, 3), padding='same', name="conv2d_2"),
        BatchNormalization(name="batchnorm_2"),
        Activation('relu', name="relu_2"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_2"),
        
        # Third Convolutional Block
        Conv2D(128, kernel_size=(3, 3), padding='same', name="conv2d_3"),
        BatchNormalization(name="batchnorm_3"),
        Activation('relu', name="relu_3"),
        
        # Classifier
        GlobalAveragePooling2D(name="global_avg_pool"),
        Dense(256, name="dense_1"),
        BatchNormalization(name="batchnorm_4"),
        Activation('relu', name="relu_4"),
        Dropout(0.5, name="dropout_1"),
        
        Dense(num_classes, activation='softmax', name="output")
    ], name="cnn_advanced")
    
    return model

if __name__ == '__main__':
    model = create_advanced_cnn()
    model.summary()
