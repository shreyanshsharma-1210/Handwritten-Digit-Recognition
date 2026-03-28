from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D

def create_advanced_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Constructs an Advanced CNN architecture with Batch Normalization and 
    Global Average Pooling for superior generalization.

    This architecture is designed to handle more complex variations 
    and augmentations. Key features include repeated Batch Normalization 
    to prevent internal covariate shift and Global Average Pooling to 
    reduce parameters and prevent overfitting in the classifier head.

    Args:
        input_shape (tuple): Shape of individual input images (H, W, C). 
            Defaults to (28, 28, 1).
        num_classes (int): Number of target classification classes. 
            Defaults to 10.

    Returns:
        tf.keras.Model: The constructed and uncompiled Sequential model.
    """
    model = Sequential([
        # Block 1: 32 filters, 3x3, same padding
        Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape, name="conv2d_1"),
        BatchNormalization(name="batchnorm_1"),
        Activation('relu', name="relu_1"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_1"),
        
        # Block 2: 64 filters, 3x3, same padding
        Conv2D(64, kernel_size=(3, 3), padding='same', name="conv2d_2"),
        BatchNormalization(name="batchnorm_2"),
        Activation('relu', name="relu_2"),
        MaxPool2D(pool_size=(2, 2), name="maxpool_2"),
        
        # Block 3: 128 filters, 3x3, same padding
        Conv2D(128, kernel_size=(3, 3), padding='same', name="conv2d_3"),
        BatchNormalization(name="batchnorm_3"),
        Activation('relu', name="relu_3"),
        
        # Classifier Head
        GlobalAveragePooling2D(name="global_avg_pool"),
        Dense(256, name="dense_1"),
        BatchNormalization(name="batchnorm_4"),
        Activation('relu', name="relu_4"),
        Dropout(0.5, name="dropout_1"),
        
        Dense(num_classes, activation='softmax', name="output")
    ], name="cnn_advanced")
    
    return model

if __name__ == '__main__':
    # Build and print model summary for verification
    model = create_advanced_cnn()
    model.summary()
