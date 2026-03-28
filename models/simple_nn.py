from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

def create_simple_nn(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates a simple Dense Neural Network for digit classification.
    Architecture: Flatten -> Dense(512) -> Dropout -> Dense(256) -> Dropout -> Dense(10)
    """
    model = Sequential([
        Flatten(input_shape=input_shape, name="flatten_input"),
        Dense(512, activation='relu', name="dense_1"),
        Dropout(0.2, name="dropout_1"),
        Dense(256, activation='relu', name="dense_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(num_classes, activation='softmax', name="output")
    ], name="simple_nn")
    
    return model

if __name__ == '__main__':
    model = create_simple_nn()
    model.summary()
