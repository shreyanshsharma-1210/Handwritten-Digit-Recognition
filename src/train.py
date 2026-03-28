import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def compile_model(model, learning_rate=0.001):
    """
    Compiles a Keras model with Adam optimizer and Sparse Categorical Crossentropy loss.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): Initial learning rate for Adam. Defaults to 0.001.

    Returns:
        tf.keras.Model: The compiled model.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_callbacks(model_name):
    """
    Constructs a list of standard callbacks for training.

    Includes EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and CSVLogger.

    Args:
        model_name (str): The name of the model, used for file naming.

    Returns:
        list: A list of tf.keras.callbacks.Callback objects.
    """
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        # Save the best version of the model
        ModelCheckpoint(f'models/saved_models/{model_name}_best.keras', save_best_only=True),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        # Save training metrics to a CSV for later plotting
        CSVLogger(f'results/logs/{model_name}_training.log')
    ]
    return callbacks

def train_model(model, model_name, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
    """
    Executes the training loop for a given model.

    Args:
        model (tf.keras.Model): The model to train.
        model_name (str): Identifier for naming saved artifacts.
        X_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation images.
        y_val (numpy.ndarray): Validation labels.
        epochs (int): Maximum number of training epochs. Defaults to 50.
        batch_size (int): Size of training batches. Defaults to 128.

    Returns:
        tf.keras.callbacks.History: The history object containing training logs.
    """
    callbacks = get_callbacks(model_name)
    print(f"Starting training for model: {model_name}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history
