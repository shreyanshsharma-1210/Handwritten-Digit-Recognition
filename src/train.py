import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

def compile_model(model, learning_rate=0.001):
    """Compile model with optimizer and loss"""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_callbacks(model_name):
    """Return list of callbacks for training"""
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(f'models/saved_models/{model_name}_best.keras', save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        CSVLogger(f'results/logs/{model_name}_training.log')
    ]
    return callbacks

def train_model(model, model_name, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
    """Train model with callbacks"""
    callbacks = get_callbacks(model_name)
    print(f"Training model: {model_name}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history
