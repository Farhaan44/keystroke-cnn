import os
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, X_train, y_train, X_test, y_test,
                save_path="models/best_model.h5", epochs=20, batch_size=32):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = ModelCheckpoint(save_path, monitor="val_accuracy",
                                 save_best_only=True, mode="max")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
        verbose=1
    )
    return history
