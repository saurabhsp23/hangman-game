
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model, x_train, y_train, epochs=6, batch_size=300):
    for epoch in range(epochs):
        checkpoint = ModelCheckpoint(f'model_checkpoint_h{epoch+20}.keras', monitor='val_accuracy',
                                      save_best_only=True, mode='max', verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_split=0.02,
                  callbacks=[checkpoint, early_stopping])
