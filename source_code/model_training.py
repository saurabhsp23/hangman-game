from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class ModelTrainer:
    def __init__(self, model, x_train, y_train, batch_size=300):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size

    def train(self, epochs=6):
        """
        Trains the model with checkpointing and early stopping.
        """
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}/{epochs}...")
            checkpoint = ModelCheckpoint(f'model_checkpoint_h{epoch + 20}.keras', monitor='val_accuracy',
                                         save_best_only=True, mode='max', verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=self.batch_size, validation_split=0.02,
                           callbacks=[checkpoint, early_stopping])


