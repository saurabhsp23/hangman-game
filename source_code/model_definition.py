import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Flatten, Attention


class ModelBuilder:
    @staticmethod
    def build(input_len, vocab_size, embedding_dim=64, output_dim=26):

        """
        Builds and compiles the neural network model for Hangman.

        The model uses an embedding layer, LSTM for sequence processing, attention mechanism
        to focus on relevant features, and dense layers for classification. It is designed
        for predicting the next letter in the game of Hangman.

        Args:
            input_len (int): The length of the input sequence (e.g., masked word + additional features).
            vocab_size (int): The size of the vocabulary, including letters and special characters.
            embedding_dim (int, optional): The dimensionality of the embedding layer. Defaults to 64.
            output_dim (int, optional): The number of output classes, representing possible next guesses.
                Defaults to 26 for the English alphabet.

        Returns:
            tensorflow.keras.Model: A compiled Keras model ready for training.

        """


        input_layer = Input(shape=(input_len,))
        embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)
        lstm_out = LSTM(64, return_sequences=True)(embeddings)
        attention_out = Attention()([lstm_out, lstm_out])
        attention_out = Flatten()(attention_out)
        dense_layer = Dense(64, activation='relu')(attention_out)
        output_layer = Dense(output_dim, activation='softmax')(dense_layer)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
