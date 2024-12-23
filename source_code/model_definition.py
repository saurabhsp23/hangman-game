
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Flatten, Attention

def build_model(input_len, vocab_size, embedding_dim=64, output_dim=26):
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
