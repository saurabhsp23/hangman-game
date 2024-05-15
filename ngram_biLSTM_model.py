import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Input, Flatten, Attention, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import random
import string
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import re
import functools

max_tries = 6
max_epochs = 3
c2i = {letter: idx for idx, letter in enumerate(string.ascii_lowercase)}
c2i[' '] = 26
c2i['_'] = 27
max_word_length = 35
alhabets_len = 26


class NeuralNetworkHangman:
    def __init__(self, dictionary):
        self.max_word_length = max_word_length
        self.dictionary = dictionary
        self.embedding_dim = 64  # Dimensionality of embedding layer
        self.c2i = c2i
        self.model = self.build_model()
        self.preprocess_data()


    def build_model(self):
        input_layer = Input(shape=(self.max_word_length,))
        embeddings = Embedding(input_dim=len(self.c2i), output_dim=self.embedding_dim)(input_layer)
        lstm_out = Bidirectional(LSTM(64, return_sequences=True))(embeddings)

        attention_out = Attention()([lstm_out, lstm_out])
        attention_out = Flatten()(attention_out)

        # Fully connected layers
        dense_layer = Dense(64, activation='relu')(attention_out)
        output_layer = Dense(alhabets_len, activation='softmax')(dense_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model


    def encode_word(self, word):
        # Encode the word into integer indices for embedding
        encoded = [self.c2i.get(char, self.c2i['_']) for char in word.rjust(self.max_word_length, ' ')]
        return np.array(encoded)


    def preprocess_data(self):

        self.x_train = np.array(pd.read_csv('x_train_ngrams_0.8_split_sequential.csv', index_col=0))
        self.y_train = np.array(pd.read_csv('y_train_ngrams_0.8_split_sequential.csv', index_col=0))
        return
        # x_train = []
        # y_train = []
        #
        # for word in self.dictionary:
        #     self.letters_masked = None
        #     for _ in range(26):  # Generating multiple examples per word
        #
        #         masked_word, target = self.random_mask(word)
        #         self.letters_masked.remove(target)
        #
        #         x_train.append(self.encode_word(masked_word))
        #         y_train.append(self.c2i[target] if target in self.c2i else self.c2i['_'])
        #
        #         if len(self.letters_masked) == 0:
        #             break
        #
        # self.x_train = np.array(x_train)
        # self.y_train = np.array(y_train)
        # pd.DataFrame(self.x_train).to_csv('x_train_ngrams_0.8_split_sequential.csv')
        # pd.DataFrame(self.y_train).to_csv('y_train_ngrams_0.8_split_sequential.csv')


    def train(self, epochs=max_epochs, batch_size=64):
        checkpoint = ModelCheckpoint(
            f'model_checkpoint_h8.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            save_weights_only=False,
            verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.01, callbacks=[checkpoint, early_stopping])


    @functools.lru_cache(maxsize=21)
    def find_most_common_ngrams(self, word, n=5):

        n_gram_counts = Counter()
        n = min(n, len(word))
        for j in range(0, n):
            grams = [word[i:i + j+1] for i in range(len(word)-j)]
            gram_counts = Counter(grams)
            n_gram_counts += gram_counts
        return n_gram_counts.most_common()


    def random_mask(self, word):
        if self.letters_masked is None:
            self.letters_masked = set(word)
        masked_word_list = list(word)

        # Mask letters
        for i in range(len(masked_word_list)):
            if masked_word_list[i] in self.letters_masked:
                masked_word_list[i] = '_'

        masked_word = ''.join(masked_word_list)

        common_ngrams = self.find_most_common_ngrams(word)
        targeted_guess_counts = Counter()

        # Analyze each ngram and its context within the word
        for ngram, count in common_ngrams:

            for match in re.finditer('(?=' + re.escape(ngram) + ')', word):
                start = match.start()
                end = start + len(ngram)
                sub_word = masked_word[start:end]
                unknown_letters = [ngram[i] for i, c in enumerate(sub_word) if c == "_"]

                if len(unknown_letters) == 1 and unknown_letters[0].isalpha():
                    targeted_guess_counts[unknown_letters[0]] += len(ngram)

        # Determine the most frequently occurring targeted guess
        if targeted_guess_counts:
            n_mc_occurence = targeted_guess_counts.most_common()[0][1]
            largest_occuring_targets = [k for k, v in targeted_guess_counts.most_common() if v == n_mc_occurence]
            # Choose the most common masked letter from the ngram analysis
            target = random.choice(largest_occuring_targets)
        else:
            # If no suitable target found, default to any masked letter
           target = Counter(letters_to_mask).most_common()[0][0]

        return masked_word, target


class HangmanLocal:
    def __init__(self, new_model=True):
        full_dictionary_location = "words_250000_train.txt"
        self.train_dict, self.test_dict = self.build_dictionary(full_dictionary_location, new_model)
        if not new_model:
            self.model = load_model('model_checkpoint_h8.keras')
        else:
            self.model = NeuralNetworkHangman(self.train_dict)
            self.model.train(epochs=max_epochs)

        self.guessed_letters = []
        self.word = ""
        self.masked_word = ""
        self.tries_remain = max_tries

    def encode_word(self, word):
        # Encode the word into integer indices for embedding
        encoded = [c2i.get(char, c2i['_']) for char in word.rjust(max_word_length, ' ')]
        return np.array(encoded)

    def build_dictionary(self, dictionary_file_location, new_model, tt_split=0.8):
        if new_model:
            with open(dictionary_file_location, "r") as text_file:
                full_dict = text_file.read().splitlines()
                random.shuffle(full_dict)
                pd.DataFrame(full_dict).to_csv('shuffled_dict.csv', index=None)

        else:

            full_dict = pd.read_csv('shuffled_dict.csv')
            full_dict = list(full_dict['0'])

        return full_dict[:int(tt_split*len(full_dict))], full_dict[int(tt_split*len(full_dict)):]


    def predict_letter(self, current_masked_word, guessed_letters):
        encoded = self.encode_word(current_masked_word.replace(' ', ''))
        probabilities = self.model.predict(np.array([encoded]))[0]
        for idx, letter in enumerate(string.ascii_lowercase):
            if letter in guessed_letters:
                probabilities[idx] = -2  # Exclude guessed letters
        return string.ascii_lowercase[np.argmax(probabilities)]

    def guess(self, word):
        return self.predict_letter(word, self.guessed_letters)

    def start_new_game(self, word=None):
        self.guessed_letters = []
        self.word = word if word else random.choice(self.test_dict)
        self.masked_word = '_ ' * len(self.word)
        self.tries_remain = 6

    def make_guess(self, letter):
        self.guessed_letters.append(letter)
        if letter in self.word:
            new_masked = "".join([w if w == letter or m != '_' else '_'
                                  for w, m in zip(self.word, self.masked_word.split())])
            self.masked_word = " ".join(new_masked)
            return True
        else:
            self.tries_remain -= 1
            return False

    def play_game(self, word=None):
        self.start_new_game(word)
        while self.tries_remain > 0 and "_" in self.masked_word:
            print(f"Current masked word: {self.masked_word}")
            guess_letter = self.guess(self.masked_word.replace(" ", ""))
            was_correct = self.make_guess(guess_letter)
            if not was_correct:
                print(f"Incorrect guess: {guess_letter}")
            else:
                print(f"Correct guess: {guess_letter}")
            print(f"Word: {self.masked_word}, Tries left: {self.tries_remain}")
            if "_" not in self.masked_word:
                print(f"Successfully guessed the word: {self.word}")
                return True
        print(f"Failed to guess the word: {self.word}")
        return False


if __name__ == '__main__':
    hangman = HangmanLocal(new_model=True)
    n_trials = 500
    wins = 0
    for _ in range(n_trials):
        if hangman.play_game():
            wins += 1
    print(f"Accuracy Score: {wins / n_trials:.2%}")