import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Input, Flatten, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import random
import string
from collections import Counter
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import re
import functools

alphabet_list = string.ascii_lowercase
max_tries = 6
max_epochs = max_tries
c2i = {letter: idx for idx, letter in enumerate(string.ascii_lowercase + '_ ')}
max_input_len = 35
alhabets_len = 26


def encode_word(masked_word, clean_word, incorrect_guesses=None):
    # Encode the masked word into integer indices for embedding
    if incorrect_guesses is None:
        possible_incorrect_guesses = list(set(string.ascii_lowercase) - set(clean_word))
        incorrect_guesses = random.sample(possible_incorrect_guesses, k=random.randint(0, max_tries))
        incorrect_guesses = "".join(incorrect_guesses)

    masked_word = incorrect_guesses.ljust(max_tries, ' ') + masked_word.rjust(max_input_len - max_tries, ' ')
    encoded = [c2i.get(char) for char in masked_word]
    return np.array(encoded)

class NeuralNetworkHangman:
    def __init__(self, dictionary, new_model):
        self.max_input_len = max_input_len
        self.dictionary = dictionary
        self.embedding_dim = 64  # Dimensionality of embedding layer
        self.c2i = c2i
        self.model = self.build_model()
        self.preprocess_data(new_model)

    def build_model(self):
        input_layer = Input(shape=(self.max_input_len,))
        embeddings = Embedding(input_dim=len(self.c2i), output_dim=self.embedding_dim)(input_layer)
        lstm_out = LSTM(64, return_sequences=True)(embeddings)

        attention_out = Attention()([lstm_out, lstm_out])
        attention_out = Flatten()(attention_out)

        # Fully connected layers
        dense_layer = Dense(64, activation='relu')(attention_out)
        output_layer = Dense(alhabets_len, activation='softmax')(dense_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_data(self, new_model):
        # if not new_model:
        self.x_train = np.array(pd.read_csv('x_train_ngrams_0.8_incorrect_words_encoded.csv', index_col=0))
        self.y_train = np.array(pd.read_csv('y_train_ngrams_0.8_incorrect_words_encoded.csv', index_col=0))
        return


        # x_train = []
        # y_train = []
        # for clean_word in self.dictionary:
        #     self.masked_set = set()
        #     for _ in range(20):  # Generating multiple examples per word
        #         masked_word, target = self.random_mask(clean_word)
        #         if masked_word is not None:
        #             x_train.append(encode_word(masked_word, clean_word))
        #             y_train.append(self.c2i[target])
        # self.x_train = np.array(x_train)
        # self.y_train = np.array(y_train)
        # pd.DataFrame(self.x_train).to_csv('x_train_ngrams_0.8_incorrect_words_encoded.csv')
        # pd.DataFrame(self.y_train).to_csv('y_train_ngrams_0.8_incorrect_words_encoded.csv')
        print()

    def train(self, epochs=max_epochs, batch_size=300):

        for epoch in range(epochs):
            checkpoint = ModelCheckpoint(
                f'model_checkpoint_h{epoch+20}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                save_weights_only=False,
                verbose=1)
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=batch_size,
                           validation_split=0.05, callbacks=[checkpoint, early_stopping])

            probabilities = self.model.predict(self.x_train)

            for i in range(len(self.x_train)):
                guess = np.argmax(probabilities[i])

                if guess != self.y_train[i]:
                    self.x_train[i][epoch] = guess
                if epoch == 0:
                    self.x_train[i][epoch+1:max_tries] = c2i.get(' ')



    @functools.lru_cache(maxsize=21)
    def find_most_common_ngrams(self, word, n=5):

        n_gram_counts = Counter()
        n = min(n, len(word))
        for j in range(0, n):
            grams = [word[i:i + j + 1] for i in range(len(word) - j)]
            gram_counts = Counter(grams)
            n_gram_counts += gram_counts
        return n_gram_counts.most_common()

    def random_mask(self, word):
        letters_to_mask = set(random.sample(word, k=random.randint(1, len(word))))
        masked_word_list = list(word)

        # Mask letters randomly
        for i in range(len(masked_word_list)):
            if masked_word_list[i] in letters_to_mask:
                masked_word_list[i] = '_'

        masked_word = ''.join(masked_word_list)
        if masked_word in self.masked_set:
            return None, None

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
            self.model = load_model('model_checkpoint_h10.keras')
        else:
            self.nn = NeuralNetworkHangman(self.train_dict, new_model)
            self.nn.train(epochs=max_epochs)
            self.model = self.nn.model

        self.guessed_letters = []
        self.clean_word = ""
        self.masked_word = ""
        self.tries_remain = max_tries
        self.incorrect_guesses = None


    def build_dictionary(self, dictionary_file_location, new_model, tt_split=0.8):
        if new_model:
            with open(dictionary_file_location, "r") as text_file:
                full_dict = text_file.read().splitlines()
                random.shuffle(full_dict)
                pd.DataFrame(full_dict).to_csv('shuffled_dict.csv', index=None)

        else:
            full_dict = pd.read_csv('shuffled_dict.csv')
            full_dict = list(full_dict['0'])

        return full_dict[:int(tt_split * len(full_dict))], full_dict[int(tt_split * len(full_dict)):]

    def predict_letter(self, incorrect_guesses=None):
        encoded = encode_word(self.masked_word.replace(" ",""), self.clean_word, incorrect_guesses)
        probabilities = self.model.predict(np.array([encoded]))[0]


        for idx, letter in enumerate(string.ascii_lowercase):
            if letter in self.guessed_letters:
                probabilities[idx] = -2  # Exclude guessed letters

        guess = string.ascii_lowercase[np.argmax(probabilities)]
        if guess not in self.clean_word:
            if incorrect_guesses is not None:
                self.incorrect_guesses += guess
            else:
                self.incorrect_guesses = guess

        return guess

    def guess(self, incorrect_guesses=None):
        return self.predict_letter(incorrect_guesses)

    def start_new_game(self, word=None):
        self.guessed_letters = []
        self.clean_word = word if word else random.choice(self.test_dict)
        self.masked_word = ' _' * len(self.clean_word)
        self.tries_remain = 6
        self.incorrect_guesses = None

    def make_guess(self, letter):
        self.guessed_letters.append(letter)
        if letter in self.clean_word:
            new_masked = "".join([w if w == letter or m != '_' else '_'
                                  for w, m in zip(self.clean_word, self.masked_word.split())])
            self.masked_word = " ".join(new_masked)
            return True
        else:
            self.tries_remain -= 1
            return False

    def play_game(self, word=None):
        self.start_new_game(word)

        while self.tries_remain > 0 and "_" in self.masked_word:
            print(f"Current masked word: {self.masked_word}")
            guess_letter = self.guess(self.incorrect_guesses)
            was_correct = self.make_guess(guess_letter)
            if not was_correct:
                print(f"Incorrect guess: {guess_letter}")
            else:
                print(f"Correct guess: {guess_letter}")
            print(f"Word: {self.masked_word}, Tries left: {self.tries_remain}")
            if "_" not in self.masked_word:
                print(f"Successfully guessed the word: {self.clean_word}")
                return True
        print(f"Failed to guess the word: {self.clean_word}")
        return False


if __name__ == '__main__':
    hangman = HangmanLocal(new_model=True)
    n_trials = 200
    wins = 0
    for _ in range(n_trials):
        if hangman.play_game():
            wins += 1
    print(f"Accuracy Score: {wins / n_trials:.2%}")