
import random
from encode_utils import encode_word

class HangmanLocal:
    def __init__(self, train_dict, test_dict, model):
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.model = model
        self.clean_word = ""
        self.masked_word = ""
        self.guessed_letters = []
        self.tries_remain = 6
        self.incorrect_guesses = ""

    def start_new_game(self, word=None):
        self.guessed_letters = []
        self.clean_word = word if word else random.choice(self.test_dict)
        self.masked_word = "_ " * len(self.clean_word)
        self.tries_remain = 6
        self.incorrect_guesses = ""

    def predict_letter(self):
        encoded = encode_word(self.masked_word.replace(" ", ""), self.clean_word, self.incorrect_guesses)
        probabilities = self.model.predict([encoded])[0]
        for idx, letter in enumerate(probabilities):
            if letter in self.guessed_letters:
                probabilities[idx] = -2
        return max(probabilities, key=probabilities.get)
