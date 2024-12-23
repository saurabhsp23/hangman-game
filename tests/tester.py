import random
import re
import collections


class HangmanLocal:
    def __init__(self):
        # Loading dicts
        full_dictionary_location = "words_250000_train.txt"
        self.train_dict, self.test_dict = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter(
            "".join(self.train_dict)
        ).most_common()
        self.current_dictionary = []
        self.guessed_letters = []
        self.word = ""
        self.masked_word = ""
        self.tries_remain = 6

    def build_dictionary(self, dictionary_file_location):
        with open(dictionary_file_location, "r") as text_file:

            # splitting the train dict in 50% train test split. splitting such that test and train
            # has same number of letters from a through z
            full_dict = text_file.read().splitlines()
            train_dict = full_dict[::2]
            test_dict = full_dict[1::2]
            return train_dict, test_dict

    def start_new_game(self, word=None):
        self.guessed_letters = []
        self.current_dictionary = self.train_dict

        #fetch word from test_dict
        self.word = word if word else random.choice(self.test_dict)
        self.masked_word = "_ " * len(self.word)
        self.tries_remain = 6
        return self.masked_word

    def guess(self, word):
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)
        current_dictionary = self.current_dictionary
        new_dictionary = []

        for dict_word in current_dictionary:
            if len(dict_word) != len_word:
                continue
            if re.match(clean_word, dict_word):
                new_dictionary.append(dict_word)

        self.current_dictionary = new_dictionary
        full_dict_string = "".join(new_dictionary)
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()
        guess_letter = "!"

        for letter, _ in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break

        if guess_letter == "!":
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter, _ in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break

        return guess_letter

    def make_guess(self, letter):
        self.guessed_letters.append(letter)
        if letter in self.word:
            new_masked = ""
            for w, m in zip(self.word, self.masked_word.split(" ")):
                new_masked += f"{w} " if w == letter or m != "_" else "_ "
            self.masked_word = new_masked.strip()
            return True
        else:
            self.tries_remain -= 1
            return False

    def play_game(self, word=None):
        word = self.start_new_game(word)
        while self.tries_remain > 0 and "_" in self.masked_word:
            print(f"guessing: {self.masked_word}")

            guess_letter = self.guess(self.masked_word)
            if not self.make_guess(guess_letter):
                print(f"Incorrect guess: {guess_letter}")
            else:
                print(f"Correct guess: {guess_letter}")

            print(f"Word: {self.masked_word}, Tries left: {self.tries_remain}")

        if "_" not in self.masked_word:
            print(f"Successfully guessed word: {self.word}")
            return True
        else:
            print(f"Failure: # of tries exceeded. word: {self.word}")
            return False



if __name__ == '__main__':
    n_trials = 400
    wins = 0
    for _ in range(n_trials):
        hangman_local = HangmanLocal()
        wins += int(hangman_local.play_game())
    accuracy_score = wins / n_trials
    print(f"Accuracy Score: {(wins/n_trials):.2%}")

