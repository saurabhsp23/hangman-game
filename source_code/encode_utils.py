# encode_utils.py
import numpy as np
import string


class Encoder:
    alphabet_list = string.ascii_lowercase
    max_tries = 6
    max_input_len = 35
    c2i = {letter: idx for idx, letter in enumerate(string.ascii_lowercase + '_ ')}

    @staticmethod
    def encode_word(masked_word, clean_word, incorrect_guesses):
        """
        Encodes the word for model processing.
        """
        masked_word = incorrect_guesses.ljust(Encoder.max_tries, ' ') + masked_word.rjust(
            Encoder.max_input_len - Encoder.max_tries, ' ')
        encoded = [Encoder.c2i.get(char) for char in masked_word]
        return np.array(encoded)


