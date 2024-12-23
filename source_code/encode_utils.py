
import numpy as np
import string

alphabet_list = string.ascii_lowercase
max_tries = 6
max_input_len = 35
c2i = {letter: idx for idx, letter in enumerate(string.ascii_lowercase + '_ ')}

def encode_word(masked_word, clean_word, incorrect_guesses):
    masked_word = incorrect_guesses.ljust(max_tries, ' ') + masked_word.rjust(max_input_len - max_tries, ' ')
    encoded = [c2i.get(char) for char in masked_word]
    return np.array(encoded)
