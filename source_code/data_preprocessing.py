import numpy as np
import random
from encode_utils import Encoder


class DataPreprocessor:
    @staticmethod
    def preprocess_data(dictionary, c2i):
        """
        Preprocesses the data for model training.
        """
        x_train, y_train = [], []
        masked_set = set()
        for clean_word in dictionary:
            for _ in range(20):
                masked_word, target = DataPreprocessor.random_mask(clean_word, c2i, masked_set)
                if masked_word is not None:
                    x_train.append(Encoder.encode_word(masked_word, clean_word, ""))
                    y_train.append(c2i[target])
        x_train, y_train = np.array(x_train), np.array(y_train)
        return x_train, y_train

    @staticmethod
    def random_mask(word, c2i, masked_set):
        """
        Randomly masks letters in the word for training data generation.
        """
        letters_to_mask = set(random.sample(word, k=random.randint(1, len(word))))
        masked_word_list = list(word)
        for i in range(len(masked_word_list)):
            if masked_word_list[i] in letters_to_mask:
                masked_word_list[i] = '_'
        masked_word = ''.join(masked_word_list)
        if masked_word in masked_set:
            return None, None
        masked_set.add(masked_word)
        target = random.choice(list(letters_to_mask))
        return masked_word, target

