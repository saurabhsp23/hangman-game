from model_definition import ModelBuilder
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from game_logic import HangmanLocal
from encode_utils import Encoder


def read_words_from_file(file_path):
    """
    Reads a text file with one word per line and converts it to a Python list.
    """

    with open(file_path, 'r') as file:
        words = [line.strip() for line in file if line.strip()]
    return words

if __name__ == '__main__':

    print('fetching data dictionary..')
    dict_path = '../data/words_250000_train.txt'
    dictionary = read_words_from_file(dict_path)

    print('splitting train & test data..')
    train_dict, test_dict = dictionary[:int(0.8 * len(dictionary))], dictionary[int(0.8 * len(dictionary)):]

    print('building model..')
    # Build and train model
    model = ModelBuilder.build(input_len=35, vocab_size=28)
    x_train, y_train = DataPreprocessor.preprocess_data(train_dict, Encoder.c2i)

    print('started training..')
    trainer = ModelTrainer(model, x_train, y_train)
    trainer.train(epochs=6)

    print('playing on test data..')
    hangman = HangmanLocal(train_dict, test_dict, model)
    hangman.start_new_game()
