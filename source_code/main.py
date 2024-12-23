
from model_definition import build_model
from data_preprocessing import preprocess_data
from model_training import train_model
from game_logic import HangmanLocal
import random

if __name__ == '__main__':
    # Prepare data
    dictionary = ["example", "hangman", "words"]
    train_dict, test_dict = dictionary[:int(0.8 * len(dictionary))], dictionary[int(0.8 * len(dictionary)):]
    
    # Build and train model
    model = build_model(input_len=35, vocab_size=28)
    x_train, y_train = preprocess_data(train_dict, c2i)
    train_model(model, x_train, y_train, epochs=6)

    # Play game
    hangman = HangmanLocal(train_dict, test_dict, model)
    hangman.start_new_game()
