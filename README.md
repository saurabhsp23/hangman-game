#  Hangman Project

This project implements a LSTM-based Hangman game. The code has been modularized for better readability, maintainability, and extensibility.

## Directory Structure

- **`encode_utils.py`**: Contains utility functions for encoding the masked word and guesses.
- **`model_definition.py`**: Defines the neural network architecture for the Hangman game.
- **`data_preprocessing.py`**: Includes logic for processing and preparing the training data.
- **`model_training.py`**: Provides the training logic for the neural network model.
- **`game_logic.py`**: Implements the core Hangman game logic, including the local Hangman game class.
- **`main.py`**: Orchestrates the entire workflow, from data preparation to model training and playing the game.

## How to Run

1. **Install Dependencies**:
   Ensure you have the necessary Python libraries installed. You can install them using:
   ```bash
   pip install numpy pandas tensorflow
   ```

2. **Prepare the Dataset**:
   Update the `data_preprocessing.py` with the path to your dataset if needed.

3. **Train the Model**:
   Run the `main.py` file to preprocess the data, train the model, and save checkpoints.

4. **Play the Game**:
   After training, the game logic can be tested by invoking the methods in `game_logic.py` or directly running `main.py`.

## Model Overview

1. **Embedding Layer**:
    - Converts input tokens (characters) into dense vector representations, enabling the model to learn
      semantic relationships between characters.
2. **LSTM Layer**:
    - Processes sequences of characters, capturing dependencies and context in the game.
3. **Attention Mechanism**:
    - Focuses on the most relevant parts of the sequence, which helps in prioritizing critical regions
      of the masked word and context.
4. **Dense Layers**:
    - Transforms the processed features into a prediction over the possible next letters.
5. **Softmax Output**:
    - Provides a probability distribution over all possible letters, enabling the model to make an informed guess.

This architecture is ideal for Hangman because it models sequential dependencies and uses attention to effectively prioritize parts of the input that are crucial for guessing the next letter.

