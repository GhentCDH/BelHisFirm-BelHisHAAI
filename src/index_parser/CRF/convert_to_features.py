import os
import json
from functools import partial
from pathlib import Path

# This class handles the conversion of strings to feature representations 
# for training a CRF (Conditional Random Field) model.
# The code has been reworked from city-directory-entry-parser.
class Convert_To_Features:

    @staticmethod
    # Tokenizes a string into individual tokens, handles special cases for '*',
    # commas, and whitespace. Includes START and END tokens for ground-truth readability.
    def tokenize_string(string):
        tokenized_list = ["START"]  # Add START token to the beginning for CRF
        token = ''
        counter = 0
        # Set of characters that are considered "weird" (e.g., special characters)
        weird_characters = {'x', '"', '«', '»', '“', '”', '*', '-'}

        # Iterate over each character in the string
        for character in string:
            if character in weird_characters and counter == 0:
                if token:  # If there's an accumulated token, append it to the list
                    tokenized_list.append(token)
                    token = ''
                tokenized_list.append('*')  # Add '*' token for special characters
            elif character == ',':
                if token:
                    tokenized_list.append(token)
                    token = ''
                tokenized_list.append(character)  # Add comma as a token
            elif character == ' ':
                if token:  # If there's a token, append it
                    tokenized_list.append(token)
                token = ''  # Reset the token
            else:
                token += character  # Build up the token
            counter += 1
        if token:
            tokenized_list.append(token)  # Append any remaining token

        tokenized_list.append("END")  # Add END token at the end for CRF
        return tokenized_list

    @staticmethod
    # Tokenizes the string into tokens and also creates corresponding labels for training.
    # The labels are inserted as a separate list alongside the tokenized string.
    def tokenize_training_string(string, label):
        tokenized_list = []
        token = ''
        token_number = 0
        # Iterate through the string, separating tokens by spaces and commas
        for character in string:
            if character == ' ':
                if token:
                    tokenized_list.append(token)  # Append the token when space is encountered
                    token_number += 1
                token = ''  # Reset the token
            elif character == ',':
                if token:
                    tokenized_list.append(token)  # Append token for a comma
                    token_number += 1
                tokenized_list.append(',')  # Add comma as a separate token
                label.insert(token_number, 'D')  # Insert label 'D' for comma
                token_number += 1
                token = ''  # Reset token after comma
            else:
                token += character  # Build up the token
        if token:  # Append any remaining token
            tokenized_list.append(token)

        return tokenized_list, label

    @staticmethod
    # Extracts features for tokens based on neighboring tokens in the list.
    def get_token_features_from_tokenized_list(list_with_tokens):
        features = []

        # Iterate through each token in the list and gather features from neighboring tokens
        for index, token in enumerate(list_with_tokens):
            token_features = {}
            for rela_pos in range(-2, 3):  # Check for tokens in the range of -2 to 2 relative positions
                shuffle_pos = index + rela_pos
                if 0 <= shuffle_pos < len(list_with_tokens):  # Ensure the relative position is within bounds
                    relative_token = list_with_tokens[shuffle_pos]
                    token_features.update({f"{rela_pos}:token": relative_token})  # Add the relative token
                    # Add hard-coded features for each token (using feature functions defined below)
                    token_features.update({
                        f"{rela_pos}:{name}": func(relative_token)
                        for name, func in Convert_To_Features.__feature_generator().items()
                    })
            features.append(token_features)
        return features

    @staticmethod
    # Returns a dictionary of feature-generating functions
    def __feature_generator():
        return {
            "token.has.special.character": Convert_To_Features.__has_special_character,
            "token.has.only.numbers": Convert_To_Features.__only_contains_numbers,
            "token.has.some.numbers": Convert_To_Features.__contains_some_numbers,
            "token.contains.event": Convert_To_Features.__contains_event,
            "token.has.capital.letter": Convert_To_Features.__has_capital_letter,
            "token.has.only.capital.letters": Convert_To_Features.__has_only_capital_letters,
            "token.last.3.characters": partial(Convert_To_Features.__get_the_last_characters, 3),
            "token.last.2.characters": partial(Convert_To_Features.__get_the_last_characters, 2),
            "token.last.character": partial(Convert_To_Features.__get_the_last_characters, 1),
            "token.is.start": Convert_To_Features.__is_start,
            "token.is.end": Convert_To_Features.__is_end,
            "token.is.seperator": Convert_To_Features.__is_seperator,
            "token.contains.punctuation": Convert_To_Features.__contains_punctuation
        }

    # Hard-coded feature functions that will help the CRF model decide how to classify each token
    @staticmethod
    # Check if the token contains a special character (e.g., '*')
    def __has_special_character(token):
        special_character_list = ['*']  # List of special characters
        return token in special_character_list

    @staticmethod
    # Check if the token contains only numbers
    def __only_contains_numbers(token):
        return token.isdigit()

    @staticmethod
    # Check if the token contains some numbers
    def __contains_some_numbers(token):
        return any(char.isdigit() for char in token)

    @staticmethod
    # Check if the token corresponds to an event based on a predefined event list
    def __contains_event(token):
        with open(Path(__file__).parent / "features" / "events.json", 'r') as file:
            event_map = json.load(file)  # Load the event map from a JSON file
        return token in event_map.values()  # Check if the token is in the event list

    @staticmethod
    # Check if the token contains at least one capital letter
    def __has_capital_letter(token):
        return any(char.isupper() for char in token)

    @staticmethod
    # Check if the token consists entirely of capital letters
    def __has_only_capital_letters(token):
        return token.isupper()

    @staticmethod
    # Check if the token is the START token
    def __is_start(token):
        return token == 'START'

    @staticmethod
    # Check if the token is the END token
    def __is_end(token):
        return token == 'END'

    @staticmethod
    # Check if the token is a separator (whitespace, comma, hyphen, or dash)
    def __is_seperator(token):
        return token in [' ', ',', '-', '--', '—']

    @staticmethod
    # Check if the token contains punctuation (specifically a period)
    def __contains_punctuation(token):
        return '.' in token

    @staticmethod
    # Return the last 'amount' characters of the token
    def __get_the_last_characters(amount, token):
        return str(token)[-amount:]
