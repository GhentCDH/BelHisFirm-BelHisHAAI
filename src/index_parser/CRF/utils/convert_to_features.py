import json
from functools import partial
from pathlib import Path
import unicodedata
import regex as re

# This class handles the conversion of strings to feature representations 
# for training a CRF (Conditional Random Field) model.
# The code has been reworked from city-directory-entry-parser.
class Convert_To_Features:

    @staticmethod
    def text_normalisation(text: str) -> str:
        text = unicodedata.normalize("NFC", text)

        text = text.replace("\u200B", "")
        text = text.replace("\u00A0", " ")

        text = re.sub(r"[’‘ʼʹ´`]", "'", text)
        text = re.sub(r"[“”„‟«»]", '"', text)

        text = re.sub(r"[–—―]", " — ", text)
        text = re.sub(r"[‐-‒−]", "-", text)

        text = text.replace("…", "...")
        text = re.sub(r"[·•●]", ".", text)

        text = re.sub(r"\s+([.,;:!?°])", r"\1", text)
        text = re.sub(r"([.,;:!?])(?=\S)", r"\1 ", text)

        text = re.sub(r"\s*—\s*", " — ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[_\s]+", "", text)

        return text

    @staticmethod
    def tokenize_string(
        string: str,
        labels: list[str] | None = None,
    ) -> list[str] | tuple[list[str], list[str]]:

        token_pattern = re.compile(r"""
            \p{L}+(?:['’]\p{L}+)?   # words, including d'associés
        | \d+                     # numbers
        | [—-]                    # dash / hyphen
        | [.,;:!?()°/&"]           # punctuation
        """, re.VERBOSE)

        delimiters = {".", ",", ";", ":", "!", "?", "(", ")", "°", "/", "&", '"', "—", "-"}

        normalized_string = Convert_To_Features.text_normalisation(string)
        tokens = token_pattern.findall(normalized_string)

        tokens = ["START"] + tokens + ["END"]

        if labels is None:
            return tokens

        new_labels = ["START"]
        label_index = 0

        for token in tokens[1:-1]:
            if token in delimiters:
                new_labels.append("D")
            else:
                new_labels.append(labels[label_index])
                label_index += 1

        new_labels.append("END")

        if label_index != len(labels):
            raise ValueError(
                f"Label/token mismatch: used {label_index} labels, "
                f"but received {len(labels)} labels."
            )

        return tokens, new_labels

    @staticmethod
    # Extracts features for tokens based on neighboring tokens in the list.
    def generate_token_features(tokens):
        features = []
        # Iterate through each token in the list and gather features from neighboring tokens
        for index, token in enumerate(tokens):
            token_features = {}
            for rela_pos in range(-2, 3):  # Check for tokens in the range of -2 to 2 relative positions
                shuffle_pos = index + rela_pos
                if 0 <= shuffle_pos < len(tokens):  # Ensure the relative position is within bounds
                    relative_token = tokens[shuffle_pos]
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
        with open(Path(__file__).parent.parent / "features" / "events.json", 'r') as file:
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
