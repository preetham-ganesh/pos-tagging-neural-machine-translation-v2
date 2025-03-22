import os

import tensorflow_datasets as tfds
import tensorflow as tf

from src.utils import check_directory_path_existence, load_text_file

from typing import Dict, Any, List


class Dataset(object):
    """Loads the dataset based on the model configuration."""

    def __init__(
        self,
        model_configuration: Dict[str, Any],
        input_language: str,
        target_language: str,
    ) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.
            input_language: A string for the name of input language.
            target_language: A string for the name of target language.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration
        self.input_language = input_language
        self.target_language = target_language
        self.dataset_pairs = {"train": list(), "validation": list(), "test": list()}
        self.tokenizer = dict()

    def load_data(self, split_name: str) -> None:
        """Loads text pairs for the specified dataset split (train, validation, test).

        Loads text pairs for the specified dataset split (train, validation, test) for all available chunks.

        Args:
            split_name: A string for the name of the split the text belongs to.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            split_name, str
        ), "Variable split_name should be of type 'str'."
        assert split_name in [
            "train",
            "validation",
            "test",
        ], "Variable split_name should have value as 'train', 'validation' or 'test'."

        # Checks if the following directory path exists.
        base_directory_path = check_directory_path_existence(
            os.path.join(
                "data",
                "processed_data",
                f"{self.model_configuration['language']}-en",
                f"v{self.model_configuration['dataset']['version']}",
            )
        )

        # Creates empty dictionary to store the text pairs in current split.
        self.dataset_pairs[split_name] = {
            "en": list(),
            self.model_configuration["language"]: list(),
        }

        # If split name is 'train', then loads all training data chunks.
        if split_name == "train":
            s_id = 0
            while True:
                # If file path does not exist, then exits the loop.
                if not os.path.exists(
                    os.path.join(
                        base_directory_path,
                        f"train_{s_id}.{self.model_configuration['language']}",
                    )
                ):
                    break
                # Loads current split id's en & eu text as lines.
                en_lines = load_text_file(
                    f"train_{s_id}.en", base_directory_path
                ).split("\n")
                eu_lines = load_text_file(
                    f"train_{s_id}.{self.model_configuration['language']}",
                    base_directory_path,
                ).split("\n")

                # Iterates across loaded English & European lines, and adds them to the dataset if not empty.
                for en_text, eu_text in zip(en_lines, eu_lines):
                    if en_text != "" and eu_text != "":
                        self.dataset_pairs["train"]["en"].append(en_text)
                        self.dataset_pairs["train"][
                            self.model_configuration["language"]
                        ].append(eu_text)
                s_id += 1

        # Else, loads validation & test split files.
        else:
            # Loads current split's en & eu text as lines.
            en_lines = load_text_file(f"{split_name}.en", base_directory_path).split(
                "\n"
            )
            eu_lines = load_text_file(
                f"{split_name}.{self.model_configuration['language']}",
                base_directory_path,
            ).split("\n")

            # Iterates across loaded English & European lines, and adds them to the dataset if not empty.
            for en_text, eu_text in zip(en_lines, eu_lines):
                if en_text != "" and eu_text != "":
                    self.dataset_pairs["train"]["en"].append(en_text)
                    self.dataset_pairs["train"][
                        self.model_configuration["language"]
                    ].append(eu_text)
        print(
            f"No. of text pairs in the {split_name} split: {len(self.dataset_pairs[split_name]['en'])}"
        )
