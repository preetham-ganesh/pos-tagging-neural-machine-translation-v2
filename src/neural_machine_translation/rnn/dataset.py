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
