import os
import sys
import warnings
import argparse
import logging
import unicodedata
import re
import zipfile
import tarfile


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


from bs4 import BeautifulSoup
import pandas as pd
import tensorflow_datasets as tfds
from sklearn.utils import shuffle
import nltk


from src.utils import check_directory_path_existence, save_text_file, load_text_file
from src.neural_machine_translation.download_data import check_language

from typing import List


# Downloads Punkt tab module.
nltk.download("punkt_tab")


class PreprocessDataset(object):
    """This class supports preprocessing datasets in multiple languages (English, Spanish, French, German,
    and Italian). It handles text cleaning, named entity preservation, out-of-vocabulary (OOV) handling,
    sentence tokenization, and dataset splitting."""

    def __init__(
        self,
        language: str,
        dataset_size: str,
        n_max_words_per_text: int,
        dataset_version: str,
    ) -> None:
        """Creates object attributes for the PreprocessDataset class.

        Creates object attributes for the PreprocessDataset class.

        Args:
            language: A string for the name of the european language the text belongs to.
            dataset_size: A string for the size of the dataset. Accepted values 'mini' or 'full'.
            n_max_words_per_text: An integer for the maximum no. of the words in a text.
            dataset_version: A string for the version by which the processed dataset should be saved as.

        Returns:
            None.
        """
        # Asserts type & values of the arguments.
        check_language(language)
        assert isinstance(
            dataset_size, str
        ), "Variable dataset_size should be of type 'str'."
        assert dataset_size in [
            "mini",
            "full",
        ], "Variable dataset_size should be 'mini' or 'full'."
        assert isinstance(
            n_max_words_per_text, int
        ), "Variable n_max_words_per_text should be of type 'int'."

        # Initializes class variables.
        self.language = language
        self.dataset_size = dataset_size
        self.n_max_words_per_text = 50
        self.unique_words_count = {"en": dict(), language: dict()}
        self.rare_words = {"en": set(), self.language: set()}
        self.processed_texts = list()
        self.dataset_version = dataset_version

    def extract_tatoeba_dataset(self) -> None:
        """Extracts the Tatoeba dataset for the language given as input by user.

        Extracts the Tatoeba dataset for the language given as input by user.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory path for downloaded data zip file.
        zip_file_path = os.path.join(
            BASE_PATH, "data", "raw_data", "tatoeba", f"{self.language}-en.zip"
        )

        # Creates the directory path.
        extracted_data_directory_path = check_directory_path_existence(
            os.path.join("data", "extracted_data", "tatoeba", f"{self.language}-en")
        )

        # A dictionary for the name of the text files in each language.
        text_name = {"fr": "fra.txt", "de": "deu.txt", "es": "spa.txt"}

        # If file does not exist, then extracts files from the directory.
        if not os.path.exists(
            os.path.join(extracted_data_directory_path, text_name[self.language])
        ):

            # Extracts files from downloaded data zip file into a directory.
            try:
                with zipfile.ZipFile(zip_file_path, "r") as zip_file:
                    zip_file.extractall(extracted_data_directory_path)
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"{zip_file_path} does not exist. Run 'download_data.py' to download the data."
                )
            print(
                f"Finished extracting files from '{self.language}-en.zip' to {extracted_data_directory_path}."
            )
            print()
