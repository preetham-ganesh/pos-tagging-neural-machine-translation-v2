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
