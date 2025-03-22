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
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

    def extract_europarl_dataset(self) -> None:
        """Extracts the Europarl dataset for the language given as input by user.

        Extracts the Europarl dataset for the language given as input by user.

        Args:
            None.

        Returns:
            None.
        """
        # Creates absolute directory path for downloaded data tar file.
        tar_file_path = os.path.join(
            BASE_PATH, "data", "raw_data", "europarl", f"{self.language}-en.tgz"
        )

        # Creates the directory path.
        extracted_data_directory_path = check_directory_path_existence(
            os.path.join("data", "extracted_data", "europarl", f"{self.language}-en")
        )

        # If file does not exist, then extracts files from the directory.
        if not os.path.exists(
            os.path.join(
                extracted_data_directory_path,
                f"europarl-v7.{self.language}-en.{self.language}",
            )
        ):
            # Extracts files from downloaded data tar file into a directory.
            try:
                file = tarfile.open(tar_file_path)
                file.extractall(extracted_data_directory_path)
                file.close()
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"{tar_file_path} does not exist. Run 'download_data.py' to download the data."
                )
            print(
                f"Finished extracting files from '{self.language}-en.tgz' to {extracted_data_directory_path}."
            )
            print()

    def remove_html_markup(self, text: str) -> str:
        """Removes HTML markup components from text.

        Removes HTML markup components from text.

        Args:
            text: A string for the text which needs to be processed.

        Returns:
            A string for the processed text without HTML markup components.
        """
        # Asserts type & values of the arguments.
        assert isinstance(text, str), "Variable text should be of type 'str'."

        # Creates an object for BeautifulSoup.
        soup = BeautifulSoup(text, "lxml")

        # Get the text content of all visible elements.
        text = soup.get_text(strip=True)

        # Remove any leading or trailing spaces.
        text = text.strip()

        # Replace consecutive whitespace characters with a single space.
        return " ".join(text.split())

    def preprocess_text(self, text: str, language: str, update_word_count: bool) -> str:
        """Preprocesses text to remove unwanted characters from it.

        Preprocesses text to remove unwanted characters from it.

        Args:
            text: A string for the text which needs to be processed.
            language: A string for the language the text belongs to.
            update_unique_words: A boolean value for updating the word count.

        Returns:
            A string for processed version of input text.
        """
        # Asserts type & values of the arguments.
        assert isinstance(text, str), "Variable text should be of type 'str'."
        assert isinstance(
            update_word_count, bool
        ), "Variable update_word_count should be of type 'bool'."

        # Removes HTML markup components from text provided as input.
        text = self.remove_html_markup(text)

        # Strip leading & trailing whitespace.
        text = text.strip()

        # Replaces unwanted characters in text.
        text = text.replace("##at##-##at##", "-")
        text = text.replace("&apos;", "'")
        text = text.replace("&quot;", '"')
        text = text.replace("&#91;", "")
        text = text.replace("&#93;", "")
        text = text.replace("&#124;", "")
        text = text.replace('"', ' " ')

        # Splits text into list of words as strings, and ignores empty words.
        text_words = [word for word in text.split(" ") if word not in ["", " "]]

        # If no. of words in current text is more than maximum limit, then text is ignored.
        if len(text_words) > self.n_max_words_per_text:
            return ""

        # Adds spaces before & after special characters for all languages.
        special_chars = "-!$&(),./%:;?€'\""
        for char in special_chars:
            text = text.replace(char, f" {char} ")

        # Converts all characters in text to lowercase.
        text = text.lower()

        # Based on name of the language, removes characters from text.
        if language == "en":
            text = "".join(
                id_0
                for id_0 in unicodedata.normalize("NFKD", str(text))
                if unicodedata.category(id_0) != "Mn"
            )
            text = re.sub(r"[^-!$&(),./%0-9:;?a-z€'\"]+", " ", text)

        elif language == "es":
            text = re.sub(r"[^-!$&(),./%0-9:;?áéíóúñü¿¡a-z€'\"]+", " ", text)

        elif language == "fr":
            text = re.sub(r"[^-!$&(),./%0-9:;?!çàâæéèêëîïôöûüù'€\"*a-z]+", " ", text)

        elif language == "de":
            text = re.sub(r"[^-!$&(),./%0-9:;?!äöüßœáéíóúñüa-z'€\"*]+", " ", text)

        # Splits text into list of words as strings.
        text_words = text.split(" ")

        # Iterates across words in text.
        filtered_words = list()
        for word in text_words:
            # If word is not empty, then it is appended to list.
            if word != "":
                filtered_words.append(word)

                # Unique word count is updated for current word.
                if update_word_count:
                    self.unique_words_count[language][word] = (
                        1 + self.unique_words_count[language].get(word, 0)
                    )

        # Converts of list of filtered words into a single string.
        filtered_text = " ".join(filtered_words)
        return filtered_text

    def split_text_into_sentences(self, text: str, language: str) -> List[str]:
        """Splits text into sentences using NLTK.

        Splits text into sentences using NLTK.

        Args:
            text: A string for the text that needs to be processed.
            language: A string for the name of the language the text belongs to.

        Returns:
            A list of strings for sentences extracted from the text.
        """
        # Creates a dictionary to store language full names.
        supported_languages = {
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
        }

        # Splits text into sentences.
        return nltk.tokenize.sent_tokenize(text, language=supported_languages[language])
