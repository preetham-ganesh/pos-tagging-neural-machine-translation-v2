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

    def preprocess_tatoeba_dataset(self) -> None:
        """Preprocesses the Tatoeba dataset for the language given as input by user.

        Preprocesses the Tatoeba dataset for the language given as input by user.

        Args:
            None.

        Returns:
            None.
        """
        # A dictionary for the name of the text files in each language.
        text_name = {"fr": "fra.txt", "de": "deu.txt", "es": "spa.txt"}

        # Loads the Tatoeba dataset for the language given as input by user.
        data = pd.read_csv(
            os.path.join(
                BASE_PATH,
                "data",
                "extracted_data",
                "tatoeba",
                f"{self.language}-en",
                text_name[self.language],
            ),
            sep="\t",
            encoding="utf-8",
            names=["en", self.language, "x"],
        )
        print(
            f"No. of original {self.language}-en pairs in Tatoeba dataset: {len(data)}"
        )
        print()

        # Iterates across rows in the dataset.
        n_processed_pairs = 0
        for id_0, row in data.iterrows():

            # Splits text into sentences using NLTK.
            en_sentences = self.split_text_into_sentences(str(row["en"]), "en")
            eu_sentences = self.split_text_into_sentences(
                str(row[self.language]), self.language
            )

            # If no. of sentences in the english & european text are not equal, then skips it.
            if len(en_sentences) != len(eu_sentences):
                continue

            # Iterates across sentence pairs in the text.
            for en_text, eu_text in zip(en_sentences, eu_sentences):

                # Preprocesses the text in the dataset.
                processed_en_text = self.preprocess_text(en_text, "en", True)
                processed_eu_text = self.preprocess_text(eu_text, self.language, True)

                # If text is not empty, then it is appended to list.
                if processed_en_text != "" and processed_eu_text != "":
                    self.processed_texts.append(
                        {
                            "en": processed_en_text,
                            self.language: processed_eu_text,
                            "dataset": "tatoeba",
                        }
                    )
                    n_processed_pairs += 1

            if id_0 % 1000 == 0:
                print(
                    f"Finished processing {((id_0 / len(data)) * 100):.3f}% {self.language}-en pairs in Tatoeba "
                    + "dataset."
                )

            # If dataset size is mini, then only 10% of the dataset is processed.
            if self.dataset_size == "mini" and n_processed_pairs >= int(
                len(data) * 0.01
            ):
                break

        # Deletes data variable.
        del data

        print()
        print(
            f"No. of processed {self.language}-en pairs in Tatoeba dataset: {n_processed_pairs}"
        )
        print()

    def preprocess_europarl_dataset(self) -> None:
        """Preprocesses the Europarl dataset for the language given as input by user.

        Preprocesses the Europarl dataset for the language given as input by user.

        Args:
            None.

        Returns:
            None.
        """
        # Loads the Europarl dataset for the language given as input by user.
        original_en_texts = load_text_file(
            f"europarl-v7.{self.language}-en.en",
            os.path.join(
                BASE_PATH, "data", "extracted_data", "europarl", f"{self.language}-en"
            ),
        ).split("\n")
        original_eu_texts = load_text_file(
            f"europarl-v7.{self.language}-en.{self.language}",
            os.path.join(
                BASE_PATH, "data", "extracted_data", "europarl", f"{self.language}-en"
            ),
        ).split("\n")

        # Checks if the length of both the lists are equal.
        assert len(original_en_texts) == len(
            original_eu_texts
        ), f"Length of en and {self.language} texts should be equal."
        print(
            f"No. of original {self.language}-en pairs in Europarl dataset: {len(original_en_texts)}"
        )
        print()

        # Iterates across rows in the dataset.
        n_processed_pairs = 0
        for id_0 in range(len(original_en_texts)):

            # Splits text into sentences using NLTK.
            en_sentences = self.split_text_into_sentences(original_en_texts[id_0], "en")
            eu_sentences = self.split_text_into_sentences(
                original_eu_texts[id_0], self.language
            )

            # If no. of sentences in the english & european text are not equal, then skips it.
            if len(en_sentences) != len(eu_sentences):
                continue

            # Iterates across sentence pairs in the text.
            for en_text, eu_text in zip(en_sentences, eu_sentences):

                # Preprocesses the text in the dataset.
                processed_en_text = self.preprocess_text(en_text, "en", True)
                processed_eu_text = self.preprocess_text(eu_text, self.language, True)

                # If text is not empty, then it is appended to list.
                if processed_en_text != "" and processed_eu_text != "":
                    self.processed_texts.append(
                        {
                            "en": processed_en_text,
                            self.language: processed_eu_text,
                            "dataset": "europarl",
                        }
                    )
                    n_processed_pairs += 1

            if id_0 % 1000 == 0:
                print(
                    f"Finished processing {((id_0 / len(original_en_texts)) * 100):.3f}% {self. language}-en pairs in "
                    + "Europarl dataset."
                )

            # If dataset size is mini, then only 10% of the dataset is processed.
            if self.dataset_size == "mini" and n_processed_pairs >= int(
                len(original_en_texts) * 0.01
            ):
                break

        # Deletes original_en_texts & original_eu_texts variables.
        del original_en_texts, original_eu_texts
        print()
        print(
            f"No. of processed {self.language}-en pairs in Europarl dataset: {n_processed_pairs}"
        )
        print()

    def preprocess_paracrawl_dataset(self) -> None:
        """Preprocesses the Paracrawl dataset for the language given as input by user.

        Preprocesses the Paracrawl dataset for the language given as input by user.

        Args:
            None.

        Returns:
            None.
        """
        # Loads the Paracrawl dataset for the language given as input by user.
        dataset, info = tfds.load(
            f"para_crawl/en{self.language}_plain_text",
            split="train",
            with_info=True,
            shuffle_files=True,
            data_dir=os.path.join(
                BASE_PATH, "data", "raw_data", "paracrawl", f"{self.language}-en"
            ),
        )
        n_rows = info.splits["train"].num_examples
        print(
            f"No. of original {self.language}-en pairs in Paracrawl dataset: {n_rows}"
        )

        # Iterates across rows in the dataset.
        n_processed_pairs = 0
        for id_0, row in enumerate(dataset):

            # Splits text into sentences using NLTK.
            en_sentences = self.split_text_into_sentences(
                row["en"].numpy().decode("utf-8"), "en"
            )
            eu_sentences = self.split_text_into_sentences(
                row[self.language].numpy().decode("utf-8"), self.language
            )

            # If no. of sentences in the english & european text are not equal, then skips it.
            if len(en_sentences) != len(eu_sentences):
                continue

            # Iterates across sentence pairs in the text.
            for en_text, eu_text in zip(en_sentences, eu_sentences):

                # Preprocesses the text in the dataset.
                processed_en_text = self.preprocess_text(en_text, "en", True)
                processed_eu_text = self.preprocess_text(eu_text, self.language, True)

                # If text is not empty, then it is appended to list.
                if processed_en_text != "" and processed_eu_text != "":
                    self.processed_texts.append(
                        {
                            "en": processed_en_text,
                            self.language: processed_eu_text,
                            "dataset": "paracrawl",
                        }
                    )
                    n_processed_pairs += 1

            if id_0 % 1000 == 0:
                print(
                    f"Finished processing {((id_0 / n_rows) * 100):.3f}% {self.language}-en pairs in Paracrawl dataset."
                )

            # If dataset size is mini, then only 10% of the dataset is processed.
            if self.dataset_size == "mini" and n_processed_pairs >= int(n_rows * 0.01):
                break

        # Deletes dataset variable.
        del dataset
        print()
        print(
            f"No. of processed {self.language}-en pairs in Paracrawl dataset: {n_processed_pairs}"
        )
        print()

    def identify_language_rare_words(self, language: str) -> None:
        """Identifies rare words for the language in the dataset.

        Identifies rare words for the language in the dataset.

        Args:
            language: A string for the language the rare words should be identified for.

        Returns:
            None.
        """
        print(
            f"No. of unique words for {language} language in the dataset: {len(self.unique_words_count[language])}"
        )

        # Iterates across unique words in the dataset based on language.
        for word in self.unique_words_count[language].keys():
            # If count of word is 1, word consists of only alphabets, and length of word is between 8 & 10,
            # then word is added to rare words list.
            if (
                self.unique_words_count[language][word] == 1
                and word.isalpha()
                and not word.isdigit()
            ):
                self.rare_words[language].add(word)
        print(
            f"No. of rare words for {language} language in the dataset: {len(self.rare_words[language])}"
        )
        print()

    def identify_common_rare_words(self) -> None:
        """Identifies common rare words between en & the european language.

        Identifies common rare words between en & the european language.

        Args:
            None.

        Returns:
            None.
        """
        # Identifies rare words for each language in the dataset.
        self.identify_language_rare_words("en")
        self.identify_language_rare_words(self.language)

        # Identifies common rare words between en & the european language.
        self.common_rare_words = set(
            self.rare_words[self.language] & self.rare_words["en"]
        )
        print(
            f"No. of common rare words between en & {self.language} texts: {len(self.common_rare_words)}"
        )
        print()

    def oov_handling_per_pair(self, en_text: str, eu_text: str) -> List[str]:
        """Handles out-of-vocabulary words in the text pairs.

        Handles out-of-vocabulary words in the text pairs. Converts rare words into unique format.

        Args:
            en_text: A string for the English text.
            eu_text: A string for the European text.

        Returns:
            A list of strings for the English & European text after handling out-of-vocabulary words.
        """
        # Asserts type & values of the arguments.
        assert isinstance(en_text, str), "Variable en_text should be of type 'str'."
        assert isinstance(eu_text, str), "Variable eu_text should be of type 'str'."

        # Converts text into unique set of words.
        en_words = set(en_text.split(" "))
        eu_words = set(eu_text.split(" "))

        # Identifies common list of rare words between en & eu text.
        common_words = list(en_words & self.common_rare_words & eu_words)

        # Iterates across words in common words list.
        unk_count = 0
        for word in common_words:
            if f" {word} " in en_text and f" {word} " in eu_text:
                en_text = en_text.replace(f" {word} ", f" <unk{unk_count}> ")
                eu_text = eu_text.replace(f" {word} ", f" <unk{unk_count}> ")
                unk_count += 1
        return [en_text, eu_text]

    def oov_handling(self) -> None:
        """Handles out-of-vocabulary words for all text pairs in the dataset.

        Handles out-of-vocabulary words for all text pairs in the dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Identifies common rare words between en & the european language.
        self.identify_common_rare_words()

        # Iterates across text pairs in the dataset.
        n_oov_handled_pairs = 0
        self.oov_handled_texts = list()
        for id_0, row in enumerate(self.processed_texts):

            # Handles out-of-vocabulary words in the text pairs.
            en_text, eu_text = self.oov_handling_per_pair(row["en"], row[self.language])

            # If text is not empty, then it is appended to list.
            if en_text != "" and eu_text != "":
                self.oov_handled_texts.append({"en": en_text, self.language: eu_text})
                n_oov_handled_pairs += 1

            if id_0 % 1000 == 0:
                print(
                    f"Finished OOV handling {round((id_0 / len(self.processed_texts)) * 100, 3)}% {self.language}-en"
                    + " pairs in the dataset."
                )

            # If dataset size is mini, then only 10% of the dataset is processed.
            if (
                self.dataset_size == "mini"
                and n_oov_handled_pairs // len(self.processed_texts) == 0.1
            ):
                break

        # Deletes processed texts variable.
        del self.processed_texts
        print()
        print(
            f"No. of OOV handled {self.language}-en pairs in the dataset: {n_oov_handled_pairs}"
        )
        print()
