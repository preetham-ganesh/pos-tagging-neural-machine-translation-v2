import os
import sys
import warnings
import argparse
import time
import requests
import logging


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_PATH)
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import tensorflow_datasets as tfds

from src.utils import check_directory_path_existence


def check_language(language: str) -> None:
    """Checks if the language is valid or not.

    Checks if the language is valid or not.

    Args:
        language: A string for the language which needs to be checked.

    Returns:
        None.
    """
    # Asserts type & value of the arguments.
    assert isinstance(language, str), "Variable language should be of type 'str'."
    assert language in [
        "es",
        "fr",
        "de",
    ], "Variable language should have value as 'es', 'fr', or 'de'."


def download_tatoeba_dataset(language: str) -> None:
    """Downloads the Tatoeba dataset for the language given as input by user.

    Downloads the Tatoeba dataset for the language given as input by user.

    Args:
        language: A string for the language the Europarl dataset should be downloaded.

    Returns:
        None.
    """
    # Checks if the language is valid or not.
    check_language(language)

    # A dictionary for the language-based Europarl dataset links.
    dataset_links = {
        "fr": "https://www.manythings.org/anki/fra-eng.zip",
        "de": "https://www.manythings.org/anki/deu-eng.zip",
        "es": "https://www.manythings.org/anki/spa-eng.zip",
    }

    # Adds headers to the request.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        + "Chrome/91.0.4472.124 Safari/537.36"
    }

    # Download the compressed file.
    start_time = time.time()

    # Checks if the following directory path exists.
    dataset_directory_path = check_directory_path_existence(
        os.path.join("data", "raw_data", "tatoeba")
    )

    # Checks if the file already exists. If yes, then does not download the file.
    file_path = os.path.join(dataset_directory_path, f"{language}-en.zip")
    if os.path.exists(file_path):
        print(f"{language}-en.zip already exists.")

    else:
        # Sends request for the current language dataset file.
        response = requests.get(dataset_links[language], headers=headers)

        # Checks if the response has a success code. If not then prints the error message.
        assert response.status_code == 200, response.text

        # Saves the compressed file in the response as a .zip file.
        with open(file_path, "wb") as out_file:
            out_file.write(response.content)
        out_file.close()
        print(
            f"Finished downloading Tatoeba dataset for {language}-en in {(time.time() - start_time):.3f} sec."
        )
    print()


def download_europarl_dataset(language: str) -> None:
    """Downloads the Europarl dataset for the language given as input by user.

    Downloads the Europarl dataset for the language given as input by user.

    Args:
        language: A string for the language the Europarl dataset should be downloaded.

    Returns:
        None.
    """
    # Checks if the language is valid or not.
    check_language(language)

    # A dictionary for the language-based Europarl dataset links.
    dataset_links = {
        "fr": "https://www.statmt.org/europarl/v7/fr-en.tgz",
        "de": "https://www.statmt.org/europarl/v7/de-en.tgz",
        "es": "https://www.statmt.org/europarl/v7/es-en.tgz",
    }

    # Download the compressed file.
    start_time = time.time()

    # Checks if the following directory path exists.
    dataset_directory_path = check_directory_path_existence(
        os.path.join("data", "raw_data", "europarl")
    )

    # Checks if the file already exists. If yes, then does not download the file.
    file_path = os.path.join(dataset_directory_path, f"{language}-en.tgz")
    if os.path.exists(file_path):
        print(f"{language}-en.tgz already exists.")

    # Else, downloads it and saves it.
    else:
        # Sends request for the current language dataset file.
        response = requests.get(dataset_links[language])

        # Checks if the response has a success code. If not then prints the error message.
        assert response.status_code == 200, response.text

        # Saves the compressed file in the response as a .tgz file.
        with open(file_path, "wb") as out_file:
            out_file.write(response.content)
        out_file.close()

        print(
            f"Finished downloading Europarl dataset for {language}-en in {(time.time() - start_time):.3f} sec."
        )
    print()


def download_paracrawl_dataset(language: str) -> None:
    """Downloads the Paracrawl dataset for the language given as input by user.

    Downloads the Paracrawl dataset for the language given as input by user.

    Args:
        language: A string for the language the Europarl dataset should be downloaded.

    Returns:
        None.
    """
    # Checks if the language is valid or not.
    check_language(language)

    # Downloads paracrawl dataset into the corresponding directory for the current european language.
    start_time = time.time()
    _, _ = tfds.load(
        f"para_crawl/en{language}_plain_text".format(language),
        split="train",
        with_info=True,
        shuffle_files=True,
        data_dir=os.path.join("data", "raw_data", "paracrawl", f"{language}-en"),
    )
    print(
        f"Finished downloading paracrawl dataset for {language}-en in {(time.time() - start_time):.3f} sec."
    )
    print()
