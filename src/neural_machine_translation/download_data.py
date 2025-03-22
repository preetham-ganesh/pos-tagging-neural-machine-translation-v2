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
