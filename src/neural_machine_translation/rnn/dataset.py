import os
import io

import sentencepiece as spm
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
        self.word_to_ids = {"en": dict(), model_configuration["language"]: dict()}
        self.ids_to_words = {"en": dict(), model_configuration["language"]: dict()}

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

    def train_tokenizer(self, language: str) -> None:
        """Trains SentencePiece tokenize on the text in the training split.

        Combines text in training split into a string, and converts it into text stream to train the SentencePiece
        tokenizer. Populates the words -> ids and ids -> words dictionary.

        Args:
            language: A string for the name of the language the text belongs to.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(language, str), "Variable language should be of type 'str'."
        assert language in [
            "en",
            "es",
            "de",
            "fr",
        ], "Variable language should have value as 'en', 'es', 'fr', or 'de'."
        assert isinstance(
            tokenizer_directory_path, str
        ), "Variable tokenizer_directory_path should be of type 'str'."

        # Checks if the following directory path exists.
        tokenizer_directory_path = check_directory_path_existence(
            os.path.join(
                "models",
                f"{self.input_language}-{self.target_language}",
                f"v{self.model_configuration['version']}",
                "tokenizer",
            )
        )

        # Combines text in train split.
        combined_text = ""
        for text in self.dataset_pairs["train"][language]:
            combined_text += text
            combined_text += "\n"

        # Trains the tokenizer for current language.
        with io.StringIO(combined_text) as text_stream:
            spm.SentencePieceTrainer.Train(
                sentence_iterator=text_stream,
                model_prefix=os.path.join(tokenizer_directory_path, language),
                vocab_size=self.model_configuration["tokenizer"]["language"][
                    "vocab_size"
                ],
                model_type=self.model_configuration["tokenizer"]["model_type"],
            )
        print(f"Finished training tokenizer for {language} language.")

        # Loads the trained tokenizer.
        self.tokenizer[language] = spm.SentencePieceProcessor(
            model_file=os.path.join(tokenizer_directory_path, f"{language}.model")
        )

        # Populates the words <-> ids dictionary for the current language.
        for id_0 in range(self.tokenizer[language].get_piece_size()):
            self.word_to_ids[language][self.tokenizer[language].id_to_piece(id_0)] = (
                id_0 + 1
            )
            self.ids_to_words[language][id_0 + 1] = self.tokenizer[
                language
            ].id_to_piece(id_0)
        print(f"Vocabulary size for {language}: {len(self.ids_to_words[language]) + 1}")
        print()

    def shuffle_slice_dataset(self) -> None:
        """Converts split data into tensor dataset & slices them based on batch size.

        Converts split data into input & target data. Zips the input & target data, and slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips images & classes into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.dataset_pairs["train"][self.input_language],
                self.dataset_pairs["train"][self.target_language],
            )
        )
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.dataset_pairs["validation"][self.input_language],
                self.dataset_pairs["validation"][self.target_language],
            )
        )
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.dataset_pairs["test"][self.input_language],
                self.dataset_pairs["test"][self.target_language],
            )
        )

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.batch_size = self.model_configuration["model"]["batch_size"]
        self.train_dataset = self.train_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.batch_size, drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = (
            len(self.dataset_pairs["train"][self.input_language]) // self.batch_size
        )
        self.n_validation_steps_per_epoch = (
            len(self.dataset_pairs["validation"][self.input_language])
            // self.batch_size
        )
        self.n_test_steps_per_epoch = (
            len(self.dataset_pairs["test"][self.input_language]) // self.batch_size
        )
        print(f"No. of train steps per epoch: {self.n_train_steps_per_epoch}")
        print(f"No. of validation steps per epoch: {self.n_validation_steps_per_epoch}")
        print(f"No. of test steps per epoch: {self.n_test_steps_per_epoch}")
        print()

        # Deletes dataset pairs from memory.
        del self.dataset_pairs

    def tokenize_text(self, text: str, language: str) -> List[int]:
        """Tokenizes text to convert into ids using trained SentencePiece tokenizer for the language.

        Tokenizes text to convert into pieces. Encodes pieces into unique ids, and adds start & end tokens.

        Args:
            text: A string for the text that should be tokenized.
            language: A string for the language the text belongs to.

        Returns:
            A list of integers for tokenized & encoded version of input text.
        """
        # Checks types & values of arguments.
        assert isinstance(text, str), "Variable text should be of type 'str'."
        assert isinstance(language, str), "Variable language should be of type 'str'."

        # Converts text into tokens based on tokenizer trained for the language.
        text_tokens = self.tokenizer[language].EncodeAsPieces(text)

        # Encodes tokens into ids.
        text_ids = [self.word_to_ids[language][token] for token in text_tokens]

        # Adds starting (<s>) & ending (</s>) tokens to the text ids.
        text_ids = (
            [self.word_to_ids[language]["<s>"]]
            + text_ids
            + [self.word_to_ids[language]["</s>"]]
        )
        return text_ids
