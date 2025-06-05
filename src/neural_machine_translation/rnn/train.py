import os

from src.utils import load_json_file
from src.neural_machine_translation.rnn.dataset import Dataset
from src.neural_machine_translation.rnn.model import (
    Encoder,
    LuongDecoder,
    BahdanauDecoder,
)


class Train(object):
    """Trains the RNN-based Neural Machine Translation model based on the configuration."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
        dataset_size: str,
        dataset_version: str,
        units: int,
        n_blocks: int,
    ) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_name: A string for the name of the model.
            model_version: A string for the version of the current model.
            dataset_size: A string for the size of the dataset used for training the model.
            dataset_version: A string for the version of the dataset used for training the model.
            units: An integer for the no. of units in each RNN layer.
            n_blocks: An integer for the no. of blocks in the Encoder & Decoder models.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_name, str), "Variable model_version of type 'str'."
        assert isinstance(model_version, str), "Variable model_version of type 'str'."
        assert isinstance(dataset_size, str), "Variable dataset_size of type 'str'."
        assert dataset_size in [
            "mini",
            "full",
        ], "Argument dataset_size should have value as 'mini' or 'full'."
        assert isinstance(
            dataset_version, str
        ), "Variable dataset_version of type 'str'."
        assert isinstance(units, int), "Variable d_units of type 'int'."
        assert units in [
            512,
            1024,
        ], "Variable d_units should have value as 512 or 1024."
        assert isinstance(n_blocks, int), "Variable n_layers of type 'int'."
        assert 0 < n_blocks <= 4, "Variable n_layers should be between 1 & 4."

        # Initalizes class variables.
        self.model_name = model_name
        self.model_version = model_version
        self.dataset_size = dataset_size
        self.dataset_version = dataset_version
        self.units = units
        self.n_blocks = n_blocks
        self.best_validation_loss = None
        self.step = 0

        # Extracts input & target languages, and checks if they are valid.
        self.input_language, self.target_language = model_name.split("-")
        self.dataset_name = (
            f"{self.target_language}-en"
            if self.input_language == "en"
            else f"{self.input_language}-en"
        )
        assert "en" in [
            self.input_language,
            self.target_language,
        ], "Argument experiment_name should have 'en' as input or target language."

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for model version.

        Loads the model configuration file for model version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = os.path.join(
            self.home_directory_path,
            f"configs/neural_machine_translation/{self.input_language}-{self.target_language}",
        )
        self.model_configuration = load_json_file(
            f"v{self.model_version}", model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads the dataset based on model configuration.

        Loads the dataset based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Creates object attributes for the Dataset class.
        self.dataset = Dataset(
            self.model_configuration, self.input_language, self.target_language
        )

        # Loads text pairs for the specified dataset split (train, validation, test).
        self.dataset.load_data("train")
        self.dataset.load_data("validation")
        self.dataset.load_data("test")
        print()

        # Trains SentencePiece tokenize on the text in the training split.
        self.dataset.train_tokenizer(self.input_language)
        self.dataset.train_tokenizer(self.target_language)

        # Converts split data into tensor dataset & slices them based on batch size.
        self.dataset.shuffle_slice_dataset()

        # Updates input & target vocab size, and pe input & target in model configuration.
        self.model_configuration["model"]["input_vocab_size"] = (
            self.dataset.tokenizer[self.input_language].vocab_size + 1
        )
        self.model_configuration["model"]["target_vocab_size"] = (
            self.dataset.tokenizer[self.target_language].vocab_size + 1
        )
