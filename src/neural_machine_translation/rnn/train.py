import os

import mlflow
import tensorflow as tf

from src.utils import load_json_file, check_directory_path_existence
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
        attention_type: str,
    ) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_name: A string for the name of the model.
            model_version: A string for the version of the current model.
            dataset_size: A string for the size of the dataset used for training the model.
            dataset_version: A string for the version of the dataset used for training the model.
            units: An integer for the no. of units in each RNN layer.
            n_rnn_blocks: An integer for the no. of RNN blocks in the Encoder & Decoder models.
            attention_type: A string for the type of attention in the Decoder model.

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
        assert isinstance(attention_type, int), "Variable attention_type of type 'str'."
        assert attention_type in [
            "luong",
            "bahdanau",
        ], "Variable attention_type should have value as 'luong' or 'bahdanau'."

        # Initalizes class variables.
        self.model_name = model_name
        self.model_version = model_version
        self.dataset_size = dataset_size
        self.dataset_version = dataset_version
        self.units = units
        self.n_blocks = n_blocks
        self.attention_type = attention_type
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
            self.home_directory_path, "configs", "neural_machine_translation", "rnn"
        )
        self.model_configuration = load_json_file(
            f"{self.dataset_size}_{self.units}", model_configuration_directory_path
        )

        # Updates model configuration with model version, dataset name & version.
        self.model_configuration["version"] = self.model_version
        self.model_configuration["language"] = (
            self.input_language
            if self.target_language == "en"
            else self.target_language
        )
        self.model_configuration["dataset"] = dict()
        self.model_configuration["dataset"]["name"] = self.dataset_name
        self.model_configuration["dataset"]["version"] = self.dataset_version
        self.model_configuration["model"]["n_blocks"] = self.n_blocks

        # Sets tag for architecture, model type, dataset version, input & target languages.
        mlflow.set_tag("dataset_version", f"v{self.dataset_version}")
        mlflow.set_tag("dataset_size", self.dataset_size)
        mlflow.set_tag("architecture", "rnn")

        # Logs parameters (n_layers & d_units).
        mlflow.log_param("attention_type", self.attention_type)
        mlflow.log_param("n_blocks", self.n_blocks)
        mlflow.log_param("units", self.units)

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
            len(self.dataset.ids_to_words[self.input_language]) + 1
        )
        self.model_configuration["model"]["target_vocab_size"] = (
            len(self.dataset.ids_to_words[self.target_language]) + 1
        )

    def load_model(self) -> None:
        """Loads model & other utilies based on model configuration.

        Loads model & other utilies based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Loads encoder model for current model configuration.
        self.encoder = Encoder(
            self.model_configuration["model"]["units"],
            self.model_configuration["model"]["input_vocab_size"],
            self.model_configuration["model"]["n_blocks"],
            self.model_configuration["model"]["rate"],
        )

        # Loads the decoder model based on attention type for current model configuration.
        if self.attention_type == "luong":
            self.decoder = LuongDecoder(
                self.model_configuration["model"]["units"],
                self.model_configuration["model"]["target_vocab_size"],
                self.model_configuration["model"]["n_blocks"],
                self.model_configuration["model"]["rate"],
            )
        else:
            self.decoder = BahdanauDecoder(
                self.model_configuration["model"]["units"],
                self.model_configuration["model"]["target_vocab_size"],
                self.model_configuration["model"]["n_blocks"],
                self.model_configuration["model"]["rate"],
            )

        # Builds plottable graph for the encoder & decoder models.
        self.encoder = self.encoder.build_graph()
        self.decoder = self.decoder.build_graph()

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["learning_rate"]
        )

        # Creates checkpoint manager for the neural network model.
        self.checkpoint_directory_path = os.path.join(
            self.home_directory_path,
            "models",
            "neural_machine_translation",
            self.model_name,
            f"v{self.model_version}",
            "checkpoints",
        )
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_directory_path, max_to_keep=1
        )
        print("Finished loading model for current configuration.")
        print()

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Compiles the encoder model to log the model summary.
        model_summary = list()
        self.encoder.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        print(model_summary)
        mlflow.log_text(model_summary, f"v{self.model_version}/encoder_summary.txt")

        # Compiles the encoder model to log the model summary.
        model_summary = list()
        self.decoder.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        print(model_summary)
        mlflow.log_text(model_summary, f"v{self.model_version}/decoder_summary.txt")

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            os.path.join(
                "models",
                "neural_machine_translation",
                self.model_name,
                f"v{self.model_version}",
                "reports",
            )
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            tf.keras.utils.plot_model(
                self.encoder,
                os.path.join(self.reports_directory_path, "encoder_plot.png"),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )
            tf.keras.utils.plot_model(
                self.decoder,
                os.path.join(self.reports_directory_path, "decoder_plot.png"),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )

            # Logs the saved model plot PNG file.
            mlflow.log_artifact(
                os.path.join(self.reports_directory_path, "encoder_plot.png"),
                f"v{self.model_version}",
            )
            mlflow.log_artifact(
                os.path.join(self.reports_directory_path, "decoder_plot.png"),
                f"v{self.model_version}",
            )

    def initialize_metric_trackers(self) -> None:
        """Initializes trackers which computes the mean of all metrics.

        Initializes trackers which computes the mean of all metrics.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.validation_accuracy = tf.keras.metrics.Mean(name="validation_accuracy")
