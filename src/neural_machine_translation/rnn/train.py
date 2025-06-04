import os

from src.utils import load_json_file


class Train(object):
    """Trains the RNN-based Neural Machine Translation model based on the configuration."""

    def __init__(
        self, model_version: str, input_language: str, target_language: str
    ) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.
            input_language: A string for the language of the input text.
            target_language: A string for the language of the target text.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."
        assert isinstance(input_language, str), "Variable input_language of type 'str'."
        assert isinstance(
            target_language, str
        ), "Variable target_language of type 'str'."

        # Initalizes class variables.
        self.model_version = model_version
        self.input_language = input_language
        self.target_language = target_language
        self.best_validation_loss = None

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
