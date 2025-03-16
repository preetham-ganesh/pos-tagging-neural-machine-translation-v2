import tensorflow as tf

from typing import Dict, Any


class Encoder(tf.keras.Model):
    """"""

    def __init__(self, units: int, vocab_size: int, n_layers: int, rate: float):
        """Initializes the layers in the Encoder model, by adding various layers.

        Initializes the layers in the Encoder model, by adding various layers.

        Args:
            units: An integer for the size of each layer in the model.
            vocab_size: An integer for the size of the vocabulary in the input language.
            n_layers: An integer for the no. of RNN layers in the model.
            rate: A floating point value for the dropout rate in the model.

        Returns:
            None.
        """
        super(Encoder, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(units, int), "Variable units should be of type 'int'."
        assert isinstance(
            vocab_size, int
        ), "Variable vocab_size should be of type 'int'."
        assert isinstance(n_layers, int), "Variable n_layers should be of type 'int'."
        assert (
            n_layers % 2 == 0 and n_layers > 0
        ), "Variable n_layers should be divisible by 2 and greater than 0."
        assert isinstance(rate, float), "Variable rate should be of type 'float'."

        # Initializes class variables.
        self.units = units
        self.n_layers = n_layers
        self.model_layers = dict()

        # Initializes bidirectional RNN block.
        self.model_layers["embedding"] = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=units, name="embedding_0"
        )
        self.model_layers["rnn_fwd"] = tf.keras.layers.LSTM(
            units=units, return_state=True, return_sequence=True, name="rnn_fwd"
        )
        self.model_layers["rnn_bwd"] = tf.keras.layers.LSTM(
            units=units,
            return_state=True,
            return_sequence=True,
            go_backwards=True,
            name="rnn_bwd",
        )
        self.model_layers["bi_rnn"] = tf.keras.layers.Bidirectional(
            layer=self.model_layers["rnn_fwd"],
            backward_layer=self.model_layers["rnn_bwd"],
            merge_mode="concat",
            name="bi_rnn",
        )
        self.model_layers["dropout_0"] = tf.keras.layers.Dropout(
            rate=rate, name="dropout_0"
        )
        self.model_layers["dropout_1"] = tf.keras.layers.Dropout(
            rate=rate, name="droptout_1"
        )

        # Initializes RNN blocks.
        l_id = 2
        while l_id < n_layers:
            self.model_layers[f"rnn_{l_id}"] = tf.keras.layers.LSTM(
                units=units,
                return_state=True,
                return_sequence=True,
                name=f"rnn_{l_id}",
            )
            self.model_layers[f"rnn_{l_id + 1}"] = tf.keras.layers.LSTM(
                units=units,
                return_state=True,
                return_sequence=True,
                name=f"rnn_{l_id + 1}",
            )
            self.model_layers[f"add_{l_id - 2}"] = tf.keras.layers.Add(
                name=f"add_{l_id - 2}"
            )
            self.model_layers[f"add_{l_id - 1}"] = tf.keras.layers.Add(
                name=f"add_{l_id - 1}"
            )
            self.model_layers[f"dropout_{l_id}"] = tf.keras.layers.Dropout(
                rate=rate, name=f"dropout_{l_id}"
            )
            self.model_layers[f"dropout_{l_id}"] = tf.keras.layers.Dropout(
                rate=rate, name=f"dropout_{l_id}"
            )
            l_id += 2
