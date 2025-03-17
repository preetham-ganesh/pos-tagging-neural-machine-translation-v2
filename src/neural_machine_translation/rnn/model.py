import tensorflow as tf

from typing import List, Any


class Encoder(tf.keras.Model):
    """"""

    def __init__(self, units: int, vocab_size: int, n_layers: int, rate: float) -> None:
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
        self.model_layers["embedding_0"] = tf.keras.layers.Embedding(
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
            self.model_layers[f"dropout_{l_id + 1}"] = tf.keras.layers.Dropout(
                rate=rate, name=f"dropout_{l_id + 1}"
            )
            l_id += 2

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = False,
        masks: List[tf.Tensor] = None,
    ) -> List[tf.Tensor]:
        """Input tensor is passed through the layers in the model.

        Input tensor is passed through the layers in the model.

        Args:
            inputs: A list for the inputs from the input batch.
            training: A boolean value for the flag of training/testing state.
            masks: A tensor for the masks from the input batch.

        Returns:
            A tensor for the processed output from the components in the layer.
        """
        # Asserts type & values of the input arguments.
        assert isinstance(inputs, list), "Variable inputs should be of type 'list'."
        assert isinstance(training, bool), "Variable training should be of type 'bool'."
        assert (
            isinstance(masks, list) or masks is None
        ), "Variable masks should be of type 'list' or masks should have value as 'None'."

        # Passes input through bidirectional RNN block.
        x, memory_state, carry_state = inputs[0]
        x = self.model_layers["embedding_0"](x)
        x, memory_state, carry_state = self.model_layers["bi_rnn"](
            x, initial_state=[memory_state, carry_state]
        )
        x = self.model_layers["dropout_0"](x)
        memory_state = self.model_layers["dropout_1"](memory_state)
        carry_state = self.model_layers["dropout_1"](memory_state)

        # Passes inputs through RNN blocks.
        l_id = 2
        while l_id < self.n_layers:
            x_, memory_state_, carry_state_ = self.model_layers[f"rnn_{l_id}"](
                x, initial_state=[memory_state, carry_state]
            )
            x = self.model_layers[f"add_{l_id - 2}"]([x, x_])
            memory_state = self.model_layers[f"add_{l_id - 1}"](
                [memory_state, memory_state_]
            )
            carry_state = self.model_layers[f"add_{l_id - 1}"](
                [carry_state, carry_state_]
            )
            del x_, memory_state_, carry_state_
            x, memory_state, carry_state = self.model_layers[f"rnn_{l_id + 1}"](
                x, initial_state=[memory_state, carry_state]
            )
            x = self.model_layers[f"dropout_{l_id}"](x)
            memory_state = self.model_layers[f"dropout_{l_id + 1}"](memory_state)
            carry_state = self.model_layers[f"dropout_{l_id + 1}"](memory_state)
        return [x, memory_state, carry_state]
