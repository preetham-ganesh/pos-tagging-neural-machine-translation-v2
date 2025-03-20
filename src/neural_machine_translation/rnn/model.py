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
        assert (
            units % 2 == 0
        ), "Variable units should be divisible by 2, and greater than 0."
        assert isinstance(
            vocab_size, int
        ), "Variable vocab_size should be of type 'int'."
        assert vocab_size > 0, "Variable vocab_size should be greater than 0."
        assert isinstance(n_layers, int), "Variable n_layers should be of type 'int'."
        assert (
            n_layers % 2 == 0 and n_layers > 0
        ), "Variable n_layers should be divisible by 2 and greater than 0."
        assert isinstance(rate, float), "Variable rate should be of type 'float'."
        assert 0 <= rate <= 1, "Variable rate should be between 0 & 1."

        # Initializes class variables.
        self.units = units
        self.n_layers = n_layers
        self.model_layers = dict()

        # Initializes bidirectional RNN block.
        self.model_layers["embedding_0"] = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=units, name="embedding_0"
        )
        self.model_layers["rnn_fwd"] = tf.keras.layers.LSTM(
            units=units // 2, return_state=True, return_sequences=True, name="rnn_fwd"
        )
        self.model_layers["rnn_bwd"] = tf.keras.layers.LSTM(
            units=units // 2,
            return_state=True,
            return_sequences=True,
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
            rate=rate, name="dropout_1"
        )
        self.model_layers["concat_0"] = tf.keras.layers.Concatenate(
            axis=-1, name="concat_0"
        )
        self.model_layers["concat_1"] = tf.keras.layers.Concatenate(
            axis=-1, name="concat_1"
        )

        # Initializes RNN blocks.
        l_id = 2
        while l_id < n_layers:
            self.model_layers[f"rnn_{l_id}"] = tf.keras.layers.LSTM(
                units=units,
                return_state=True,
                return_sequences=True,
                name=f"rnn_{l_id}",
            )
            self.model_layers[f"rnn_{l_id + 1}"] = tf.keras.layers.LSTM(
                units=units,
                return_state=True,
                return_sequences=True,
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
        """Inputs are passed through the layers in the model.

        Inputs are passed through the layers in the model.

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
        (
            x,
            forward_memory_state,
            forward_carry_state,
            backward_memory_state,
            backward_carry_state,
        ) = inputs
        x = self.model_layers["embedding_0"](x)
        (
            x,
            forward_memory_state,
            forward_carry_state,
            backward_memory_state,
            backward_carry_state,
        ) = self.model_layers["bi_rnn"](
            x,
            initial_state=[
                forward_memory_state,
                forward_carry_state,
                backward_memory_state,
                backward_carry_state,
            ],
        )
        memory_state = self.model_layers["concat_0"](
            [forward_memory_state, backward_memory_state]
        )
        carry_state = self.model_layers["concat_1"](
            [forward_carry_state, backward_carry_state]
        )
        del (
            forward_memory_state,
            forward_carry_state,
            backward_memory_state,
            backward_carry_state,
        )
        x = self.model_layers["dropout_0"](x)
        memory_state = self.model_layers["dropout_1"](memory_state)
        carry_state = self.model_layers["dropout_1"](carry_state)

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
            carry_state = self.model_layers[f"dropout_{l_id + 1}"](carry_state)
            l_id += 2
        return [x, memory_state, carry_state]

    def initialize_hidden_states(self, batch_size: int) -> List[tf.Tensor]:
        """Initializes hidden states for the bidirectional RNN in the model.

        Initializes hidden states for the bidirectional RNN in the model.

        Args:
            batch_size: An integer for the size of current batch.

        Returns:
            A list of tensors for bidirectional RNN's hidden states of shape (batch_size, units / 2).
        """
        # Asserts type & values of the input arguments.
        assert isinstance(
            batch_size, int
        ), "Variable batch_size should be of type 'int'."

        # Returns initial states for forward & backward memory, and forward & backward carry states.
        return [
            tf.zeros(batch_size, self.units // 2),
            tf.zeros(batch_size, self.units // 2),
            tf.zeros(batch_size, self.units // 2),
            tf.zeros(batch_size, self.units // 2),
        ]

    def build_graph(self) -> tf.keras.Model:
        """Builds plottable graph for the model.

        Builds plottable graph for the model.

        Args:
            None.

        Returns:
            A tensorflow model based on image height, width & n_channels in the model configuration.
        """
        # Creates the input layer using the model configuration.
        inputs = [
            tf.keras.layers.Input(shape=(None,)),
            tf.keras.layers.Input(shape=(self.units // 2,)),
            tf.keras.layers.Input(shape=(self.units // 2,)),
            tf.keras.layers.Input(shape=(self.units // 2,)),
            tf.keras.layers.Input(shape=(self.units // 2,)),
        ]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs, False, None))


class BahdanauAttention(tf.keras.layers.Layer):
    """Implements Bahdanau Attention mechanism by computing context vector using Encoder output & hidden states."""

    def __init__(self, units: int) -> None:
        """Initializes components in the BahdanauAttention layer in the model.

        Initializes components in the BahdanauAttention layer in the model.

        Args:
            units: An integer for the no. of units in the layer.

        Returns:
            None.
        """
        super(BahdanauAttention, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(units, int), "Variable units should be of type 'int'."

        # Initializes class variables.
        self.dense_0 = tf.keras.layers.Dense(units=units)
        self.dense_1 = tf.keras.layers.Dense(units=units)
        self.dense_2 = tf.keras.layers.Dense(units=units)
        self.dense_3 = tf.keras.layers.Dense(units=units)
        self.reshape_0 = tf.keras.layers.Reshape(target_shape=(1, -1))
        self.add_0 = tf.keras.layers.Add()
        self.multiply_0 = tf.keras.layers.Multiply()

    def call(
        self, encoder_out: tf.Tensor, memory_state: tf.Tensor, carry_state: tf.Tensor
    ) -> tf.Tensor:
        """Inputs are passed through components in the layer.

        Inputs are passed through components in the layer.

        Args:
            encoder_out: A tensor for the output from the encoder.
            memory_state: A tensor for the memory state from the last RNN layer in encoder.
            carry_state: A tensor for the carry state from the last RNN layer in encoder.

        Returns:
            A tensor for the context vector computed using Bahdanau Attention.
        """
        # Reshapes memory & carry state to add axis for time.
        memory_time = self.reshape_0(memory_state)
        carry_time = self.reshape_0(carry_state)

        # Computes attention score using encoder output, reshaped memory & carry states.
        attention_score = self.dense_3(
            tf.keras.activations.tanh(
                self.add_0(
                    [
                        self.dense_0(encoder_out),
                        self.dense_1(memory_time),
                        self.dense_2(carry_time),
                    ]
                )
            )
        )
        attention_weights = tf.keras.activations.softmax(attention_score, axis=-1)

        # Computes context vector using attention weights & encoder out.
        context_vector = self.multiply_0([attention_weights, encoder_out])
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


class BahdanauDecoder(tf.keras.Model):
    """"""

    def __init__(
        self, units: int, vocab_size: int, n_rnn_blocks: int, rate: float
    ) -> None:
        """Initializes the layers in the BahdanauDecoder model, by adding various layers.

        Initializes the layers in the BahdanauDecoder model, by adding various layers.

        Args:
            units: An integer for the size of each layer in the model.
            vocab_size: An integer for the size of the vocabulary in the input language.
            n_rnn_blocks: An integer for the no. of RNN blocks in the model (where each block would have 2 RNN layers).
            rate: A floating point value for the dropout rate in the model.

        Returns:
            None.
        """
        super(BahdanauDecoder, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(units, int), "Variable units should be of type 'int'."
        assert (
            units % 2 == 0
        ), "Variable units should be divisible by 2, and greater than 0."
        assert isinstance(
            vocab_size, int
        ), "Variable vocab_size should be of type 'int'."
        assert vocab_size > 0, "Variable vocab_size should be greater than 0."
        assert isinstance(
            n_rnn_blocks, int
        ), "Variable n_rnn_blocks should be of type 'int'."
        assert n_rnn_blocks > 0, "Variable n_rnn_blocks should be greater than 0."
        assert isinstance(rate, float), "Variable rate should be of type 'float'."
        assert 0 <= rate <= 1, "Variable rate should be between 0 & 1."

        # Initializes class variables.
        self.units = units
        self.n_rnn_blocks = n_rnn_blocks
        self.model_layers = dict()

        # Initializes embedding, bahdanau attention, reshape & concatenate layers.
        self.model_layers["block_0_embedding_0"] = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=units, name="block_0_embedding_0"
        )
        self.model_layers["block_0_attention_0"] = BahdanauAttention(units)
        self.model_layers["block_0_concat_0"] = tf.keras.layers.Concatenate(
            axis=-1, name="block_0_concat_0"
        )
        self.model_layers["block_0_reshape_0"] = tf.keras.layers.Reshape(
            target_shape=(1, -1), name="block_0_reshape_0"
        )

        # Initializes the RNN & droptout layers.
        self.model_layers["block_1_rnn_0"] = tf.keras.layers.LSTM(
            units=units, return_state=True, return_sequences=True, name="block_1_rnn_0"
        )
        self.model_layers["block_1_dropout_0"] = tf.keras.layers.Dropout(
            rate=rate, name="block_1_dropout_0"
        )
        self.model_layers["block_1_dropout_1"] = tf.keras.layers.Dropout(
            rate=rate, name="block_1_dropout_1"
        )
        self.model_layers["block_1_rnn_1"] = tf.keras.layers.LSTM(
            units=units, return_state=True, return_sequences=True, name="block_1_rnn_1"
        )
        self.model_layers["block_1_dropout_2"] = tf.keras.layers.Dropout(
            rate=rate, name="block_1_dropout_2"
        )
        self.model_layers["block_1_dropout_3"] = tf.keras.layers.Dropout(
            rate=rate, name="block_1_dropout_3"
        )

        # Initializes RNN blocks.
        b_id = 1
        while b_id < n_rnn_blocks:
            self.model_layers[f"block_{b_id}_rnn_0"] = tf.keras.layers.LSTM(
                units=units,
                return_state=True,
                return_sequences=True,
                name=f"block_{b_id}_rnn_0",
            )
            self.model_layers[f"block_{b_id}_rnn_1"] = tf.keras.layers.LSTM(
                units=units,
                return_state=True,
                return_sequences=True,
                name=f"block_{b_id}_rnn_1",
            )
            self.model_layers[f"block_{b_id}_add_0"] = tf.keras.layers.Add(
                name=f"block_{b_id}_add_0"
            )
            self.model_layers[f"block_{b_id}_add_1"] = tf.keras.layers.Add(
                name=f"block_{b_id}_add_1"
            )
            self.model_layers[f"block_{b_id}_dropout_0"] = tf.keras.layers.Dropout(
                rate=rate, name=f"block_{b_id}_dropout_0"
            )
            self.model_layers[f"block_{b_id}_dropout_1"] = tf.keras.layers.Dropout(
                rate=rate, name=f"block_{b_id}_dropout_1"
            )
            b_id += 1
