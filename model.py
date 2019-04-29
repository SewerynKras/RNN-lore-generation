import tensorflow as tf


class MyModel(tf.keras.Model):
    """
    Model architecture:

    CudnnLSTM
    dropout
    relu
    CudnnLSTM
    dropout
    relu
    Dense
    softmax

    Arguments:
        LSTM1_units {int} -- num_units for the first LSTM
        LSTM2_units {int} -- num_units for the second LSTM
        output_size {int} -- num_units for the output dense layer
    """
    def __init__(self, LSTM1_units, LSTM2_units, output_size):
        super(MyModel, self).__init__()
        self.LSTM1 = tf.keras.layers.LSTM(LSTM1_units,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          recurrent_dropout=0,
                                          unroll=False,
                                          use_bias=True,
                                          recurrent_initializer='glorot_uniform',
                                          return_sequences=True,
                                          return_state=True)
        self.LSTM1_state = None
        self.LSTM2 = tf.keras.layers.LSTM(LSTM2_units,
                                          activation='tanh',
                                          recurrent_activation='sigmoid',
                                          recurrent_dropout=0,
                                          unroll=False,
                                          use_bias=True,
                                          recurrent_initializer='glorot_uniform',
                                          return_sequences=True,
                                          return_state=True)
        self.LSTM2_state = None
        self.OUTPUT = tf.keras.layers.Dense(output_size)

    def call(self, inputs, dropout_rate=0.0, remember=False):
        """
        Arguments:
            inputs {numpy.array} -- shape (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SHAPE)

        Keyword Arguments:
            dropout_rate {float} -- rate of both dropout functions
                (amount of units turned off) (default: {0.0})
            remember {bool} -- if true the LSTM1 and LSTM2 states will be
                saved and used during the next call,
                to override this call .forget() (default: {False})
        """
        state = self.LSTM1_state
        x, *state = self.LSTM1(inputs, initial_state=state)
        self.LSTM1_state = state

        x = tf.nn.dropout(x, rate=dropout_rate)
        x = tf.nn.relu(x)

        state = self.LSTM2_state
        x, *state = self.LSTM2(x, initial_state=state)
        self.LSTM2_state = state

        x = tf.nn.dropout(x, rate=dropout_rate)
        x = tf.nn.relu(x)

        x = self.OUTPUT(x)
        x = tf.nn.softmax(x)

        if not remember:
            self.forget()

        return x

    def forget(self):
        """
        Deletes any remembered states
        """
        self.LSTM1_state = None
        self.LSTM2_state = None
