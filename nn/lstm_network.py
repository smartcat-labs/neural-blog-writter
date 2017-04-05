import tensorflow as tf
import numpy as np


class NetworkMetadata(object):
    """
    This is used for save/restore model from file.
    Parameters
    num_layer : Number of LSTM units (layers)
    input_size : Number of input and output features (input_size = output_size)
    lstm_size : Number of units inside one LSTM
    learning_rate : Learning rate for training
    name : TensorFlow scope name
    """
    def __init__(self, num_layers, input_size, lstm_size, learning_rate, name):
        self.num_layers = num_layers
        self.in_out_size = input_size
        self.lstm_size = lstm_size
        self.learning_rate = learning_rate
        self.name = name


class LSTMNetwork(object):
    """
    Basic non-optimized multilayer LSTM neural network for generate
    Parameters
    inout_size : size of input feature vector which is size of output vector at same time
    num_layers : Number of LSTM units (layers)
    session : TensorFlow session
    learning_rate : Learning rate for training (currently AdamOptimizer)
    name : TensorFlow scope name
    """
    def __init__(self, inout_size,
                 num_layers, lstm_size,
                 session, learning_rate,
                 name='lstm_rnn'):
        self.scope = name
        self.session = session
        self.input_size = inout_size
        self.output_size = inout_size
        self.num_layers = num_layers
        self.lstm_size = lstm_size
        self.learning_rate = learning_rate

        # saved lstm state, (layers, 2 states (cell and hidden), batch, number_of_units)
        self.last_lstm_state = np.zeros((self.num_layers, 2, 1, self.lstm_size))

        with tf.variable_scope(self.scope, reuse=None):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.input_size], name="X")
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size], name="Y")

            # inital lstm state
            self.lstm_init_value = tf.placeholder(dtype=tf.float32, shape=[self.num_layers, 2, None, self.lstm_size],
                                                  name="STATE")
            # convert to tuple state
            self.tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(self.lstm_init_value[i, 0, :, :],
                                                                    self.lstm_init_value[i, 1, :, :]) for i in
                                      range(self.num_layers)])

            # make lstm unit cells
            self.lstm_cells = [tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True) for i in
                               range(self.num_layers)]
            # wire them together
            self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=True)

            # unroll lstm
            self.outputs, self.lstm_new_state = tf.nn.dynamic_rnn(self.lstm, self.X, initial_state=self.tuple_state,
                                                                  dtype=tf.float32)

            # fully connected layer
            W = tf.Variable(tf.random_normal((lstm_size, self.output_size), stddev=0.01))
            b = tf.Variable(tf.random_normal((self.output_size,), stddev=0.01))

            # get outputs from lstm and wire them with fc layer
            self.outputs_reshaped = tf.reshape(self.outputs, [-1, self.lstm_size])
            self.network_output = tf.matmul(self.outputs_reshaped, W) + b

            batch_time_shape = tf.shape(self.outputs)

            # final output is prob distribution for next character
            self.final_outputs = tf.reshape(tf.nn.softmax(self.network_output),
                                            (batch_time_shape[0], batch_time_shape[1], self.output_size))

            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.network_output, labels=self.Y))
            self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def get_metadata(self):
        """
        Make NetworkMetadata model of current configuration
        :return: NetworkMetadata
        """
        return NetworkMetadata(self.num_layers, self.input_size, self.lstm_size, self.learning_rate, self.scope)

    def train_step(self, x, y, reset_state=True):
        """
        One train step on batch. It can use current state of lstm or start from begining (zero state)
        :param x: ndarray of feature values
        :param y: ndarray of labels
        :param reset_state: Flag to reset state
        :return: cost on this batch
        """
        if reset_state:
            init_state = np.zeros((self.num_layers, 2, 1, self.lstm_size))
        else:
            init_state = self.last_lstm_state

        next_state, cost, _ = self.session.run([self.lstm_new_state, self.cost, self.train_op],
                                               feed_dict={self.X: [x], self.Y: y, self.lstm_init_value: init_state})

        self.last_lstm_state = next_state

        return cost

    def run_step(self, x, reset_state=True):
        """
        For given one character(one vector) generate following vector
        :param x: OHE input vector
        :param reset_state: Flag for reseting lstm state
        :return: probability distribution for following character
        """
        if reset_state:
            init_state = np.zeros((self.num_layers, 2, 1, self.lstm_size))
        else:
            init_state = self.last_lstm_state

        out, next_lstm_state = self.session.run([self.final_outputs, self.lstm_new_state],
                                                feed_dict={self.X: [[x]], self.lstm_init_value: init_state})

        self.last_lstm_state = next_lstm_state

        return out[0][0]
