import tensorflow as tf
import numpy as np


class LSTMNetwork(object):

    def __init__(self, vocab, n_cells, n_layers, session):
        self.n_cells = n_cells
        self.n_layers = n_layers
        self.n_chars = len(vocab)
        self.vocab = vocab
        self.session = session

        self.X = tf.placeholder(tf.int32, [None, None], name='X')

        # We'll have a placeholder for our true outputs
        self.Y = tf.placeholder(tf.int32, [None, None], name='Y')

        # inital lstm state
        self.lstm_init_value = tf.placeholder(dtype=tf.float32, shape=[n_layers, 2, None, n_cells],
                                              name="STATE")
        # convert to tuple state
        self.tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(self.lstm_init_value[i, 0, :, :],
                                                                self.lstm_init_value[i, 1, :, :]) for i in
                                  range(n_layers)])

        # we first create a variable to take us from our one-hot representation to our LSTM cells
        embedding = tf.get_variable("embedding", [self.n_chars, n_cells], tf.float32)

        # And then use tensorflow's embedding lookup to look up the ids in X
        Xs = tf.nn.embedding_lookup(embedding, self.X)

        cells = tf.contrib.rnn.BasicLSTMCell(num_units=n_cells, state_is_tuple=True)
        cells = tf.contrib.rnn.MultiRNNCell(
            [cells] * n_layers, state_is_tuple=True)

        outputs, self.last_state = tf.nn.dynamic_rnn(cell=cells, inputs=Xs, initial_state=self.tuple_state, scope='rnnlm')
        outputs_flat = tf.reshape(outputs, [tf.shape(outputs)[0] * tf.shape(outputs)[1], n_cells])
        W = tf.get_variable(
            "W",
            shape=[n_cells, self.n_chars],
            initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable(
            "b",
            shape=[self.n_chars],
            initializer=tf.random_normal_initializer(stddev=0.1))

        # Find the output prediction of every single character in our minibatch
        # we denote the pre-activation prediction, logits.
        logits = tf.matmul(outputs_flat, W) + b

        # We get the probabilistic version by calculating the softmax of this
        self.probs = tf.nn.softmax(logits)

        # And then we can find the index of maximum probability
        Y_pred = tf.argmax(self.probs, 1)

        # Compute mean cross entropy loss for each output.
        Y_true_flat = tf.reshape(self.Y, shape=[tf.shape(self.X)[1] * tf.shape(self.X)[0]])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y_true_flat)
        self.mean_loss = tf.reduce_mean(loss)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        gradients = []
        clip = tf.constant(5.0, name="clip")
        for grad, var in optimizer.compute_gradients(self.mean_loss):
            gradients.append((tf.clip_by_value(grad, -clip, clip), var))
        self.updates = optimizer.apply_gradients(gradients)

    def train_step(self, x, y, reset_state=True):
        #if reset_state:
        init_state = np.zeros((self.n_layers, 2, len(x), self.n_cells))
        #else:
        #    init_state = self.last_lstm_state

        next_state, cost, _ = self.session.run([self.last_state, self.mean_loss, self.updates],
                                               feed_dict={self.X: x, self.Y: y, self.lstm_init_value: init_state})

        return cost

    def run_step(self, x, reset_state=True):
        if reset_state:
            init_state = np.zeros((self.n_layers, 2, len(x), self.n_cells))
        else:
            init_state = self.last_lstm_state

        out, next_lstm_state = self.session.run([self.probs, self.last_state],
                                                feed_dict={self.X: [x], self.lstm_init_value: init_state})

        self.last_lstm_state = next_lstm_state

        return out[0]


if __name__ == "__main__":
    tf.reset_default_graph()
    root_file = '/home/stanko/SmartCat/blog-generator/ghost-dataset-raw'
    import os

    lyrics = os.listdir(root_file)
    txt = []
    for f in lyrics:
        with open(os.path.join(root_file, f)) as g:
            txt += g.read()

    vocab = list(set(txt))

    print(np.array(vocab).shape)

    encoder = dict(zip(vocab, range(len(vocab))))
    decoder = dict(zip(range(len(vocab)), vocab))

    sess = tf.Session()
    net = LSTMNetwork(vocab, 256, 2, sess)

    init = tf.global_variables_initializer()
    sess.run(init)
    cursor = 0
    it_i = 0
    sequence_length = 150
    batch_size = 150
    num_iter = 4001
    start = [encoder[" "]]
    while it_i < num_iter:
        Xs, Ys = [], []
        for batch_i in range(batch_size):
            if (cursor + sequence_length) >= len(txt) - sequence_length - 1:
                cursor = 0
            Xs.append([encoder[ch]
                       for ch in txt[cursor:cursor + sequence_length]])
            Ys.append([encoder[ch]
                       for ch in txt[cursor + 1: cursor + sequence_length + 1]])

            cursor = (cursor + sequence_length)
        Xs = np.array(Xs).astype(np.int32)
        Ys = np.array(Ys).astype(np.int32)

        cost = net.train_step(Xs, Ys, True)
        print(it_i, cost)

        if it_i%800 == 0:
            print("PREDICTION")
            probs = decoder[np.argmax(net.run_step(start, reset_state=True))]
            prediction = [probs]
            print(prediction)
            for i in range(500):
                prediction += decoder[np.argmax(net.run_step([encoder[prediction[-1]]], reset_state=False))]

            print("".join(prediction))

        it_i += 1

    sess.close()