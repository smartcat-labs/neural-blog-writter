# encoding: utf-8
import utils.document as d
import tensorflow as tf
from nn.lstm_network import LSTMNetwork
import pickle as p
import numpy as np
import os
import sys
import json


class TextGenerator(object):

    def __init__(self, dataset_dir, model_dir):
        self._NETWORK_DIR = "lstm_model"
        self._VOCABLARY_NAME = "vocab.model"
        self._NETWORK_NAME = "model.ckp"
        self._METADATA_DIR = "metadata"
        self._NETWORK_METADATA = "network.model"

        if not os.path.exists(model_dir):
            raise ValueError('Not valid dataset directory')

        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self._metadata_dir = os.path.join(self.model_dir, self._METADATA_DIR)
        self._network_model = os.path.join(self.model_dir, self._NETWORK_DIR)
        self.vocabulary = None
        self.network_metadata = None

    def _save(self, saver, session):
        if not os.path.exists(self._metadata_dir):
            os.makedirs(self._metadata_dir)
        if not os.path.exists(self._network_model):
            os.makedirs(self._network_model)

        saver.save(session, os.path.join(self._network_model, self._NETWORK_NAME))
        with open(os.path.join(self._metadata_dir, self._VOCABLARY_NAME), 'w') as f:
            p.dump(self.vocabulary, f)

        with open(os.path.join(self._metadata_dir, self._NETWORK_METADATA), 'w') as f:
            p.dump(self.network_metadata, f)

    def _load(self, loader, session):
        loader.restore(session, os.path.join(self._network_model, self._NETWORK_NAME))
        with open(os.path.join(self._metadata_dir, self._VOCABLARY_NAME)) as f:
            self.vocabulary = p.load(f)

    def retrain(self, batch_size, batch_doc, epoch):
        dataset = d.BlogDataset(self.dataset_dir)
        self._load_network_metadata()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.Session(config=config)

        network = LSTMNetwork(self.network_metadata.in_out_size, self.network_metadata.num_layers,
                              self.network_metadata.lstm_size, session, self.network_metadata.learning_rate,
                              self.network_metadata.name)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        self._load(saver, session)

        last_costs = []
        costs = []
        for i in range(epoch):
            batch = dataset.next_batch(batch_doc)
            for mini_b in batch:
                reset_state = True
                for b in np.array_split(mini_b, mini_b.shape[0]/batch_size, axis=0):
                    b_x = np.array(b)
                    b_y = np.roll(b_x, -1, axis=0)
                    b_x = (b_x - dataset.mean) / ((1.0 * dataset.std) + 1.0e-6)
                    cost = network.train_step(b_x, b_y, reset_state)
                    costs.append(cost)
                    reset_state = False
            if i%5 == 0:
                print "Epoch: {}/{}, cost: {}".format(i, epoch, np.mean(np.array(costs)))
                last_costs = costs
                costs = list()

        print "Last epoch cost: {}".format(np.mean(np.array(last_costs)))
        self._save(saver, session)
        session.close()

    def train(self, batch_size, batch_doc, epoch, lstm_size, num_layers, learning_rate, name='lstm_rnn'):
        dataset = d.BlogDataset(self.dataset_dir)
        self.vocabulary = dataset.vocabulary
        in_out_size = len(self.vocabulary)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.Session(config=config)

        network = LSTMNetwork(in_out_size, num_layers,
                              lstm_size, session, learning_rate, name)

        self.network_metadata = network.get_metadata()

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        last_costs = []
        costs = []
        for i in range(epoch):
            batch = dataset.next_batch(batch_doc)
            for mini_b in batch:
                reset_state = True
                for b in np.array_split(mini_b, mini_b.shape[0]/batch_size, axis=0):
                    b_x = np.array(b)
                    b_y = np.roll(b_x, -1, axis=0)
                    b_x = (b_x - dataset.mean) / ((1.0 * dataset.std) + 1.0e-6)
                    cost = network.train_step(b_x, b_y, reset_state)
                    costs.append(cost)
                    reset_state = False
            if i%5 == 0:
                print "Epoch: {}/{}, cost: {}".format(i, epoch, np.mean(np.array(costs)))
                last_costs = costs
                costs = list()

        print "Last epoch cost: {}".format(np.mean(np.array(last_costs)))
        self._save(saver, session)
        session.close()

    def _load_network_metadata(self):
        with open(os.path.join(self._metadata_dir, self._NETWORK_METADATA)) as f:
            self.network_metadata = p.load(f)

    def generate(self,  output_file=None,  stop_letter=None, max_iterations=8000):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        session = tf.Session(config=config)

        self._load_network_metadata()

        network = LSTMNetwork(self.network_metadata.in_out_size, self.network_metadata.num_layers,
                              self.network_metadata.lstm_size, session, self.network_metadata.learning_rate,
                              self.network_metadata.name)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        self._load(saver, session)

        if not stop_letter:
            sw = d.DocumentModel.DOC_END
        else:
            sw = stop_letter

        history = []
        curr_p = network.run_step(((self.vocabulary.encode(d.DocumentModel.DOC_BEGIN) - self.vocabulary.mean)/((1.0 * self.vocabulary.std) + 1.0e-6)), True)
        current_letter = self.vocabulary.decode(np.random.choice(len(self.vocabulary), p=curr_p))
        history.append(current_letter)
        for i in range(max_iterations):
            if current_letter != sw:
                while True:
                    curr_p = network.run_step(((self.vocabulary.encode(current_letter) - self.vocabulary.mean)/((1.0 * self.vocabulary.std) + 1.0e-6)), False)
                    current_letter = self.vocabulary.decode(np.random.choice(len(self.vocabulary), p=curr_p))
                    if current_letter != d.DocumentModel.DOC_BEGIN:
                        break
                history.append(current_letter.decode("utf-8"))

        session.close()

        return "".join(history)


if __name__ == "__main__":
    conf = './conf.json'
    if len(sys.argv) > 1:
        conf = sys.argv[1]

    with open(conf) as f:
        conf_parsed = json.load(f)

    tf.reset_default_graph()
    if conf_parsed['command'].lower() == 'train':
        textGenerator = TextGenerator(conf_parsed['dataset_path'], conf_parsed['model_path'])
        textGenerator.train(conf_parsed['train_params']['batch_size'], conf_parsed['train_params']['docs_in_batch'],
                            conf_parsed['train_params']['epoch'],
                            conf_parsed['network_params']['lstm_size'], conf_parsed['network_params']['num_layers'],
                            conf_parsed['train_params']['learning_rate'],
                            conf_parsed['network_params']['name'])
    elif conf_parsed['command'].lower() == 'retrain':
        textGenerator = TextGenerator(conf_parsed['dataset_path'], conf_parsed['model_path'])
        textGenerator.retrain(conf_parsed['train_params']['batch_size'], conf_parsed['train_params']['docs_in_batch'],
                              conf_parsed['train_params']['epoch'])
    elif conf_parsed['command'].lower() == 'generate':
        textGenerator = TextGenerator(conf_parsed['dataset_path'], conf_parsed['model_path'])
        text = textGenerator.generate()
        with open(conf_parsed['generate']['output_dir'], 'a') as f:
            f.write(text)
    else:
        raise ValueError('Not recognized command {}'.format(conf_parsed['command']))

