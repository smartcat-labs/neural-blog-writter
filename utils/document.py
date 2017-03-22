# encode: utf-8
import numpy as np
import os
import multiprocessing
import string
import copy_reg
import types
import re
import random


def _pickle_method(m):
    """
    Used to avoid exception in multiprocessing with lambda functions
    :param m:
    :return:
    """
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class DocumentModel(object):
    """
    Loaded document representation
    """
    DOC_BEGIN = '\nBEGIN\n'
    DOC_END = '\nEND\n'

    def __init__(self, path, vocabulary):
        self._path = path
        self._vocab = vocabulary
        self._make_array()

    def _make_array(self):
        """
        Parse text file into list of characters and encode it to OHE
        :return: Returns 2d array [num_char, vocab_size]
        """
        doc = list()
        doc.append(self._vocab.encode(DocumentModel.DOC_BEGIN))
        with open(self._path) as f:
            for line in f:
                line = re.sub("\n+","\n",line)
                for c in line:
                    doc.append(self._vocab.encode(c))
        doc.append(self._vocab.encode(DocumentModel.DOC_END))
        self._document = np.array(doc)

    def get_numpy_array(self):
        """
        :return: Returns 2d array [num_char, vocab_size]
        """
        return self._document

    def get_path(self):
        return self._path


class Vocabulary(object):
    """
    Vocabulary of dataset, contains all characters that are in dataset. They are used for encoding and decoding
    """
    def __init__(self, predefined=None):
        if predefined:
            self._vocabulary = list(predefined)
        else:
            self._vocabulary = list()
        self.mean = None
        self.std = None

    def add(self, item):
        """
        Adding new character to vocabulary, check if character exists in vocabulary
        :param item: new character
        :return: None
        """
        if item not in self._vocabulary:
            self._vocabulary.append(item)

    def __len__(self):
        return len(self._vocabulary)

    def get_vocabulary(self):
        return self._vocabulary

    def encode(self, item):
        """
        OHE one character
        :param item: character
        :return: encoded character
        """
        zeros = np.zeros(len(self._vocabulary))
        if item not in self._vocabulary:
            return zeros
        else:
            zeros[self._vocabulary.index(item)] = 1.0
            return zeros

    def decode(self, index):
        """
        Decoding to character
        :param index: index of character
        :return: decoded character
        """
        return self._vocabulary[index]


class BlogDataset(object):
    """
    All dataset (TODO: batch is optimized)
    """
    def __init__(self, dataset_path):
        self._dataset_path = dataset_path
        self.vocabulary = self._make_vocabulary()
        self._dataset = self._make_documents()
        self._mean_norm()

    def _vocab_helper(self, path):
        """
        Read file, convert to chars, decode it to utf-8 and add to vocabulary
        :param path:
        :return:
        """
        vocab = Vocabulary()
        with open(os.path.join(self._dataset_path, path)) as f:
            vocab.add(DocumentModel.DOC_BEGIN)
            for line in f:
                for c in line:
                    try:
                        char = c.decode("utf-8", errors='strict')
                        vocab.add(c)
                    except:
                        continue
            vocab.add(DocumentModel.DOC_END)
        return vocab

    def _make_vocabulary(self):
        """
        Read files and make vocabulary in threads, them reduce them in single vocab
        :return: Vocabulary of dataset
        """
        dummy_vocab = Vocabulary(list(string.ascii_letters + string.digits))
        pool = multiprocessing.Pool()
        vocabs = pool.map(self._vocab_helper, os.listdir(self._dataset_path))
        final_vocab = reduce(lambda a, b: Vocabulary(set(a.get_vocabulary() + b.get_vocabulary())), vocabs, dummy_vocab)
        pool.close()
        return final_vocab

    def _doc_helper(self, path):
        """
        Encode document in array using vocabulary
        :param path:
        :return: DocumentModel
        """
        return DocumentModel(os.path.join(self._dataset_path,path), self.vocabulary)

    def _make_documents(self):
        """
        Read docs in threads and encode them in ndarrays
        :return:
        """
        pool = multiprocessing.Pool()
        docs = pool.map(self._doc_helper, os.listdir(self._dataset_path))
        pool.close()
        return np.array(docs)

    def _mean_norm(self):
        """
        Calculate mean and std of dataset. First it will group all dataset in single ndarray.
        :return: None
        """
        self.single_array = self._dataset[0].get_numpy_array()
        for doc in self._dataset[1:]:
            self.single_array = np.append(self.single_array, doc.get_numpy_array(), axis=0)
        self.mean = np.mean(self.single_array, axis=0)
        self.std = np.std(self.single_array, axis=0)
        self.vocabulary.mean = self.mean
        self.vocabulary.std = self.std

    def get_dataset(self):
        return self._dataset

    def next_batch(self, batch_size):
        """
        Random select batch from documents. Future work: genrator
        :param batch_size: number of documents to take for batch
        :return: ndarray of documents
        """
        choice = np.random.choice(range(len(self._dataset)), batch_size)
        return np.array([i.get_numpy_array() for i in self._dataset[choice]])

    def batch_from_fat(self, batch_size):
        index = random.randint(0, self.single_array.shape[0] - batch_size - 1)
        return (self.single_array[index:index + batch_size, :] - self.mean)/self.std
