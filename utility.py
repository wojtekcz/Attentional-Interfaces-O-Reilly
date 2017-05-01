"""
Utility functions
"""
import os
import argparse
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %pylab inline


"""
Preprocess the reviews.
"""
import argparse
import json
import os
import pickle
import re
import tensorflow as tf
import numpy as np

from unidecode import (
    unidecode,
)

from random import (
    shuffle,
)

from tqdm import (
    tqdm,
)

from collections import (
    Counter,
)

UNKNOWN_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'



class Vocab():
    """
    Class for processing tokens to ids and vice versa.
    """
    def __init__(self, vocab_file, max_vocab_size=200000, verbose=True):
        """
        """
        self.verbose = verbose
        self._token_to_id = {}
        self._id_to_token = {}
        self._size = -1

        with open(vocab_file, 'rt', encoding='utf-8') as f:
            for line in f:
                tokens = line.split()

                # White space in vocab file (' ': <count>)
                if len(tokens) == 1:
                    count = tokens[0]
                    idx = line.index(count)
                    t = line[:idx-1]
                    tokens = (t, count)

                if len(tokens) != 2:
                    continue

                if tokens[0] in self._token_to_id:
                    continue

                self._size += 1
                if self._size > max_vocab_size:
                    print ('Too many tokens! >%i/n' % max_vocab_size)
                    break

                self._token_to_id[tokens[0]] = self._size
                self._id_to_token[self._size] = tokens[0]

    def __len__(self):
        """
        Return vocabulary size.
        """
        return self._size+1

    def token_to_id(self, token):
        """
        Return the corresponding id for a token.
        """
        if token not in self._token_to_id:
            if self.verbose:
                print ("ID not found for %s" % token)
            return self._token_to_id[UNKNOWN_TOKEN]
        return self._token_to_id[token]

    def id_to_token(self, _id):
        """
        Returnn the correspoding token for an id.
        """
        if _id not in self._id_to_token:
            if self.verbose:
                print ("Token not found for ID: %i" % _id)
            return UNKNOWN_TOKEN
        return self._id_to_token[_id]

def ids_to_tokens(ids_list, vocab):
    """
    Convert a list of ids to tokens.
    Args:
        ids_list: list of ids to convert to tokens.
        vocab: Vocab class object.
    Returns:
        answer: list of tokens that corresponds to ids_list.
    """
    answer = []
    for _id in ids_list:
        token = vocab.id_to_token(_id)
        if token == PAD_TOKEN:
            continue
        answer.append(token)
    return answer



def sample_data(data_path, basedir):
    """
    Sample format of the processed
    data from data.py
    Args:
        data_path: path for train.p|valid.p
    """

    # global basedir

    with open(data_path, 'rb') as f:
        entries = pickle.load(f)

    # Choose a random sample
    rand_index = random.randint(0, len(entries))

    # Prepare vocab
    vocab_file = os.path.join(basedir, 'data/processed_reviews/vocab.txt')
    vocab = Vocab(vocab_file, verbose=False)

    # Sample
    (processed_review,
     review_seq_len,
     label) = entries[rand_index]

    print ("==> Number of entries:", len(entries))
    print ("==> Random index:", rand_index)
    print ("==> Processed Review:", processed_review)
    print ("==> Review Len:", review_seq_len)
    print ("==> Label:", label)
    print ("==> See if processed review makes sense:",
        ids_to_tokens(
            processed_review,
            vocab=vocab,
            ))