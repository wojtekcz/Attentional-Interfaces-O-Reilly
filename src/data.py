import pickle
import random
import os

from src.vocab import (
    Vocab,
    ids_to_tokens
)


def generate_batch(features, seq_lens, batch_size):
    """
    Generate batches of size <batch_size>.
    Args:
        features: processed contexts, questions and answers.
        seq_lens: context and question actual (pre-padding) seq-lens.
        batch_size: samples per each batch
    """
    data_size = len(features[0])
    num_batches = data_size//batch_size

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num+1)*batch_size, data_size)

        batch_features = []
        for feature in features:
            batch_features.append(feature[start_index:end_index])
        batch_lens = []
        for seq_len in seq_lens:
            batch_lens.append(seq_len[start_index:end_index])

        yield batch_features, batch_lens


def generate_epoch(data_path, num_epochs, batch_size):
    """
    Generate num_epoch epochs.
    Args:
        data_path: path for train.p|valid.p
        num_epochs: number of epochs to run for
        batch_size: samples per each batch
    """
    with open(data_path, 'rb') as f:
        entries = pickle.load(f)

    processed_contexts, processed_answers = [], []
    context_lens = []

    for entry in entries:
        processed_contexts.append(entry[0])
        context_lens.append(entry[1])
        processed_answers.append(entry[2])

    features = [processed_contexts, processed_answers]
    seq_lens = [context_lens,]

    for epoch_num in range(num_epochs):
        yield generate_batch(features, seq_lens, batch_size)


def sample_data(data_path, basedir, specified_index=None):
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

    print("==> Number of entries:", len(entries))
    print("==> Random index:", rand_index)
    print("==> Processed Review:", processed_review)
    print("==> Review Len:", review_seq_len)
    print("==> Label:", label)
    print("==> See if processed review makes sense:",
          ids_to_tokens(
              processed_review,
              vocab=vocab,
          ))
