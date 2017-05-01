import argparse
import os
import tensorflow as tf
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from tqdm import (
    tqdm,
)

from utility import *

def plot_attn(input_sentence, attentions, num_rows, save_loc=None):
    """
    Plot the attention scores.
    Args:
        input_sentence: input sentence (tokens) without <pad>
        attentions: attention scores for each token in input_sentence
        num_rows: how many rows you want the figure to have (we will add 1)
        save_loc: fig will be saved to this location
    """

    # Determine how many words per row
    words_per_row = (len(input_sentence.split(' '))//num_rows)

    # Use one extra row in case of remained for quotient above
    fig, axes = plt.subplots(nrows=num_rows+1, ncols=1, figsize=(20, 10))
    for row_num, ax in enumerate(axes.flat):

        # Isolate pertinent part of sentence and attention scores
        start_index = row_num*words_per_row
        end_index = (row_num*words_per_row)+words_per_row
        _input_sentence = \
            input_sentence.split(' ')[start_index:end_index]
        _attentions = np.reshape(
            attentions[0, start_index:end_index],
            (1, len(attentions[0, start_index:end_index]))
            )

        # Plot attn scores (constrained to [0.9, 1] for emphasis)
        im = ax.imshow(_attentions, cmap='Blues', vmin=0.9, vmax=1)

        # Set up axes
        ax.set_xticklabels(
            [''] + _input_sentence,
            rotation=90,
            minor=False,
            )
        ax.set_yticklabels([''])

        # Set x tick to top
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='black')

        # Show corresponding words at the ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Add color bar
    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])

    # display color bar
    cb = fig.colorbar(im, cax=cbar)
    cb.set_ticks([]) # clean color bar

    if save_loc is None:
        # Show the plot
        plt.show()
    else:
        # Save the plot
        fig.savefig(save_loc, dpi=fig.dpi, bbox_inches='tight') # dpi=fig.dpi for high res. save


def get_attn_inputs(FLAGS, review, review_len, raw_attn_scores, basedir):
    """
    Return the inputs needed to
    plot the attn scores. These include
    input_sentence and attn_scores.
    Args:
        FLAGS: parameters
        review: list of ids
        review_len: len of the relevant review
    Return:
        input_sentence: inputs as tokens (words) on len <review_len>
        plot_attn_scoes: (1, review_len) shaped scores
    """

    review_len = 300

    # Data paths
    vocab_path = os.path.join(
        basedir, 'data/processed_reviews/vocab.txt')
    vocab = Vocab(vocab_path)

    review = review[:review_len]
    attn_scores = raw_attn_scores[:review_len]

    # Process input_sentence
    input_sentence = ' '.join([item for item in ids_to_tokens(review, vocab)])

    # Process attn scores (normalize scores between [0,1])
    min_attn_score = min(attn_scores)
    max_attn_score = max(attn_scores)
    normalized_attn_scores = ((attn_scores - min_attn_score) / \
        (max_attn_score - min_attn_score))

    # Reshape attn scores for plotting
    plot_attn_scores = np.zeros((1, review_len))
    for i, score in enumerate(normalized_attn_scores):
        plot_attn_scores[0, i] = score

    return input_sentence, plot_attn_scores

def process_sample_attn(FLAGS, basedir):
    """
    Use plot_attn from utils.py to visualize
    the attention scores for a particular
    sample FLAGS.sample_num.
    """

    # Load the attn history
    attn_history_path = os.path.join(
        basedir, FLAGS.ckpt_dir, 'attn_history.p')
    with open(attn_history_path, 'rb') as f:
        attn_history = pickle.load(f)

    # Process the history to get the right sample
    sample = "sample_%i" % (FLAGS.sample_num)
    review_len = attn_history[sample]["review_len"]
    review = attn_history[sample]["review"]
    label = attn_history[sample]["label"]
    attn_scores = attn_history[sample]["attn_scores"][-1]

    input_sentence, plot_attn_scores = get_attn_inputs(
        FLAGS=FLAGS,
        review=review,
        review_len=review_len,
        raw_attn_scores=attn_scores,
        basedir=basedir
        )

    # Plot and save fig
    fig_name = "sample_%i" % (FLAGS.sample_num)
    save_loc = os.path.join(basedir, FLAGS.ckpt_dir, fig_name)
    plot_attn(
        input_sentence=input_sentence,
        attentions=plot_attn_scores,
        num_rows=FLAGS.num_rows,
        save_loc=None,
        )

