import warnings
import time


# Establish basedir (useful if running as python package)
basedir = ""

# Hide all warning messages
warnings.filterwarnings('ignore')


from src.data import (
    sample_data
)

from src.train import (
    train
)


class Parameters:
    """
    Arguments for data processing.
    """
    def __init__(self):
        """
        """
        self.data_dir = "data/processed_reviews/train.p"           # location of reviews data (train|validation)

FLAGS = Parameters()
sample_data(FLAGS.data_dir, basedir)


class Parameters:
    """
    Arguments for data processing.
    """
    def __init__(self):
        """
        """
        self.data_dir="data/processed_reviews"           # location of reviews data
        self.ckpt_dir="output/wcz-6"                     # location of model checkpoints
        self.model_name="imdb_model"                     # Name of the model
        self.mode="train"                                # train|infer
        self.model="new"                                 # old|new
        self.lr=1e-4                                     # learning rate
        self.num_epochs=1                                # num of epochs
        self.batch_size=256                              # batch size
        self.hidden_size=200                             # num hidden units for RNN
        self.embedding="glove"                           # random|glove
        self.emb_size=200                                # num hidden units for embeddings
        self.max_grad_norm=5                             # max gradient norm
        self.keep_prob=0.9                               # Keep prob for dropout layers
        self.num_layers=1                                # number of layers for recursion
        self.max_input_length=300                        # max number of words per review
        self.min_lr=1e-6                                 # minimum learning rate
        self.decay_rate=0.96                             # Decay rate for lr per global step (train batch)
        self.save_every=1                                # Save the model every <save_every> epochs


start_time = time.time()

FLAGS = Parameters()
train(FLAGS, basedir)

print('--- %s seconds ---' % (time.time() - start_time))
