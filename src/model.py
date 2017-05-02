"""
Simple GRU Encoder/Decoder Model w/ Attentional Interface
"""
import tensorflow as tf
import numpy as np


from src.operation_functions import (
    _linear,
    custom_GRUCell,
    add_dropout_and_layers
)


class Model:
    """
    Tensorflow graph.
    """
    def __init__(self, FLAGS, vocab_size):
        """
        """
        self.FLAGS = FLAGS
        self._vsize = vocab_size

    def train(self, sess, batch_reviews, batch_labels,
        batch_review_lens, embeddings, keep_prob):
        """
        Train the model using a batch and predicted guesses.
        """
        outputs = [
            self._train_op,
            self._logits,
            self._loss,
            self._accuracy,
            self._lr,
            self._Z,
                   ]
        inputs = {
            self._reviews: batch_reviews,
            self._labels: batch_labels,
            self._review_lens: batch_review_lens,
            self._embeddings: embeddings,
            self._keep_prob: keep_prob,
                  }
        return sess.run(outputs, inputs)

    def eval(self, sess, batch_reviews, batch_labels,
        batch_review_lens, embeddings, keep_prob=1.0):
        """
        Evaluation of validation set.
        """
        outputs = [
            self._logits,
            self._loss,
            self._accuracy,
        ]
        inputs = {
            self._reviews: batch_reviews,
            self._labels: batch_labels,
            self._review_lens: batch_review_lens,
            self._embeddings: embeddings,
            self._keep_prob: keep_prob,
            }
        return sess.run(outputs, inputs)

    def infer(self, sess, batch_reviews,
        batch_review_lens, embeddings, keep_prob=1.0):
        """
        Inference with a sample sentence.
        """
        outputs = [
            self._logits,
            self._probabilities,
            self._Z,
        ]
        inputs = {
            self._reviews: batch_reviews,
            self._review_lens: batch_review_lens,
            self._embeddings: embeddings,
            self._keep_prob: keep_prob,
            }
        return sess.run(outputs, inputs)

    def _add_placeholders(self):
        """
        Input that will be fed into our DCN graph.
        """
        print ("==> Adding placeholders:")

        FLAGS = self.FLAGS
        self._reviews = tf.placeholder(
            dtype=tf.int32,
            shape=[None, FLAGS.max_input_length],
            name="reviews")
        self._review_lens = tf.placeholder(
            dtype=tf.int32,
            shape=[None, ],
            name="review_lens")
        self._labels = tf.placeholder(
            dtype=tf.int32,
            shape=[None,],
            name="labels")
        self._embeddings = tf.placeholder(
            dtype=tf.float32,
            shape=(FLAGS.vocab_size, FLAGS.emb_size),
            name='glove_embeddings')
        self._keep_prob = tf.placeholder(
            dtype=tf.float32,
            shape=(),
            name="keep_prob")

        print ("\t self._reviews:", self._reviews.get_shape())
        print ("\t self._labels:", self._labels.get_shape())
        print ("\t self._embeddings:", self._embeddings.get_shape())
        print ("\t self._keep_prob:", self._keep_prob.get_shape())

    def _build_encoder(self):
        """
        Constructing the encoder.
        """
        print ("==> Building the encoder:")

        FLAGS = self.FLAGS
        batch_size = FLAGS.batch_size
        hidden_size = FLAGS.hidden_size
        max_input_length = FLAGS.max_input_length

        with tf.variable_scope('embedding'):
            print ("\t embedding:")

            if FLAGS.embedding == 'random':
                # Random embedding weights
                embedding = tf.get_variable(
                    name='embedding',
                    shape=[self._vsize, FLAGS.emb_size],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=1e-4),
                    trainable=FLAGS.train_embedding)
            elif FLAGS.embedding == 'glove':
                # GloVe embedding weights
                embedding = self._embeddings

            # Check embedding dim
            if embedding.get_shape()[1] != FLAGS.emb_size:
                raise Exception(
                    "Embedding's dimension does not match specified emb_size.")

            # Embedding the review
            fn = lambda x: tf.nn.embedding_lookup(embedding, x)
            c_embedding = tf.map_fn(
                lambda x: fn(x), self._reviews, dtype=tf.float32)

            print ("\t\t embedding:", embedding.get_shape())
            print ("\t\t reviews_embedded:", c_embedding.get_shape())

        with tf.variable_scope('c_encoding'):
            print ("\t c_encoding:")

            # GRU cells
            cell = add_dropout_and_layers(
                single_cell=custom_GRUCell(hidden_size),
                keep_prob=self._keep_prob,
                num_layers=FLAGS.num_layers,
                )

            # Dynamic-GRU
            # return (outputs, last_output_states (relevant))
            all_outputs, h = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=c_embedding,
                #sequence_length=self._review_lens,
                dtype=tf.float32,
                time_major=False,
                )

            self._all_outputs = all_outputs
            self._h = h

            self._z = all_outputs

            print ("\t\t self._all_outputs", self._all_outputs.get_shape())
            print ("\t\t self._h", self._h.get_shape())

    def _build_attentional_interface(self):
        """
        Adding an attentional interface
        for model interpretability.
        """
        print ("==> Building the attentional interface:")

        FLAGS = self.FLAGS
        batch_size = FLAGS.batch_size
        hidden_size = FLAGS.hidden_size
        max_input_length = FLAGS.max_input_length
        loop_until = tf.to_int32(np.array(range(batch_size)))

        with tf.variable_scope('attention') as attn_scope:
            print ("\t attention:")

            # Time-major self._all_outputs (N, M, H) --> (M, N, H)
            all_outputs_time_major = tf.transpose(self._all_outputs,
                perm=[1,0,2])

            # Apply tanh nonlinearity
            fn = lambda _input: tf.nn.tanh(_linear(
                    args=_input,
                    output_size=hidden_size,
                    bias=True,
                    bias_start=0.0,
                    nonlinearity='tanh',
                    scope=attn_scope,
                    name='attn_nonlinearity',
                    ))
            z = tf.map_fn(
                lambda x: fn(x), all_outputs_time_major, dtype=tf.float32)

            # Apply softmax weights
            fn = lambda _input: tf.nn.tanh(_linear(
                    args=_input,
                    output_size=1,
                    bias=True,
                    bias_start=0.0,
                    nonlinearity='tanh',
                    scope=attn_scope,
                    name='attn_softmax',
                    ))
            z = tf.map_fn(
                lambda x: fn(x), z, dtype=tf.float32)

            # Squeeze and convert to batch major
            z = tf.transpose(
                    tf.squeeze(
                        input=z,
                        axis=2,
                        ),
                    perm=[1,0])

            # Normalize
            self._Z = tf.nn.softmax(
                logits=z,
                )

            # Create context vector (via soft attention.)
            fn = lambda sample_num: \
                tf.reduce_sum(
                    tf.multiply(
                        self._all_outputs[sample_num][:self._review_lens[sample_num]],

                        # (500,) --> (500, 1) --> (500, 200)
                        tf.tile(
                            input=tf.expand_dims(
                                self._Z[sample_num][:self._review_lens[sample_num]], 1),
                            multiples=(1, hidden_size),
                        )),
                    axis=0)

            self._c = tf.map_fn(
                lambda sample_num: fn(sample_num), loop_until, dtype=tf.float32)

            print ("\t\t self._Z", self._Z.get_shape())
            print ("\t\t self._c", self._c.get_shape())

    def _build_decoder(self):
        """
        Applying a softmax on output of encoder.
        """
        print ("==> Building the decoder:")
        with tf.variable_scope('softmax'):
            print ("\t Softmax:")
            self._logits = _linear(
                args=self._c, # self._c (with attn) or self._h (no attn)
                output_size=self.FLAGS.num_classes,
                bias=True,
                bias_start=0.0,
                nonlinearity='relu',
                name='softmax_op',
                )
            self._probabilities = tf.nn.softmax(
                logits=self._logits,
                )
            print ("\t\t self._logits", self._logits.get_shape())
            print ("\t\t self._probabilities", self._probabilities.get_shape())

    def _add_loss(self):
        """
        Determine the loss.
        """
        print ("==> Establishing the loss function.")
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels, logits=self._logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self._logits, 1),
            tf.cast(self._labels, tf.int64)), tf.float32))
        return self.loss, self.accuracy

    def _add_train_op(self):
        """
        Add the training optimizer.
        """
        print ("==> Creating the training optimizer.")

        # Decay learning rate
        self._lr = tf.maximum(
            self.FLAGS.min_lr,
            tf.train.exponential_decay(
                learning_rate=self.FLAGS.lr,
                global_step=self.global_step,
                decay_steps=100000,
                decay_rate=self.FLAGS.decay_rate,
                staircase=False,
                ))

        # Training releaved no clipping needed

        # Initialize the optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self._lr).minimize(self.loss,
            global_step=self.global_step)
        return self.optimizer

    def _build_graph(self):
        """
        Contrust each component of the TF graph.
        """
        self._add_placeholders()
        self._build_encoder()
        self._build_attentional_interface()
        self._build_decoder()

        self.global_step = tf.Variable(0, trainable=False) # won't step
        if self.FLAGS.mode == 'train':
            self._loss, self._accuracy = self._add_loss()
            self._train_op = self._add_train_op()

        # Components for model saving
        self.saver = tf.train.Saver(tf.global_variables())
        print ("==> Review Classifier built!")
