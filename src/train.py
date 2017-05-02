import pickle
import tensorflow as tf
import numpy as np
import os

from src.vocab import (
    Vocab
)

from src.model import (
    Model,
)

from src.data import (
    generate_epoch
)


def create_model(sess, FLAGS, vocab_size, basedir):
    """
    Creates a new model or loads old one.
    """
    imdb_model = Model(FLAGS, vocab_size)
    imdb_model._build_graph()

    if FLAGS.model == 'new':
        print ('==> Created a new model.')
        sess.run(tf.global_variables_initializer())
    elif FLAGS.model == 'old':
        ckpt = tf.train.get_checkpoint_state(
            os.path.join(basedir, FLAGS.ckpt_dir))
        if ckpt and ckpt.model_checkpoint_path:
            print("==> Restoring old model parameters from %s" %
                ckpt.model_checkpoint_path)
            imdb_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print ("==> No old model to load from so initializing a new one.")
            sess.run(tf.global_variables_initializer())

    return imdb_model


def train(FLAGS, basedir):
    """
    Train a previous or new model.
    """
    # Data paths
    vocab_path = os.path.join(
        basedir, 'data/processed_reviews/vocab.txt')
    train_data_path = os.path.join(
        basedir, 'data/processed_reviews/train.p')
    validation_data_path = os.path.join(
        basedir, 'data/processed_reviews/validation.p')
    vocab = Vocab(vocab_path)
    FLAGS.num_classes = 2

    # Load embeddings (if using GloVe)
    if FLAGS.embedding == 'glove':
        with open(os.path.join(
            basedir, 'data/processed_reviews/embeddings.p'), 'rb') as f:
            embeddings = pickle.load(f)
        FLAGS.vocab_size = len(embeddings)

    # Start tensorflow session
    with tf.Session() as sess:

        # Create|reload model
        imdb_model = create_model(sess, FLAGS, len(vocab), basedir)

        # Metrics
        metrics = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": [],
        }

        # Store attention score history for few samples
        attn_history = {
            "sample_0":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_1":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_2":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_3":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
            "sample_4":
            {"review": None, "label": None, "review_len": None, "attn_scores": []},
        }

        # Start training
        for train_epoch_num, train_epoch in \
            enumerate(generate_epoch(
                train_data_path, FLAGS.num_epochs, FLAGS.batch_size)):

            print ("==> EPOCH:", train_epoch_num)

            for train_batch_num, (batch_features, batch_seq_lens) in \
                enumerate(train_epoch):

                batch_reviews, batch_labels = batch_features
                batch_review_lens, = batch_seq_lens

                # Display shapes once
                if (train_epoch_num == 0 and train_batch_num == 0):
                    print ("Reviews: ", np.shape(batch_reviews))
                    print ("Labels: ", np.shape(batch_labels))
                    print ("Review lens: ", np.shape(batch_review_lens))

                _, train_logits, train_loss, train_acc, lr, attn_scores = \
                    imdb_model.train(
                        sess=sess,
                        batch_reviews=batch_reviews,
                        batch_labels=batch_labels,
                        batch_review_lens=batch_review_lens,
                        embeddings=embeddings,
                        keep_prob=FLAGS.keep_prob,
                        )

            for valid_epoch_num, valid_epoch in \
                enumerate(generate_epoch(
                    data_path=validation_data_path,
                    num_epochs=1,
                    batch_size=FLAGS.batch_size,
                    )):

                for valid_batch_num, (valid_batch_features, valid_batch_seq_lens) in \
                    enumerate(valid_epoch):

                    valid_batch_reviews, valid_batch_labels = valid_batch_features
                    valid_batch_review_lens, = valid_batch_seq_lens

                    valid_logits, valid_loss, valid_acc = imdb_model.eval(
                        sess=sess,
                        batch_reviews=valid_batch_reviews,
                        batch_labels=valid_batch_labels,
                        batch_review_lens=valid_batch_review_lens,
                        embeddings=embeddings,
                        keep_prob=1.0, # no dropout for val|test
                        )

            print ("[EPOCH]: %i, [LR]: %.6e, [TRAIN ACC]: %.3f, [VALID ACC]: %.3f " \
                   "[TRAIN LOSS]: %.6f, [VALID LOSS]: %.6f" % (
                train_epoch_num, lr, train_acc, valid_acc, train_loss, valid_loss))

            # Store the metrics
            metrics["train_loss"].append(train_loss)
            metrics["valid_loss"].append(valid_loss)
            metrics["train_acc"].append(train_acc)
            metrics["valid_acc"].append(valid_acc)

            # Store attn history
            for i in range(5):
                sample = "sample_%i"%i
                attn_history[sample]["review"] = batch_reviews[i]
                attn_history[sample]["label"] = batch_labels[i]
                attn_history[sample]["review_len"] = batch_review_lens[i]
                attn_history[sample]["attn_scores"].append(attn_scores[i])

            # Save the model (maybe)
            if ((train_epoch_num == (FLAGS.num_epochs-1)) or
            ((train_epoch_num%FLAGS.save_every == 0) and (train_epoch_num>0))):

                # Make parents ckpt dir if it does not exist
                if not os.path.isdir(os.path.join(basedir, FLAGS.data_dir, 'ckpt')):
                    os.makedirs(os.path.join(basedir, FLAGS.data_dir, 'ckpt'))

                # Make child ckpt dir for this specific model
                if not os.path.isdir(os.path.join(basedir, FLAGS.ckpt_dir)):
                    os.makedirs(os.path.join(basedir, FLAGS.ckpt_dir))

                checkpoint_path = \
                    os.path.join(
                        basedir, FLAGS.ckpt_dir, "%s.ckpt" % FLAGS.model_name)

                print ("==> Saving the model.")
                imdb_model.saver.save(sess, checkpoint_path,
                                 global_step=imdb_model.global_step)

    # Save the metrics
    metrics_file = os.path.join(basedir, FLAGS.ckpt_dir, 'metrics.p')
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    # Save the attention scores
    attn_history_file = os.path.join(basedir, FLAGS.ckpt_dir, 'attn_history.p')
    with open(attn_history_file, 'wb') as f:
        pickle.dump(attn_history, f)
