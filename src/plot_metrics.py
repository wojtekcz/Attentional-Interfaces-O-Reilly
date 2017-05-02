import os
import pickle
import matplotlib.pyplot as plt


def plot_metrics(FLAGS, basedir):
    """
    Plot the loss and accuracy for train|test.
    """

    # Load metrics from file
    metrics_file = os.path.join(basedir, FLAGS.ckpt_dir, 'metrics.p')
    with open(metrics_file, 'rb') as f:
        metrics = pickle.load(f)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Plot results
    ax1 = axes[0]
    ax1.plot(metrics["train_acc"], label='train accuracy')
    ax1.plot(metrics["valid_acc"], label='valid accuracy')
    ax1.legend(loc=4)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('train|valid accuracy')

    ax2 = axes[1]
    ax2.plot(metrics["train_loss"], label='train loss')
    ax2.plot(metrics["valid_loss"], label='valid loss')
    ax2.legend(loc=3)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('train|valid loss')

    plt.show()
