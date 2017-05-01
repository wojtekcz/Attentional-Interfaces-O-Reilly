import pickle

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

