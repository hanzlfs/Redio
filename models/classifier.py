import tensorflow as tf

class AudioNNModel(object):
    """
    an audio classifier with Conv Nerual Network
    """
    def __init__(self, batch_size, n_labels = 10, feature_size, beta):
        self.batch_size = batch_size
        self.n_labels = n_labels
        self.acc_over_time = {}
        self.feature_size = feature_size
        self.beta = beta

    def restore(ckpt_path = None):
        return None
