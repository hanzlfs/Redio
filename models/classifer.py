import tensorflow as tf

class AudioNNModel(object):
    """
    an audio classifier with Conv Nerual Network
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
