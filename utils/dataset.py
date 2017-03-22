from feature import *
import librosa
import numpy as np

class FeatureExtractor(object):
    def __init__(self, file_dir = None, num_labels = 10):
        self.dataset = []]
        self.dir = file_dir
        self.nlabel = num_labels

    def feature_extractor(self, file_name):
        full_filepath = self.file_dir + '/' + file_name
        X, sr = librosa.load(full_filepath)
        mfccs,chroma,mel,contrast,tonnetz = get_feature(X, sr)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        stack_features = np.empty((0,193))
        stack_features = np.vstack([stack_features,ext_features])
        return stack_features

    #def pitch_shift_aug(self, )
