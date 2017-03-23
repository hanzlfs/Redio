from feature import *
import librosa
import numpy as np
import pandas as pd

class FeatureExtractor(object):
    def __init__(self, num_labels = 10):
        self.data = []
        self.df = None
        self.nlabel = num_labels

    def feature_extractor(self, full_filepath):
        X, sr = librosa.load(full_filepath)
        mfccs,chroma,mel,contrast,tonnetz = get_features(X, sr)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        stack_features = np.empty((0,193))
        stack_features = np.vstack([stack_features,ext_features])
        return stack_features

    def label_mapping(self, labels, n_class = 10):
        classes = list(set(labels))
        label_list = [(i, c) for i, c in enumerate(classes)]
        label_dict = dict(label_list)
        label_num = []
        for l in labels:
            for i in range(n_class):
                if label == label_dict[i]
                label_num.append(i)
        label_ohe = np.array(one_hot(label_num))
        for i in range(n_class):
            self.df[label_dict[i][:8]] = label_ohe[:,i]
        ll = [self.df['features'][i].ravel() for i in range(self.df.shape[0])]
        self.df['sample'] = pd.Series(ll, index = self.df.index)

    #def pitch_shift_aug(self, )
