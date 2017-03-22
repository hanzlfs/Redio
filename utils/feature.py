###
# feature extractor method for wav files
###
import numpy as np
import librosa
import scipy as sp


def one_hot(labels):
    """
    Convert a list of labels to one hot code ndarray.
    Parameter: labels: list of integer(0-9)
    """
    n = len(labels)
    m = 10
    label_array = np.zeros((n, m), dtype = np.int)
    for i in range(n):
        label_array[i,labels[i]] = 1
    return label_array


def get_features(X, sample_rate = 22050):
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz
