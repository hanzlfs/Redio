###
# feature extractor method for wav files
###
import numpy as np
import librosa
import scipy as sp

import torch
from torch.utils.serialization import load_lua


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

def stacked_feat(X, sr= 22050):
    mfccs,chroma,mel,contrast,tonnetz = get_features(X, sr)
    features = np.empty((0,193))
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features,ext_features])
    return features

def get_features(X, sample_rate = 44100):
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def load_torch_net(weight_path = '../weight/SoundNet.t7'):
    net = load_lua(weight_path)
    net.remove(25)
    net.remove(24)
    return net

def sound_net_features(input_data, sr = 44100, pretrained_weight_file = '../weight/SoundNet.t7'):
    # todo : yaml
    counter, time_segment, sound_length, layer = 1, 1, 10, 23
    length = len(input_data)
    scaler = max(max(input_data), abs(min(input_data)))
    snd = np.array(input_data*1.0/scaler).reshape(length, 1)
    snd = torch.from_numpy(snd).double()
    snd = snd.mul(2^-22)
    rep = int(np.ceil(((sound_length+1.0)*sr) / snd.size(0)))

    if snd.size(0) > sound_length*sr :
        snd = snd[0:sound_length*sr]
    snd = snd.repeat(rep,1)
    net = load_torch_net(pretrained_weight_file)
    net.forward(snd.view(1,1,-1,1))

    if layer != 24:
        feat = net.modules[23].output.float().squeeze()

    if counter == 1 :
        twindow_size = np.ceil((1.0/sound_length)*feat.size(1))
        sample_count = feat.size(1) - twindow_size + 1

    return feat
