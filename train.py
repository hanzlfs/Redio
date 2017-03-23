import os, pickle, librosa
import soundfile as sf
import pandas as pd
import tensorflow as tf

from utils import feature as futils
from utils.dataset import FeatureExtractor

### extract features from UrbanSound8K

def features_extract_store(dir_name = 'Urban-Sound-Classification',
                          sub_folders = [], n_label = 10,
                          meta_data = None, mode = 'train',
                          persist = False, export_file = './datasets/193_features_'):
    raw_sound = pd.read_csv(meta_data)
    data_handler = FeatureExtractor(num_labels = n_label)
    for folder_id, folder in enumerate(sub_folders):
        folder_path = dir_name + folder + '/'
        files_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for file_name in files_list:
            try:
                file_name_path = folder_path + file_name
                features = data_handler.feature_extractor(file_name_path)
            except:
                print(file_name)
                continue
            list_row = raw_sound.loc[raw_sound['slice_file_name']==file_name].values.tolist()
            label = list_row[0][-1]
            data_handler.data.append([features, features.shape, label, folder_id + 1])
    data_handler.df = pd.DataFrame(data = data_handler.data , columns = ["features", "shape","label", "fold"])
    data_handler.label_mapping(labels = data_handler.df['label'], n_class = 10)
    print "dataframe shape", data_handler.df.shape
    if persist :
        pickle.dump(data_handler.df, open(export_file + mode + '.p','wb'))
    else :
        return data_handler.df

def train():
    return None

### train a two layer Conv NN and store

### restore and eval it

if __name__ == '__main__':
    us_folder = '/home/paperspace/Documents/Notebooks/Urban-Sound-Classification/'
    sub_folder_list = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']

    features_extract_store(dir_name = us_folder + 'UrbanSound8K/audio/',
                           sub_folders = sub_folder_list[:1],
                           meta_data = us_folder + 'UrbanSound8K/metadata/UrbanSound8K.csv')
    #print futils.get_features(X, sr)
    #main()
