import librosa
import soundfile as sf

from utils import feature as futils
from utils.dataset import FeatureExtractor

### extract features from UrbanSound8K

def features_extract_store(folder_name = 'Urban-Sound-Classification',
                          sub_folders = [],
                          n_label = 10,
                          meta_data = None,
                          mode = 'train',
                          persist = True):

    data_handler = FeatureExtractor(file_dir = folder_name, num_labels = n_label)


    if persist :
        pickle.dump(mfcc_pd,open('./data/193_features_'+mode+'.p','wb'))



### train a two layer Conv NN and store

### restore and eval it

if __name__ == '__main__':
    us_folder = '/home/paperspace/Documents/Notebooks/Urban-Sound-Classification/'
    sub_folder_list = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
    futils.get_features
    main()
