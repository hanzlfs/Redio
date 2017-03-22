import tensorflow as tf
import pandas as pd
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

feature_size = 193
targe_class = ['siren', 'air_conditionor', 'children', 'car_horn',
               'gun_shot', 'dog_bark',	'drilling',	'engine','hammer',
               'street']
def run_prediction(input_data, model_path = '../tmp/model.ckpt'):
    """
    input_data is a np.array
    """
    graph = tf.Graph()
    feature_pd = pd.DataFrame(input_data, columns = range(feature_size))
    with graph.as_default():
        _tdata = tf.placeholder(tf.float32, shape=[None, feature_size])
        _kprob = tf.placeholder(tf.float32)

        with tf.Session() as session :
            loader = tf.train.Saver()
            loader.restore(session, model_path)
            pred = session.run(test_prediction, feed_dict={tf_data : feature_pd, keep_prob : 1.0})
            return pred
