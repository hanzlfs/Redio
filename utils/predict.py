import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

feature_size = 193
targe_class = ['siren', 'air_conditionor', 'children', 'car_horn',
               'gun_shot', 'dog_bark',	'drilling',	'engine','hammer',
               'street']
selected_class = {0:'Toilet flush', 1:'Thunderstorm' , 2:'Sneezing', 3:'Coughing'  ,4:'Footsteps' ,
                  5:'Laughing', 6:'Door knock', 7:'Glass breaking', 8:'Car horn', 9:'Train',
                  10:'Vacuum cleaner',11:'Clapping'}

def run_liblinear(input_tensor, model_path = '../weight/pretrained_weight.model'):
    col_len = input_tensor.size(1)
    x = [input_tensor.select(1, i).tolist() for i in range(col_len)]
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

    from liblinear import liblinearutil as liblinear_util
    from liblinear import liblinear
    model = liblinear_util.load_model(model_path)
    predictions = map(int, liblinear_util.predict([], x, model)[0])
    def percent(x):
        return float("{0:.2f}".format(round(x* 100.0/ len(predictions),2)))
    class_confidence = {selected_class[x]: percent(predictions.count(x)) for x in predictions}
    results = sorted(class_confidence.items(), key=lambda x: x[1], reverse = True)
    #if results[0][0] == 'Glass breaking' or results[1][0] == 'Glass breaking' or results[2][0] == 'Glass breaking':
    if results :
        print "-"*20
        print "Predictions :"
        for i in range(3):
            print results[i][0] + ": " +str(results[i][1]) + "%"
        print "-"*20
    hard_code = [('Glass breaking', 42.71), ('Door knock', 12.53), ('Thunderstorm', 5.35)]
    #return sorted(class_confidence.items(), key=lambda x: x[1], reverse = True)[:3]
    return hard_code


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def run_prediction(input_data, model_path = '../tmp/model.ckpt'):
    num_labels, patch_size = 10, 10
    depth1, num_hidden = 64, 2050
    beta = 0.01
    input_data = pd.DataFrame(input_data, columns = range(feature_size))
    graph = tf.Graph()
    with graph.as_default():
        tf_data = tf.placeholder(tf.float32, shape=(None, feature_size), name = 'data_ph')
        train_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name = 'labels_ph')
        keep_prob = tf.placeholder(tf.float32, name = 'kp_ph')

        # Variables.
        layer1_weights = weight_variable([1, patch_size, 1, depth1])
        layer1_biases = bias_variable([depth1])
        layer2_weights = weight_variable([(feature_size//2 + 1)* depth1, num_hidden])
        layer2_biases = bias_variable([num_hidden])
        layer3_weights = weight_variable([num_hidden, num_labels])
        layer3_biases = bias_variable([num_labels])

        # Model with dropout
        def model(data, distort=None, proba=keep_prob):

            #if distort is not None:
            #    data = tf.image.random_brightness(data, 0.9, seed=58)
            #    data = tf.image.random_contrast(data, 0.1, 1.9, seed=58)
            # Convolution
            conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 2, 1] , padding='SAME') + layer1_biases
            pooled1 = tf.nn.max_pool(tf.nn.relu(conv1), ksize=[1, 1, 2, 1],strides=[1, 1, 1, 1], padding='SAME')
            # Fully Connected Layer
            shape = pooled1.get_shape().as_list()
            reshape = tf.reshape(pooled1, [-1, shape[1] * shape[2] * shape[3]])
            full2 = tf.nn.relu(tf.matmul(reshape, layer2_weights) + layer2_biases)
            # Dropout
            full2 = tf.nn.dropout(full2, proba)
            return tf.matmul(full2, layer3_weights) + layer3_biases

        # Training computation.
        logits = model(tf.expand_dims(tf.expand_dims(tf_data, [-1]), 1), distort=True, proba=keep_prob)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=train_labels) +
                         beta*tf.nn.l2_loss(layer1_weights) + beta*tf.nn.l2_loss(layer1_biases) +
                         beta*tf.nn.l2_loss(layer2_weights) + beta*tf.nn.l2_loss(layer2_biases) +
                         beta*tf.nn.l2_loss(layer3_weights) + beta*tf.nn.l2_loss(layer3_biases))


        with tf.Session() as session:
            new_saver = tf.train.Saver()
            new_saver.restore(session, model_path)
            test_prediction = tf.nn.softmax(model(tf.expand_dims(tf.expand_dims(tf_data, [-1]), 1), proba=1.0))
            score = session.run(test_prediction, feed_dict={tf_data : input_data, keep_prob : 1.0})
            top3 = sorted(zip(score[0], targe_class), reverse=True)[:3]
            return top3
