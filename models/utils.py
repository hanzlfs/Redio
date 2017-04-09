import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def test_accuracy(session, test_data=test, during = True):
    test_data.reset_index(inplace=True, drop=True)
    epoch_pred = session.run(test_prediction, feed_dict={tf_data : test_data.loc[0:check_size-1,data_cols], keep_prob : 1.0})
    for i in range(check_size, test_data.shape[0], check_size):
        epoch_pred = np.concatenate([epoch_pred, session.run(test_prediction,
                                    feed_dict={tf_data : test_data.loc[i:i+check_size-1,data_cols], keep_prob : 1.0})], axis=0)
    if during:
        return accuracy(epoch_pred, test_labels)
    else:
        return epoch_pred
