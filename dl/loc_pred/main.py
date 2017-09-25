import tensorflow as tf
from dl.loc_pred.seq4model import lm, classifier_seq
from keras.layers import Dot, Input, Lambda, multiply, dot
import numpy as np
from keras import backend as K

def sequence_mask(seq_len, max_seq_len):
    mask = np.zeros([len(seq_len), max_seq_len])
    for (b,v) in enumerate(seq_len):
        for i in range(v):
            mask[b, i] = 1
    return mask
def get_next_batch(batch_size):
    seq = np.random.randint(1, num_locations, size=[batch_size, max_seq_len])
    classes = np.random.randint(1, num_locations, size=[batch_size])
    seq_len = np.random.randint(2, max_seq_len, size=[batch_size])
    weight = sequence_mask(seq_len, max_seq_len)
    return seq, classes, seq_len, weight


PAD = 0
batch_size = 6
max_seq_len = 20
num_locations = 10
embedding_size = 50
hidden_units = embedding_size

seq_input = tf.placeholder(tf.int32, shape=[None, None], name='input_seq')
class_output = tf.placeholder(tf.int32, shape=[None], name='output_class')
seq_len = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
weight_mask = tf.placeholder(tf.float32, shape=[None, None], name='weight_mask')
loss, _ = classifier_seq(seq=seq_input, labels=class_output, weight_mask=weight_mask, num_loc=num_locations,embed_size=embedding_size, seq_len=seq_len)


with tf.Session() as sess:
    seq, classes, seq_length, weight = get_next_batch(batch_size)
    sess.run(tf.global_variables_initializer())
    loss_value = sess.run(loss, feed_dict={seq_input:seq, class_output:classes, seq_len:seq_length, weight_mask: weight})
    print(loss_value)




