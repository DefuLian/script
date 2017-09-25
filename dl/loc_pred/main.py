import tensorflow as tf
from dl.loc_pred.seq4model import lm
from keras.layers import Dot, Input, Lambda, multiply, dot
import numpy as np
from keras import backend as K
tf.reset_default_graph()
PAD = 0
EOS = 1
batch_size = 6
max_seq_len_val = 20
num_locations = 10
embedding_size = 50
hidden_units = embedding_size
x = np.random.randint(1, num_locations, size=(batch_size, 5))
#y = np.vstack((x.transpose(), [EOS] * batch_size))
#y = y.transpose()[:, 1:]
y = x.copy()
seq_len_val = np.array([5,15,6,11,9])

seq_mark = tf.sequence_mask(seq_len_val - 1, max_seq_len_val) ## leave the last one for prediction
weight = tf.cast(seq_mark, tf.float32)

weight_input = tf.placeholder(tf.int32, shape=(None,), name='example_weight')
train_input_seq = tf.placeholder(tf.int32, shape=(None, None), name='input_sequence')
train_output_seq = tf.placeholder(tf.int32, shape=(None, None), name='output_sequence')
test_input_seq = tf.placeholder(tf.int32, shape=(None, None), name='input_sequence')

loss, test_pred = lm(train_input_seq, train_output_seq, test_input_seq, num_locations, embedding_size, weight_input)
train_op = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(20):
        loss_val, _ = sess.run([loss, train_op],
                               feed_dict={train_input_seq: x, train_output_seq: y, weight_input:weight, test_input_seq: [x[:-1]]})
        print(loss_val)

